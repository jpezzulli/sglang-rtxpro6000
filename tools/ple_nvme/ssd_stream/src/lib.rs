#[cfg(not(target_os = "linux"))]
compile_error!("sglang-ssd-stream requires Linux io_uring");

use io_uring::{opcode, types, IoUring};
use numpy::{PyReadonlyArray1, PyReadwriteArray2, PyUntypedArrayMethods};
use pyo3::exceptions::{PyOSError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use std::cmp::min;
use std::fs::{self, File, OpenOptions};
use std::io;
use std::os::fd::AsRawFd;
use std::path::{Path, PathBuf};
use std::ptr::NonNull;
use std::slice;
use std::sync::Mutex;
use std::time::Instant;

const PAGE_SIZE: usize = 4096;
const DEFAULT_QUEUE_DEPTH: u32 = 256;
const DEFAULT_PAGE_POOL_MIB: usize = 32;
const DEFAULT_MAX_BATCH_PAGES: usize = 4096;

struct AlignedPool {
    address: NonNull<u8>,
    len: usize,
}

impl AlignedPool {
    fn new(len: usize) -> io::Result<Self> {
        let raw = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        if raw == libc::MAP_FAILED {
            return Err(io::Error::last_os_error());
        }
        let address = NonNull::new(raw.cast::<u8>()).ok_or_else(|| {
            io::Error::new(io::ErrorKind::OutOfMemory, "mmap returned a null pointer")
        })?;
        unsafe {
            libc::madvise(raw, len, libc::MADV_DONTDUMP);
        }
        Ok(Self { address, len })
    }

    fn page_ptr(&self, page_slot: usize) -> *mut u8 {
        debug_assert!((page_slot + 1) * PAGE_SIZE <= self.len);
        unsafe { self.address.as_ptr().add(page_slot * PAGE_SIZE) }
    }
}

// The pool is only accessed while its owning ReaderState mutex is held. The
// kernel registration is removed before the mapping is released.
unsafe impl Send for AlignedPool {}

impl Drop for AlignedPool {
    fn drop(&mut self) {
        unsafe {
            libc::munmap(self.address.as_ptr().cast(), self.len);
        }
    }
}

#[derive(Clone, Copy)]
struct Piece {
    page_id: u64,
    output_offset: u32,
    page_offset: u16,
    len: u16,
}

#[derive(Clone, Copy)]
struct PageGroup {
    page_id: u64,
    piece_start: u32,
    piece_end: u32,
}

#[pyclass(frozen, skip_from_py_object, module = "sglang_ssd_stream._io")]
#[derive(Clone, Default)]
struct ReadStats {
    #[pyo3(get)]
    rows: usize,
    #[pyo3(get)]
    valid_rows: usize,
    #[pyo3(get)]
    unique_pages: usize,
    #[pyo3(get)]
    submitted_bytes: u64,
    #[pyo3(get)]
    physical_bytes: Option<u64>,
    #[pyo3(get)]
    peak_queue_depth: usize,
    #[pyo3(get)]
    read_batches: usize,
    #[pyo3(get)]
    io_ns: u64,
    #[pyo3(get)]
    scatter_ns: u64,
    #[pyo3(get)]
    completion_latency_mean_ns: u64,
    #[pyo3(get)]
    completion_latency_max_ns: u64,
    #[pyo3(get)]
    total_ns: u64,
}

#[pymethods]
impl ReadStats {
    fn __repr__(&self) -> String {
        format!(
            "ReadStats(rows={}, unique_pages={}, physical_bytes={:?}, io_ns={}, total_ns={})",
            self.rows, self.unique_pages, self.physical_bytes, self.io_ns, self.total_ns
        )
    }
}

struct ReaderState {
    file: File,
    file_len: u64,
    row_bytes: usize,
    file_row_start: i64,
    tp_start: i64,
    tp_end: i64,
    ring: IoUring,
    pool: AlignedPool,
    registered_buffers: bool,
    queue_depth: usize,
    max_batch_pages: usize,
    measure_physical_io: bool,
    pieces: Vec<Piece>,
    groups: Vec<PageGroup>,
    broken: Option<String>,
}

impl ReaderState {
    fn new(
        path: &Path,
        row_bytes: usize,
        file_row_start: i64,
        tp_start: i64,
        tp_end: i64,
        measure_physical_io: bool,
    ) -> io::Result<Self> {
        if row_bytes == 0 {
            return Err(invalid_input("row_bytes must be positive"));
        }
        if file_row_start < 0 || tp_start < file_row_start || tp_end <= tp_start {
            return Err(invalid_input("tp_start and tp_end form an invalid range"));
        }
        let pool_len = DEFAULT_PAGE_POOL_MIB
            .checked_mul(1024 * 1024)
            .ok_or_else(|| invalid_input("page pool size overflow"))?;

        let file = OpenOptions::new().read(true).open(path)?;
        let file_len = file.metadata()?.len();
        let required_rows = u64::try_from(tp_end - file_row_start)
            .map_err(|_| invalid_input("PLE row range overflow"))?;
        let required_len = required_rows
            .checked_mul(row_bytes as u64)
            .ok_or_else(|| invalid_input("table byte count overflow"))?;
        if file_len < required_len {
            return Err(invalid_input(format!(
                "PLE table is too short: expected at least {required_len} bytes, found {file_len}"
            )));
        }
        let advise_result =
            unsafe { libc::posix_fadvise(file.as_raw_fd(), 0, 0, libc::POSIX_FADV_RANDOM) };
        if advise_result != 0 {
            return Err(io::Error::from_raw_os_error(advise_result));
        }

        let ring = IoUring::new(DEFAULT_QUEUE_DEPTH)?;
        let pool = AlignedPool::new(pool_len)?;
        let registered = [libc::iovec {
            iov_base: pool.address.as_ptr().cast(),
            iov_len: pool.len,
        }];
        let registered_buffers = match unsafe { ring.submitter().register_buffers(&registered) } {
            Ok(()) => true,
            Err(error)
                if matches!(
                    error.raw_os_error(),
                    Some(code)
                        if code == libc::ENOMEM || code == libc::EPERM || code == libc::EAGAIN
                ) =>
            {
                false
            }
            Err(error) => {
                return Err(io::Error::new(
                    error.kind(),
                    format!(
                        "failed to register the {DEFAULT_PAGE_POOL_MIB} MiB io_uring page pool: {error}"
                    ),
                ));
            }
        };

        Ok(Self {
            file,
            file_len,
            row_bytes,
            file_row_start,
            tp_start,
            tp_end,
            ring,
            pool,
            registered_buffers,
            queue_depth: DEFAULT_QUEUE_DEPTH as usize,
            max_batch_pages: DEFAULT_MAX_BATCH_PAGES,
            measure_physical_io,
            pieces: Vec::new(),
            groups: Vec::new(),
            broken: None,
        })
    }

    fn gather(&mut self, row_ids: &[i64], output: &mut [u8]) -> io::Result<ReadStats> {
        if let Some(reason) = &self.broken {
            return Err(io::Error::other(format!(
                "PLE page reader is unusable after an earlier I/O failure: {reason}"
            )));
        }
        let started = Instant::now();
        let expected_output_len = row_ids
            .len()
            .checked_mul(self.row_bytes)
            .ok_or_else(|| invalid_input("output byte count overflow"))?;
        if output.len() != expected_output_len {
            return Err(invalid_input(format!(
                "output has {} bytes, expected {expected_output_len}",
                output.len()
            )));
        }
        if output.len() > u32::MAX as usize {
            return Err(invalid_input("one gather cannot exceed 4 GiB"));
        }
        output.fill(0);
        self.pieces.clear();
        self.groups.clear();

        let mut valid_rows = 0usize;
        for (row_index, &global_id) in row_ids.iter().enumerate() {
            if global_id < self.tp_start || global_id >= self.tp_end {
                continue;
            }
            valid_rows += 1;
            let file_row = (global_id - self.file_row_start) as u64;
            let row_start = file_row
                .checked_mul(self.row_bytes as u64)
                .ok_or_else(|| invalid_input("row offset overflow"))?;
            let row_end = row_start
                .checked_add(self.row_bytes as u64)
                .ok_or_else(|| invalid_input("row end overflow"))?;
            if row_end > self.file_len {
                return Err(invalid_input(format!(
                    "row {global_id} extends past the PLE table"
                )));
            }

            let output_row_start = row_index
                .checked_mul(self.row_bytes)
                .ok_or_else(|| invalid_input("output row offset overflow"))?;
            let mut file_offset = row_start;
            let mut row_offset = 0usize;
            while row_offset < self.row_bytes {
                let page_offset = (file_offset as usize) & (PAGE_SIZE - 1);
                let piece_len = min(PAGE_SIZE - page_offset, self.row_bytes - row_offset);
                self.pieces.push(Piece {
                    page_id: file_offset / PAGE_SIZE as u64,
                    output_offset: u32::try_from(output_row_start + row_offset)
                        .map_err(|_| invalid_input("output offset overflow"))?,
                    page_offset: page_offset as u16,
                    len: piece_len as u16,
                });
                file_offset += piece_len as u64;
                row_offset += piece_len;
            }
        }

        self.pieces.sort_unstable_by_key(|piece| piece.page_id);
        let mut piece_start = 0usize;
        while piece_start < self.pieces.len() {
            let page_id = self.pieces[piece_start].page_id;
            let mut piece_end = piece_start + 1;
            while piece_end < self.pieces.len() && self.pieces[piece_end].page_id == page_id {
                piece_end += 1;
            }
            self.groups.push(PageGroup {
                page_id,
                piece_start: piece_start as u32,
                piece_end: piece_end as u32,
            });
            piece_start = piece_end;
        }

        let physical_before = self.measure_physical_io.then(process_read_bytes).flatten();
        let mut stats = ReadStats {
            rows: row_ids.len(),
            valid_rows,
            unique_pages: self.groups.len(),
            submitted_bytes: (self.groups.len() * PAGE_SIZE) as u64,
            ..ReadStats::default()
        };
        let mut latency_sum_ns = 0u128;
        let mut completions = 0usize;

        for batch_start in (0..self.groups.len()).step_by(self.max_batch_pages) {
            let batch_end = min(batch_start + self.max_batch_pages, self.groups.len());
            if let Err(error) = self.read_page_batch(
                batch_start,
                batch_end,
                &mut stats,
                &mut latency_sum_ns,
                &mut completions,
            ) {
                self.broken = Some(error.to_string());
                return Err(error);
            }

            let scatter_started = Instant::now();
            for group_index in batch_start..batch_end {
                let group = self.groups[group_index];
                let page_slot = group_index - batch_start;
                let page = self.pool.page_ptr(page_slot);
                for piece in &self.pieces[group.piece_start as usize..group.piece_end as usize] {
                    let source = unsafe { page.add(piece.page_offset as usize) };
                    let destination =
                        unsafe { output.as_mut_ptr().add(piece.output_offset as usize) };
                    unsafe {
                        std::ptr::copy_nonoverlapping(source, destination, piece.len as usize);
                    }
                }
            }
            stats.scatter_ns = stats
                .scatter_ns
                .saturating_add(nanos_u64(scatter_started.elapsed().as_nanos()));
        }

        stats.physical_bytes = if self.measure_physical_io {
            match (physical_before, process_read_bytes()) {
                (Some(before), Some(after)) => Some(after.saturating_sub(before)),
                _ => None,
            }
        } else {
            None
        };
        if completions > 0 {
            stats.completion_latency_mean_ns = nanos_u64(latency_sum_ns / completions as u128);
        }
        stats.total_ns = nanos_u64(started.elapsed().as_nanos());
        Ok(stats)
    }

    fn read_page_batch(
        &mut self,
        batch_start: usize,
        batch_end: usize,
        stats: &mut ReadStats,
        latency_sum_ns: &mut u128,
        completions: &mut usize,
    ) -> io::Result<()> {
        let batch_len = batch_end - batch_start;
        for wave_start in (0..batch_len).step_by(self.queue_depth) {
            let wave_end = min(wave_start + self.queue_depth, batch_len);
            let wave_len = wave_end - wave_start;
            stats.peak_queue_depth = stats.peak_queue_depth.max(wave_len);
            stats.read_batches += 1;

            {
                let mut submission = self.ring.submission();
                for page_slot in wave_start..wave_end {
                    let group = self.groups[batch_start + page_slot];
                    let entry = if self.registered_buffers {
                        opcode::ReadFixed::new(
                            types::Fd(self.file.as_raw_fd()),
                            self.pool.page_ptr(page_slot),
                            PAGE_SIZE as u32,
                            0,
                        )
                        .offset(group.page_id * PAGE_SIZE as u64)
                        .build()
                    } else {
                        opcode::Read::new(
                            types::Fd(self.file.as_raw_fd()),
                            self.pool.page_ptr(page_slot),
                            PAGE_SIZE as u32,
                        )
                        .offset(group.page_id * PAGE_SIZE as u64)
                        .build()
                    }
                    .user_data(page_slot as u64);
                    unsafe {
                        submission
                            .push(&entry)
                            .map_err(|_| io::Error::other("io_uring submission queue is full"))?;
                    }
                }
            }

            let io_started = Instant::now();
            self.ring.submit_and_wait(wave_len)?;
            let mut wave_completions = 0usize;
            while wave_completions < wave_len {
                let mut drained = 0usize;
                {
                    let mut completion = self.ring.completion();
                    for entry in &mut completion {
                        let page_slot = entry.user_data() as usize;
                        if page_slot < wave_start || page_slot >= wave_end {
                            return Err(io::Error::other("io_uring returned an unknown page slot"));
                        }
                        let result = entry.result();
                        if result < 0 {
                            return Err(io::Error::from_raw_os_error(-result));
                        }
                        let page_offset =
                            self.groups[batch_start + page_slot].page_id * PAGE_SIZE as u64;
                        let expected = min(PAGE_SIZE as u64, self.file_len - page_offset) as usize;
                        if result as usize != expected {
                            return Err(io::Error::new(
                                io::ErrorKind::UnexpectedEof,
                                format!(
                                    "PLE page read returned {result} bytes, expected {expected}"
                                ),
                            ));
                        }
                        let latency = io_started.elapsed().as_nanos();
                        *latency_sum_ns += latency;
                        stats.completion_latency_max_ns =
                            stats.completion_latency_max_ns.max(nanos_u64(latency));
                        drained += 1;
                    }
                }
                wave_completions += drained;
                *completions += drained;
                if wave_completions < wave_len {
                    self.ring.submit_and_wait(wave_len - wave_completions)?;
                }
            }
            stats.io_ns = stats
                .io_ns
                .saturating_add(nanos_u64(io_started.elapsed().as_nanos()));
        }
        Ok(())
    }
}

impl Drop for ReaderState {
    fn drop(&mut self) {
        if self.registered_buffers {
            let _ = self.ring.submitter().unregister_buffers();
        }
    }
}

#[pyclass(module = "sglang_ssd_stream._io")]
struct PageReader {
    state: Mutex<ReaderState>,
    row_bytes: usize,
    #[pyo3(get)]
    registered_buffers: bool,
}

#[pymethods]
impl PageReader {
    #[new]
    #[pyo3(signature = (
        path,
        row_bytes,
        file_row_start,
        tp_start,
        tp_end,
        measure_physical_io = false,
    ))]
    fn new(
        path: PathBuf,
        row_bytes: usize,
        file_row_start: i64,
        tp_start: i64,
        tp_end: i64,
        measure_physical_io: bool,
    ) -> PyResult<Self> {
        let state = ReaderState::new(
            &path,
            row_bytes,
            file_row_start,
            tp_start,
            tp_end,
            measure_physical_io,
        )
        .map_err(os_error)?;
        let registered_buffers = state.registered_buffers;
        Ok(Self {
            state: Mutex::new(state),
            row_bytes,
            registered_buffers,
        })
    }

    fn gather(
        &self,
        py: Python<'_>,
        row_ids: PyReadonlyArray1<'_, i64>,
        mut output: PyReadwriteArray2<'_, u8>,
    ) -> PyResult<ReadStats> {
        let ids = row_ids
            .as_slice()
            .map_err(|_| PyValueError::new_err("row_ids must be a contiguous int64 array"))?;
        let shape = output.shape();
        if shape.len() != 2 || shape[0] != ids.len() {
            return Err(PyValueError::new_err(
                "output must be a contiguous uint8 array with one row per ID",
            ));
        }
        if shape[1] != self.row_bytes {
            return Err(PyValueError::new_err(format!(
                "output row width is {}, expected {}",
                shape[1], self.row_bytes
            )));
        }
        let output_slice = output.as_slice_mut().map_err(|_| {
            PyValueError::new_err("output must be a contiguous writable uint8 array")
        })?;

        let ids_address = ids.as_ptr() as usize;
        let ids_len = ids.len();
        let output_address = output_slice.as_mut_ptr() as usize;
        let output_len = output_slice.len();
        let state = &self.state;
        py.detach(move || {
            // The NumPy borrow guards remain alive for the entire detached call.
            let ids = unsafe { slice::from_raw_parts(ids_address as *const i64, ids_len) };
            let output =
                unsafe { slice::from_raw_parts_mut(output_address as *mut u8, output_len) };
            let mut state = state
                .lock()
                .map_err(|_| PyRuntimeError::new_err("PLE page reader mutex is poisoned"))?;
            state.gather(ids, output).map_err(os_error)
        })
    }
}

fn process_read_bytes() -> Option<u64> {
    let contents = fs::read_to_string("/proc/self/io").ok()?;
    contents.lines().find_map(|line| {
        let value = line.strip_prefix("read_bytes:")?.trim();
        value.parse().ok()
    })
}

fn invalid_input(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message.into())
}

fn nanos_u64(value: u128) -> u64 {
    value.min(u64::MAX as u128) as u64
}

fn os_error(error: io::Error) -> PyErr {
    PyOSError::new_err(error.to_string())
}

#[pymodule]
fn _io(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("PAGE_SIZE", PAGE_SIZE)?;
    module.add_class::<PageReader>()?;
    module.add_class::<ReadStats>()?;
    Ok(())
}
