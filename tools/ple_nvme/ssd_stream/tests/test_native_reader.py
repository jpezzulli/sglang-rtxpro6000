import gc
import hashlib
import os

import numpy as np
import pytest
from sglang_ssd_stream._io import PageReader

ROW_BYTES = 160
TP_START = 100
TP_END = 164


def _table(path):
    rows = np.arange((TP_END - TP_START) * ROW_BYTES, dtype=np.uint64)
    rows = ((rows * 29 + 7) % 256).astype(np.uint8).reshape(-1, ROW_BYTES)
    path.write_bytes(rows.tobytes())
    return rows


def _expected(rows, ids):
    result = np.zeros((len(ids), ROW_BYTES), dtype=np.uint8)
    valid = (ids >= TP_START) & (ids < TP_END)
    result[valid] = rows[ids[valid] - TP_START]
    return result


def _reader(path):
    return PageReader(
        path,
        ROW_BYTES,
        TP_START,
        TP_START,
        TP_END,
        measure_physical_io=True,
    )


def test_native_reader_preserves_rows_and_files(tmp_path):
    table_path = tmp_path / "table.bin"
    marker_path = tmp_path / "table.bin.complete"
    rows = _table(table_path)
    marker_path.write_text("immutable marker\n", encoding="ascii")
    ids = np.array(
        [TP_START, TP_START + 25, TP_START, TP_END - 1, TP_START - 1, TP_END],
        dtype=np.int64,
    )
    output = np.full((ids.size, ROW_BYTES), 0xA5, dtype=np.uint8)
    table_hash = hashlib.sha256(table_path.read_bytes()).digest()
    marker_hash = hashlib.sha256(marker_path.read_bytes()).digest()

    reader = _reader(table_path)
    assert isinstance(reader.registered_buffers, bool)
    stats = reader.gather(ids, output)

    np.testing.assert_array_equal(output, _expected(rows, ids))
    assert stats.rows == ids.size
    assert stats.valid_rows == 4
    assert stats.unique_pages > 0
    assert stats.submitted_bytes == stats.unique_pages * 4096
    assert stats.peak_queue_depth <= 256
    assert hashlib.sha256(table_path.read_bytes()).digest() == table_hash
    assert hashlib.sha256(marker_path.read_bytes()).digest() == marker_hash

    second = np.empty_like(output)
    reader.gather(ids, second)
    np.testing.assert_array_equal(second, output)


def test_native_reader_releases_file_descriptors(tmp_path):
    table_path = tmp_path / "table.bin"
    _table(table_path)
    reader = _reader(table_path)
    del reader
    gc.collect()
    baseline = len(os.listdir("/proc/self/fd"))

    for _ in range(3):
        reader = _reader(table_path)
        output = np.empty((1, ROW_BYTES), dtype=np.uint8)
        reader.gather(np.array([TP_START], dtype=np.int64), output)
        del reader
        gc.collect()

    assert len(os.listdir("/proc/self/fd")) <= baseline


def test_native_reader_rejects_wrong_output_shape(tmp_path):
    table_path = tmp_path / "table.bin"
    _table(table_path)
    reader = _reader(table_path)
    ids = np.array([TP_START], dtype=np.int64)

    with pytest.raises(ValueError, match="row width"):
        reader.gather(ids, np.empty((1, ROW_BYTES - 1), dtype=np.uint8))


def test_native_reader_preserves_rows_across_waves_and_batches(tmp_path):
    row_count = 140000
    rows = np.arange(row_count * ROW_BYTES, dtype=np.uint64)
    rows = ((rows * 17 + 11) % 256).astype(np.uint8).reshape(-1, ROW_BYTES)
    table_path = tmp_path / "large-table.bin"
    table_path.write_bytes(rows.tobytes())
    reader = PageReader(
        table_path,
        ROW_BYTES,
        0,
        0,
        row_count,
    )
    ids = np.arange(0, 130000, 26, dtype=np.int64)
    output = np.empty((ids.size, ROW_BYTES), dtype=np.uint8)

    stats = reader.gather(ids, output)

    np.testing.assert_array_equal(output, rows[ids])
    assert stats.unique_pages > 4096
    assert stats.read_batches > 16
    assert stats.peak_queue_depth == 256


def test_native_reader_fails_closed_after_io_error(tmp_path):
    table_path = tmp_path / "table.bin"
    _table(table_path)
    reader = _reader(table_path)
    os.truncate(table_path, 4096)
    output = np.empty((1, ROW_BYTES), dtype=np.uint8)

    with pytest.raises(OSError):
        reader.gather(np.array([TP_END - 1], dtype=np.int64), output)
    with pytest.raises(OSError, match="unusable after an earlier I/O failure"):
        reader.gather(np.array([TP_START], dtype=np.int64), output)


def test_native_reader_reads_tp_range_from_global_table(tmp_path):
    global_rows = 256
    rows = np.arange(global_rows * ROW_BYTES, dtype=np.uint64)
    rows = ((rows * 13 + 3) % 256).astype(np.uint8).reshape(-1, ROW_BYTES)
    table_path = tmp_path / "global-table.bin"
    table_path.write_bytes(rows.tobytes())
    reader = PageReader(table_path, ROW_BYTES, 0, TP_START, TP_END)
    ids = np.array([TP_START, TP_START + 9, TP_END - 1], dtype=np.int64)
    output = np.empty((ids.size, ROW_BYTES), dtype=np.uint8)

    reader.gather(ids, output)

    np.testing.assert_array_equal(output, rows[ids])
