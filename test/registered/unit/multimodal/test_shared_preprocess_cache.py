"""Cross-process behavior of the optional CPU preprocess artifact store."""

import asyncio
import contextlib
import errno
import multiprocessing
import os
import sqlite3
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=40, suite="base-a-test-cpu")

try:
    from sglang.srt.multimodal.cache import shared_preprocess_cache as shared_backend
    from sglang.srt.multimodal.cache.shared_preprocess_cache import (
        SharedCacheRequest,
        SharedPreprocessCache,
    )
except ModuleNotFoundError as error:
    if error.name != "sglang.srt.multimodal.cache.shared_preprocess_cache":
        raise
    SharedCacheRequest = SharedPreprocessCache = None
    shared_backend = None


@dataclass(frozen=True)
class ImageArtifact:
    feature: torch.Tensor
    grid: tuple[int, int, int]
    metadata: dict


class _UnwellIndex:
    """A SQLite connection that fails exactly where a full disk would.

    Only the statements the caller names are refused, so a test cannot mistake a
    swallowed compute or identity error for cache degradation.
    """

    def __init__(self, connection, refuses):
        object.__setattr__(self, "_connection", connection)
        object.__setattr__(self, "_refuses", refuses)

    def execute(self, statement, *parameters):
        if self._refuses(statement):
            raise sqlite3.OperationalError("database or disk is full")
        return self._connection.execute(statement, *parameters)

    def commit(self):
        if self._refuses("COMMIT"):
            raise sqlite3.OperationalError("database or disk is full")
        return self._connection.commit()

    def __setattr__(self, name, value):
        setattr(self._connection, name, value)

    def __getattr__(self, name):
        return getattr(self._connection, name)


@contextlib.contextmanager
def unwell_index(refuses):
    real_connect = sqlite3.connect
    with patch(
        "sglang.srt.multimodal.cache.shared_preprocess_cache.sqlite3.connect",
        side_effect=lambda *args, **kwargs: _UnwellIndex(
            real_connect(*args, **kwargs), refuses
        ),
    ):
        yield


@contextlib.contextmanager
def full_locks(where):
    """Make lock-file opens fail the way ENOSPC does, without touching data."""
    real_open = Path.open

    def opening(self, *args, **kwargs):
        if where(self):
            raise OSError(errno.ENOSPC, "No space left on device")
        return real_open(self, *args, **kwargs)

    with patch.object(Path, "open", opening):
        yield


def all_locks(path):
    return path.suffix == ".lock"


def shard_locks_only(path):
    return path.parent.name == "locks"


def artifact(number):
    return ImageArtifact(
        feature=torch.tensor([[number, 2, 3]], dtype=torch.float32),
        grid=(1, 1, 1),
        metadata={"shape": torch.tensor([1, 3]), "label": "image"},
    )


def concurrent_reader(directory, barrier, output):
    async def run():
        cache = SharedPreprocessCache(directory, max_size_bytes=65536)
        barrier.wait(timeout=30)

        async def compute(indices):
            with open(Path(directory) / "computations", "ab", buffering=0) as stream:
                stream.write(b"computed\n")
            await asyncio.sleep(0.2)
            return [artifact(7) for _ in indices]

        result = await cache.resolve_many(
            [SharedCacheRequest("same-key", b"image bytes", b"configuration")],
            compute,
            artifact_type=ImageArtifact,
        )
        output.put(result[0].feature.tolist())

    asyncio.run(run())


def interrupted_writer(directory, entered):
    async def run():
        cache = SharedPreprocessCache(directory, max_size_bytes=65536)

        async def compute(indices):
            entered.set()
            await asyncio.sleep(60)
            return [artifact(7)]

        await cache.resolve_many(
            [SharedCacheRequest("same-key", b"image bytes", b"configuration")],
            compute,
            artifact_type=ImageArtifact,
        )

    asyncio.run(run())


def many_async_readers(directory, entered):
    async def run():
        asyncio.get_running_loop().set_default_executor(
            ThreadPoolExecutor(max_workers=2)
        )
        cache = SharedPreprocessCache(directory, max_size_bytes=65536)
        request = SharedCacheRequest("same-key", b"image bytes", b"configuration")

        async def compute(indices):
            entered.set()
            await asyncio.sleep(0.2)
            return [artifact(7)]

        result = await asyncio.gather(
            *[
                cache.resolve_many([request], compute, artifact_type=ImageArtifact)
                for _ in range(8)
            ]
        )
        assert all(item[0].feature.tolist() == [[7, 2, 3]] for item in result)

    asyncio.run(run())


def interrupted_serializer(directory, entered):
    def write_partial(path, *args, **kwargs):
        Path(path).write_bytes(b"partial tensor")
        entered.set()
        time.sleep(60)

    async def run():
        cache = SharedPreprocessCache(directory, max_size_bytes=65536)

        async def compute(indices):
            return [artifact(7)]

        with patch(
            "sglang.srt.multimodal.cache.shared_preprocess_cache.np.save",
            side_effect=write_partial,
        ):
            await cache.resolve_many(
                [SharedCacheRequest("same-key", b"image bytes", b"configuration")],
                compute,
                artifact_type=ImageArtifact,
            )

    asyncio.run(run())


@unittest.skipUnless(os.name == "posix", "Shared file mappings require POSIX")
class TestSharedPreprocessCache(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.assertIsNotNone(
            SharedPreprocessCache,
            "The cross-worker preprocess cache backend is not implemented",
        )
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = self.temporary.name
        self.cache = SharedPreprocessCache(self.directory, max_size_bytes=65536)
        self.calls = []

    async def resolve(self, requests, *, cache=None, number=1):
        async def compute(indices):
            self.calls.extend(indices)
            return [artifact(number) for _ in indices]

        return await (cache or self.cache).resolve_many(
            requests, compute, artifact_type=ImageArtifact
        )

    async def test_clear_while_waiting_for_lock_does_not_repopulate(self):
        import threading

        request = SharedCacheRequest("queued", b"image", b"configuration")
        lookup_done = threading.Event()
        lookup = self.cache._lookup

        def observed_lookup(*args):
            result = lookup(*args)
            lookup_done.set()
            return result

        locks = self.cache._shard_locks([request])
        locks.__enter__()
        try:
            with patch.object(self.cache, "_lookup", side_effect=observed_lookup):
                task = asyncio.create_task(self.resolve([request]))
                self.assertTrue(await asyncio.to_thread(lookup_done.wait, 5))
                self.cache.clear()
        finally:
            locks.__exit__(None, None, None)
        result = await asyncio.wait_for(task, 5)
        self.assertEqual(result[0].feature.tolist(), [[1, 2, 3]])
        self.assertEqual(self.cache.cache_info()["entries"], 0)

    async def test_damaged_tensor_is_recomputed(self):
        request = SharedCacheRequest("key", b"image", b"configuration")
        await self.resolve([request])
        tensor_file = next(Path(self.directory).glob("object-*/tensor-0.npy"))
        tensor_file.write_bytes(b"interrupted or damaged file")
        result = await self.resolve([request], number=9)
        self.assertEqual(result[0].feature.tolist(), [[9, 2, 3]])
        self.assertEqual(self.cache.cache_info()["entries"], 1)

    async def test_truncated_tensor_is_recomputed(self):
        request = SharedCacheRequest("key", b"image", b"configuration")
        await self.resolve([request])
        next(Path(self.directory).glob("object-*/tensor-0.npy")).write_bytes(b"")
        result = await self.resolve([request], number=9)
        self.assertEqual(result[0].feature.tolist(), [[9, 2, 3]])
        self.assertEqual(self.cache.cache_info()["entries"], 1)

    async def test_scalar_empty_and_bfloat16_tensors_roundtrip(self):
        request = SharedCacheRequest("special", b"image", b"configuration")
        value = ImageArtifact(
            torch.empty((0, 3), dtype=torch.bfloat16),
            (1, 1, 1),
            {"scalar": torch.tensor(7, dtype=torch.bfloat16)},
        )

        async def compute(indices):
            return [value for _ in indices]

        await self.cache.resolve_many([request], compute, artifact_type=ImageArtifact)

        async def unexpected(indices):
            self.fail("Expected a cached empty tensor")

        result = (
            await self.cache.resolve_many(
                [request], unexpected, artifact_type=ImageArtifact
            )
        )[0]
        self.assertTrue(torch.equal(result.feature, value.feature))
        self.assertTrue(
            torch.equal(result.metadata["scalar"], value.metadata["scalar"])
        )

    async def test_another_instance_reuses_the_same_artifact(self):
        request = SharedCacheRequest("key", b"encoded-image", b"resize=32")
        first = (await self.resolve([request]))[0]
        other = SharedPreprocessCache(self.directory, max_size_bytes=65536)
        second = (await self.resolve([request], cache=other, number=9))[0]
        self.assertEqual(self.calls, [0])
        self.assertTrue(torch.equal(first.feature, second.feature))
        self.assertTrue(torch.equal(first.metadata["shape"], second.metadata["shape"]))
        self.assertEqual(second.grid, (1, 1, 1))

    async def test_colliding_keys_compare_source_bytes_and_configuration(self):
        first = SharedCacheRequest("forced-collision", b"one", b"resize=32")
        second = SharedCacheRequest("forced-collision", b"two", b"resize=32")
        third = SharedCacheRequest("forced-collision", b"one", b"resize=64")
        await self.resolve([first], number=1)
        await self.resolve([second], number=2)
        await self.resolve([third], number=3)
        for request, want in [(first, 1), (second, 2), (third, 3)]:
            result = (await self.resolve([request], number=99))[0]
            self.assertEqual(result.feature.tolist(), [[want, 2, 3]])
        self.assertEqual(len(self.calls), 3)
        self.assertEqual(self.cache.cache_info()["entries"], 3)

    async def test_duplicates_and_partial_hits_preserve_input_order(self):
        a = SharedCacheRequest("a", b"a", b"config")
        b = SharedCacheRequest("b", b"b", b"config")
        await self.resolve([a], number=1)
        result = await self.resolve([b, a, b], number=2)
        self.assertEqual([item.feature[0, 0].item() for item in result], [2, 1, 2])
        self.assertEqual(self.calls, [0, 0])

    async def test_returned_tensors_cannot_modify_another_reader(self):
        request = SharedCacheRequest("key", b"image", b"config")
        await self.resolve([request])
        result = (await self.resolve([request]))[0]
        result.feature[0, 0] = 99
        result.metadata["shape"][0] = 99
        again = (await self.resolve([request]))[0]
        self.assertEqual(again.feature.tolist(), [[1, 2, 3]])
        self.assertEqual(again.metadata["shape"].tolist(), [1, 3])

    async def test_lru_is_global_across_instances(self):
        directory = str(Path(self.directory) / "lru")
        self.cache = SharedPreprocessCache(
            directory, max_size_bytes=65536, max_entries=2
        )
        other = SharedPreprocessCache(directory, max_size_bytes=65536, max_entries=2)
        a, b, c = [SharedCacheRequest(key, key.encode(), b"config") for key in "abc"]
        await self.resolve([a])
        await self.resolve([b])
        await self.resolve([a], cache=other)
        await self.resolve([c], cache=other)
        await self.resolve([a])
        self.assertEqual(len(self.calls), 3)
        await self.resolve([b])
        self.assertEqual(len(self.calls), 4)
        self.assertEqual(other.cache_info()["entries"], 2)

    async def test_oversized_artifact_is_returned_without_retention(self):
        cache = SharedPreprocessCache(
            str(Path(self.directory) / "small"), max_size_bytes=8
        )
        request = SharedCacheRequest("key", b"image", b"config")
        for _ in range(2):
            result = (await self.resolve([request], cache=cache))[0]
            self.assertEqual(result.feature.tolist(), [[1, 2, 3]])
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(cache.cache_info()["entries"], 0)

    async def test_retained_bytes_are_bounded_across_instances(self):
        directory = str(Path(self.directory) / "byte-budget")
        cache = SharedPreprocessCache(directory, max_size_bytes=20000)
        other = SharedPreprocessCache(directory, max_size_bytes=20000)
        calls = []

        async def compute(indices):
            calls.extend(indices)
            return [
                ImageArtifact(torch.ones(2048, dtype=torch.float32), (1, 1, 1), {})
                for _ in indices
            ]

        requests = [SharedCacheRequest(key, key.encode(), b"config") for key in "abc"]
        for request in requests:
            await cache.resolve_many([request], compute, artifact_type=ImageArtifact)
        self.assertEqual(other.cache_info()["entries"], 2)
        self.assertLessEqual(other.cache_info()["size_bytes"], 20000)
        await other.resolve_many([requests[1]], compute, artifact_type=ImageArtifact)
        self.assertEqual(len(calls), 3)
        await other.resolve_many([requests[0]], compute, artifact_type=ImageArtifact)
        self.assertEqual(len(calls), 4)

    async def test_tensor_view_does_not_store_the_entire_batch(self):
        cache = SharedPreprocessCache(
            str(Path(self.directory) / "views"), max_size_bytes=4096
        )
        request = SharedCacheRequest("key", b"image", b"config")

        async def compute(indices):
            self.calls.extend(indices)
            storage = torch.arange(1_000_000, dtype=torch.float32)
            return [ImageArtifact(storage[1:4], (1, 1, 1), {})]

        for _ in range(2):
            result = await cache.resolve_many(
                [request], compute, artifact_type=ImageArtifact
            )
            self.assertEqual(result[0].feature.tolist(), [1, 2, 3])
        self.assertEqual(self.calls, [0])
        self.assertEqual(cache.cache_info()["entries"], 1)
        self.assertLessEqual(cache.cache_info()["size_bytes"], 4096)

    async def test_clear_during_compute_does_not_repopulate(self):
        request = SharedCacheRequest("key", b"image", b"config")
        entered, resume = asyncio.Event(), asyncio.Event()

        async def compute(indices):
            entered.set()
            await resume.wait()
            return [artifact(7)]

        task = asyncio.create_task(
            self.cache.resolve_many([request], compute, artifact_type=ImageArtifact)
        )
        await asyncio.wait_for(entered.wait(), timeout=5)
        self.cache.clear()
        resume.set()
        self.assertEqual((await task)[0].feature.tolist(), [[7, 2, 3]])
        self.assertEqual(self.cache.cache_info()["entries"], 0)
        await self.resolve([request])
        self.assertEqual(self.calls, [0])

    async def test_eviction_keeps_an_existing_mapping_usable(self):
        request = SharedCacheRequest("key", b"image", b"config")
        await self.resolve([request])
        result = (await self.resolve([request]))[0]
        self.cache.clear()
        self.assertEqual(result.feature.tolist(), [[1, 2, 3]])
        self.assertEqual(self.cache.cache_info()["entries"], 0)

    async def test_failed_compute_releases_the_key(self):
        request = SharedCacheRequest("key", b"image", b"config")

        async def compute(indices):
            raise ValueError("invalid image")

        with self.assertRaisesRegex(ValueError, "invalid image"):
            await self.cache.resolve_many(
                [request], compute, artifact_type=ImageArtifact
            )
        result = (await self.resolve([request]))[0]
        self.assertEqual(result.feature.tolist(), [[1, 2, 3]])

    async def test_a_full_index_still_serves_a_warm_hit(self):
        request = SharedCacheRequest("key", b"image", b"config")
        await self.resolve([request])
        with unwell_index(lambda statement: "UPDATE entries SET touched" in statement):
            result = (await self.resolve([request], number=9))[0]
        self.assertEqual(result.feature.tolist(), [[1, 2, 3]])
        self.assertEqual(len(self.calls), 1, "the verified artifact was recomputed")
        self.assertEqual(self.cache.cache_info()["entries"], 1)

    async def test_a_failed_index_commit_still_serves_a_warm_hit(self):
        request = SharedCacheRequest("key", b"image", b"config")
        await self.resolve([request])
        with unwell_index(lambda statement: statement.strip() == "COMMIT"):
            result = (await self.resolve([request], number=9))[0]
        self.assertEqual(result.feature.tolist(), [[1, 2, 3]])
        self.assertEqual(len(self.calls), 1, "the verified artifact was recomputed")

    async def test_an_unusable_index_costs_reuse_not_the_request(self):
        request = SharedCacheRequest("key", b"image", b"config")
        with unwell_index(lambda statement: True):
            result = (await self.resolve([request], number=4))[0]
        self.assertEqual(result.feature.tolist(), [[4, 2, 3]])
        self.assertEqual(self.cache.cache_info()["entries"], 0)
        result = (await self.resolve([request], number=9))[0]
        self.assertEqual(result.feature.tolist(), [[9, 2, 3]])
        self.assertEqual(len(self.calls), 2, "recovery should not recompute twice")
        self.assertEqual(self.cache.cache_info()["entries"], 1)
        result = (await self.resolve([request], number=5))[0]
        self.assertEqual(result.feature.tolist(), [[9, 2, 3]])
        self.assertEqual(len(self.calls), 2)

    async def test_a_generation_that_cannot_be_read_retains_nothing(self):
        request = SharedCacheRequest("key", b"image", b"config")
        with unwell_index(lambda statement: statement.startswith("SELECT generation")):
            result = (await self.resolve([request], number=4))[0]
        self.assertEqual(result.feature.tolist(), [[4, 2, 3]])
        self.assertEqual(self.cache.cache_info()["entries"], 0)
        await self.resolve([request], number=9)
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(self.cache.cache_info()["entries"], 1)

    async def test_lock_files_on_a_full_disk_still_answer_the_request(self):
        request = SharedCacheRequest("key", b"image", b"config")
        with full_locks(all_locks):
            result = (await self.resolve([request], number=4))[0]
        self.assertEqual(result.feature.tolist(), [[4, 2, 3]])
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(list(Path(self.directory).glob(".writing-*")), [])
        await self.resolve([request], number=9)
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(self.cache.cache_info()["entries"], 1)

    async def test_shard_lock_failure_computes_without_publishing(self):
        request = SharedCacheRequest("key", b"image", b"config")
        with full_locks(shard_locks_only):
            result = (await self.resolve([request], number=4))[0]
        self.assertEqual(result.feature.tolist(), [[4, 2, 3]])
        self.assertEqual(len(self.calls), 1)
        # An unclaimed key must not be published: cleanup would then be free to
        # treat this request's files as an abandoned publication.
        self.assertEqual(self.cache.cache_info()["entries"], 0)
        self.assertEqual(list(Path(self.directory).glob(".writing-*")), [])
        self.assertEqual(list(Path(self.directory).glob("object-*")), [])
        # Once the locks work again, retention and single-flight are back.
        await self.resolve([request], number=9)
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(self.cache.cache_info()["entries"], 1)
        result = (await self.resolve([request], number=5))[0]
        self.assertEqual(result.feature.tolist(), [[9, 2, 3]])
        self.assertEqual(len(self.calls), 2)

    async def test_a_rolled_back_eviction_removes_nothing(self):
        request = SharedCacheRequest("key", b"image", b"config")
        await self.resolve([request])
        directory = next(Path(self.directory).glob("object-*"))
        (directory / "tensor-0.npy").write_bytes(b"")
        with unwell_index(lambda statement: statement.strip() == "COMMIT"):
            result = (await self.resolve([request], number=9))[0]
        self.assertEqual(result.feature.tolist(), [[9, 2, 3]])
        # The rollback restored the row, so its files have to still be there.
        self.assertTrue(directory.exists())
        self.assertEqual(self.cache.cache_info()["entries"], 1)
        self.assertEqual(list(Path(self.directory).glob(".writing-*")), [])
        # Recovery evicts the damaged artifact for real and retains one copy. The
        # phase above kept nothing either, since its commit failed as well.
        result = (await self.resolve([request], number=5))[0]
        self.assertEqual(result.feature.tolist(), [[5, 2, 3]])
        self.assertEqual(self.cache.cache_info()["entries"], 1)
        self.assertFalse(directory.exists())
        self.assertEqual(len(self.calls), 3)

    async def test_store_trouble_does_not_swallow_a_failing_computation(self):
        warm = SharedCacheRequest("warm", b"warm", b"config")
        cold = SharedCacheRequest("cold", b"cold", b"config")
        await self.resolve([warm])

        async def broken_media(indices):
            raise OSError(errno.EIO, "cannot read the image")

        for label, refuses in (
            (
                "hit bookkeeping",
                lambda statement: "UPDATE entries SET touched" in statement,
            ),
            ("whole index", lambda statement: True),
        ):
            with self.subTest(index=label), unwell_index(refuses):
                with self.assertRaisesRegex(OSError, "cannot read the image"):
                    await self.cache.resolve_many(
                        [warm, cold], broken_media, artifact_type=ImageArtifact
                    )
        self.assertEqual(self.cache.cache_info()["entries"], 1)

    async def test_a_real_input_error_survives_unavailable_locks(self):
        request = SharedCacheRequest("key", b"image", b"config")

        async def broken_media(indices):
            raise OSError(errno.EIO, "cannot read the image")

        with full_locks(all_locks):
            with self.assertRaisesRegex(OSError, "cannot read the image"):
                await self.cache.resolve_many(
                    [request], broken_media, artifact_type=ImageArtifact
                )
        self.assertEqual(self.cache.cache_info()["entries"], 0)

    async def test_cancellation_while_the_index_is_blocked_still_cancels(self):
        request = SharedCacheRequest("key", b"image", b"config")
        entered, release = threading.Event(), threading.Event()
        lookup = self.cache._lookup

        def blocked_lookup(*args):
            entered.set()
            release.wait(5)
            return lookup(*args)

        try:
            with patch.object(self.cache, "_lookup", side_effect=blocked_lookup):
                task = asyncio.create_task(self.resolve([request]))
                self.assertTrue(await asyncio.to_thread(entered.wait, 5))
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
        finally:
            release.set()
        await asyncio.sleep(0.1)

    async def test_the_index_lock_does_not_stall_the_serving_event_loop(self):
        held, release = threading.Event(), threading.Event()

        def holder():
            # Another tokenizer or scheduler worker mid-transaction or mid-GC.
            with self.cache._file_lock(Path(self.directory) / "index.lock"):
                held.set()
                release.wait(5)

        thread = threading.Thread(target=holder, daemon=True)
        thread.start()
        self.assertTrue(held.wait(5))
        ticks = 0

        async def heartbeat():
            nonlocal ticks
            while True:
                await asyncio.sleep(0.01)
                ticks += 1

        beat = asyncio.create_task(heartbeat())
        try:
            await asyncio.sleep(0.05)
            before = ticks
            started = time.perf_counter()
            reading = asyncio.create_task(self.cache.generation())
            await asyncio.sleep(0.25)
            self.assertFalse(reading.done(), "the generation read never waited")
            release.set()
            generation = await asyncio.wait_for(reading, 5)
            waited = time.perf_counter() - started
            self.assertGreaterEqual(waited, 0.2, "the index lock was never contended")
            self.assertGreaterEqual(ticks - before, 15, "the event loop stalled")
            self.assertEqual(generation, 0)
            self.cache.clear()
            self.assertEqual(await self.cache.generation(), 1)
        finally:
            release.set()
            beat.cancel()
            thread.join(5)

    async def test_full_filesystem_does_not_fail_the_request(self):
        request = SharedCacheRequest("key", b"image", b"config")
        with patch(
            "sglang.srt.multimodal.cache.shared_preprocess_cache.np.save",
            side_effect=OSError(errno.ENOSPC, "No space left on device"),
        ):
            result = (await self.resolve([request]))[0]
        self.assertEqual(result.feature.tolist(), [[1, 2, 3]])
        self.assertEqual(self.cache.cache_info()["entries"], 0)
        self.assertEqual(list(Path(self.directory).glob(".writing-*")), [])
        result = (await self.resolve([request], number=2))[0]
        self.assertEqual(result.feature.tolist(), [[2, 2, 3]])

    def test_two_processes_compute_once(self):
        context = multiprocessing.get_context("spawn")
        barrier, output = context.Barrier(2), context.Queue()
        processes = [
            context.Process(
                target=concurrent_reader, args=(self.directory, barrier, output)
            )
            for _ in range(2)
        ]
        try:
            for process in processes:
                process.start()
            self.assertEqual(output.get(timeout=60), [[7, 2, 3]])
            self.assertEqual(output.get(timeout=60), [[7, 2, 3]])
            for process in processes:
                process.join(timeout=10)
                self.assertEqual(process.exitcode, 0)
            self.assertEqual(
                (Path(self.directory) / "computations").read_bytes(), b"computed\n"
            )
        finally:
            for process in processes:
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=5)
            output.close()

    async def test_a_terminated_writer_does_not_leave_a_permanent_lock(self):
        context = multiprocessing.get_context("spawn")
        entered = context.Event()
        process = context.Process(
            target=interrupted_writer, args=(self.directory, entered)
        )
        try:
            process.start()
            self.assertTrue(await asyncio.to_thread(entered.wait, 60))
            process.terminate()
            await asyncio.to_thread(process.join, 5)
            result = await asyncio.wait_for(
                self.resolve(
                    [SharedCacheRequest("same-key", b"image bytes", b"configuration")]
                ),
                timeout=5,
            )
            self.assertEqual(result[0].feature.tolist(), [[1, 2, 3]])
        finally:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)

    async def test_lock_waiters_do_not_starve_the_executor_needed_to_publish(self):
        context = multiprocessing.get_context("spawn")
        entered = context.Event()
        process = context.Process(
            target=many_async_readers, args=(self.directory, entered)
        )
        try:
            process.start()
            self.assertTrue(await asyncio.to_thread(entered.wait, 60))
            await asyncio.to_thread(process.join, 5)
            self.assertFalse(
                process.is_alive(),
                "Lock waiters exhausted the executor needed to publish the result",
            )
            self.assertEqual(process.exitcode, 0)
        finally:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)

    async def test_restart_removes_files_left_by_an_interrupted_publication(self):
        context = multiprocessing.get_context("spawn")
        entered = context.Event()
        process = context.Process(
            target=interrupted_serializer, args=(self.directory, entered)
        )
        try:
            process.start()
            self.assertTrue(await asyncio.to_thread(entered.wait, 60))
            process.terminate()
            await asyncio.to_thread(process.join, 5)
            cache = SharedPreprocessCache(self.directory, max_size_bytes=65536)
            self.assertEqual(list(Path(self.directory).glob(".writing-*")), [])
            request = SharedCacheRequest("same-key", b"image bytes", b"configuration")
            result = (await self.resolve([request], cache=cache))[0]
            self.assertEqual(result.feature.tolist(), [[1, 2, 3]])
        finally:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
