"""Unit tests for ``ShmPointerMMData`` when its segment disappears in transit.

Multimodal features travel tokenizer -> scheduler as POSIX shared-memory
pointers. If the segment is unlinked before a receiver attaches (another
consumer materialized it first, a leaked-segment sweep, or a lost race between
TP ranks), the receiver used to raise FileNotFoundError from inside
``recv_pyobj`` and take the whole scheduler process down with it
(``sgl_shm_mm_<pid>_<rand>`` crashes under image-heavy agentic load).

The pointer must now survive unpickling and report the loss on
``materialize()`` so the request receiver can reject just that request.
No server / GPU / weight loading involved.
"""

import pickle
import unittest
from http import HTTPStatus
from multiprocessing import shared_memory
from types import SimpleNamespace
from unittest import mock

import msgspec
import torch

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.io_struct import (
    BatchTokenizedGenerateReqInput,
    MMInputsProcessError,
    TokenizedGenerateReqInput,
)
from sglang.srt.managers.mm_utils import (
    ShmPointerMMData,
    discard_shm_features,
    has_shm_features,
    unwrap_shm_features,
)
from sglang.srt.managers.schedule_batch import (
    FINISH_ABORT,
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.multimodal.transport.cuda_ipc import CudaIpcTensorTransportProxy
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.session.session_controller import Session
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _segment_exists(name: str) -> bool:
    try:
        shm = shared_memory.SharedMemory(name=name)
    except FileNotFoundError:
        return False
    shm.close()
    return True


class TestShmPointerMMData(CustomTestCase):
    def test_roundtrip_materializes_and_unlinks(self):
        src = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
        ptr = ShmPointerMMData(src, precomputed_hash=42)
        self.assertTrue(_segment_exists(ptr.shm_name))

        received = pickle.loads(pickle.dumps(ptr))
        self.assertIsNone(received._materialization_error)
        out = received.materialize()
        torch.testing.assert_close(out, src)
        self.assertEqual(received.precomputed_hash, 42)
        self.assertFalse(_segment_exists(ptr.shm_name))
        self.assertIsNone(received._shm_handle)

    def test_missing_segment_does_not_raise_on_unpickle(self):
        src = torch.ones(16, dtype=torch.float16)
        ptr = ShmPointerMMData(src)
        payload = pickle.dumps(ptr)
        # Simulate the lost race: the segment is gone before this receiver attaches.
        shared_memory.SharedMemory(name=ptr.shm_name).unlink()

        received = pickle.loads(payload)  # must not raise
        self.assertIsNotNone(received._materialization_error)
        self.assertIn("FileNotFoundError", received._materialization_error)
        self.assertIsNone(received.tensor)

        with self.assertRaises(RuntimeError) as ctx:
            received.materialize()
        self.assertIn(ptr.shm_name, str(ctx.exception))
        # cleanup is idempotent on a dead segment
        received.close_and_unlink()

    def test_second_receiver_after_materialize_is_rejected_not_fatal(self):
        src = torch.zeros(8, dtype=torch.float32)
        ptr = ShmPointerMMData(src)
        payload = pickle.dumps(ptr)
        first = pickle.loads(payload)
        first.materialize()  # unlinks
        second = pickle.loads(payload)  # the double-consumption case
        self.assertIsNotNone(second._materialization_error)
        with self.assertRaises(RuntimeError):
            second.materialize()

    def test_discard_shm_features_releases_segments(self):
        feats = [ShmPointerMMData(torch.zeros(4)), ShmPointerMMData(torch.zeros(4))]
        item = MultimodalDataItem(
            modality=Modality.IMAGE,
            offsets=[(0, 2), (2, 4)],
            feature=feats,
        )
        req = _tokenized_req(mm_inputs=MultimodalInputs(mm_items=[item]))
        self.assertTrue(has_shm_features([req]))
        names = [f.shm_name for f in feats]
        discard_shm_features(req)
        for name in names:
            self.assertFalse(_segment_exists(name))


def _tokenized_req(**overrides) -> TokenizedGenerateReqInput:
    """Build a TokenizedGenerateReqInput with every required field defaulted to None."""
    kwargs = {
        f.name: None
        for f in msgspec.structs.fields(TokenizedGenerateReqInput)
        if f.required
    }
    kwargs.update(rid="r1", input_text="", input_ids=[1, 2, 3])
    kwargs.update(overrides)
    return TokenizedGenerateReqInput(**kwargs)


class TestMarkerConsumers(CustomTestCase):
    """The MMInputsProcessError marker planted by the request receiver reaches
    other consumers of mm_inputs: the SHM helpers on the next pipeline stage,
    session continuation (BOS strip), and the scheduler's request handlers.
    None of them may raise; the scheduler must reject before session handling."""

    def _marker_req(self, **overrides):
        return _tokenized_req(
            input_ids=[1, 2],
            mm_inputs=MMInputsProcessError(message="lost segment"),
            **overrides,
        )

    def test_shm_helpers_tolerate_the_marker(self):
        req = self._marker_req()
        self.assertFalse(has_shm_features([req]))  # next PP stage's detection
        with mock.patch(
            "sglang.srt.managers.mm_utils._get_is_default_transport", return_value=False
        ), mock.patch(
            "sglang.srt.managers.mm_utils.get_serving",
            return_value=SimpleNamespace(skip_tokenizer_init=False),
        ):
            self.assertIs(unwrap_shm_features(req), req)
        discard_shm_features(req)  # must not raise
        self.assertIsInstance(
            req.mm_inputs, MMInputsProcessError
        )  # rejection semantics kept
        batch = BatchTokenizedGenerateReqInput(batch=[req])
        self.assertFalse(has_shm_features([batch]))
        discard_shm_features(batch)

    def test_session_bos_strip_tolerates_the_marker(self):
        req = self._marker_req()
        Session._strip_bos_token(req, SimpleNamespace(bos_token_id=1))
        self.assertEqual(list(req.input_ids), [2])
        self.assertIsInstance(req.mm_inputs, MMInputsProcessError)

    def test_scheduler_rejects_marker_before_session_processing(self):
        queued = []
        session_touched = []

        class _Sessions:  # any session lookup would be a bug
            def __contains__(self, key):
                session_touched.append(key)
                return False

        stub = SimpleNamespace(
            model_config=SimpleNamespace(vocab_size=32),
            tokenizer=None,
            disaggregation_mode=DisaggregationMode.NULL,
            session_controller=_Sessions(),
            init_req_max_new_tokens=lambda req: None,
            _add_request_to_queue=lambda req, is_retracted=False: queued.append(req),
        )
        req = self._marker_req(
            session_params=SimpleNamespace(id="s1"),
            sampling_params=SamplingParams(max_new_tokens=4),
        )
        with mock.patch(
            "sglang.srt.managers.schedule_batch.get_parallel",
            return_value=SimpleNamespace(tp_rank=0),
        ):
            self.assertTrue(Scheduler._reject_mm_transport_failure(stub, req))
        self.assertEqual(len(queued), 1)
        fin = queued[0].to_finish
        self.assertIsInstance(fin, FINISH_ABORT)
        self.assertEqual(fin.status_code, HTTPStatus.INTERNAL_SERVER_ERROR)
        self.assertEqual(fin.err_type, "InternalServerError")
        self.assertIn("lost segment", fin.message)
        self.assertEqual(session_touched, [])
        # a healthy request is not touched by the guard
        self.assertFalse(
            Scheduler._reject_mm_transport_failure(
                stub, _tokenized_req(input_ids=[1, 2], mm_inputs=None)
            )
        )

    def _stub_scheduler(self, mode, queued, streamed, touched):
        class _Sessions:  # any session lookup would be a bug
            def __contains__(self, key):
                touched.append(("session", key))
                return False

        class _Queue:  # any disaggregation queue entry would be a bug
            def add(self, *a, **kw):
                touched.append(("queue", mode))
                raise AssertionError("aborted request reached a disaggregation queue")

        return SimpleNamespace(
            model_config=SimpleNamespace(vocab_size=32),
            tokenizer=None,
            disaggregation_mode=mode,
            session_controller=_Sessions(),
            disagg_prefill_bootstrap_queue=_Queue(),
            disagg_decode_prealloc_queue=_Queue(),
            init_req_max_new_tokens=lambda req: None,
            _add_request_to_queue=lambda req, is_retracted=False: queued.append(req),
            output_streamer=SimpleNamespace(
                stream_output=lambda reqs, return_logprob, skip_req=None: streamed.append(
                    list(reqs)
                )
            ),
        )

    def test_scheduler_rejects_marker_on_disaggregated_workers(self):
        """PREFILL/DECODE workers answer the error directly: no bootstrap, no prealloc."""
        for mode in (DisaggregationMode.PREFILL, DisaggregationMode.DECODE):
            queued, streamed, touched = [], [], []
            stub = self._stub_scheduler(mode, queued, streamed, touched)
            req = self._marker_req(
                session_params=SimpleNamespace(id="s1"),
                sampling_params=SamplingParams(max_new_tokens=4),
                bootstrap_host="10.0.0.7",
                bootstrap_port=8998,
                bootstrap_room=4242,
            )
            with mock.patch(
                "sglang.srt.managers.schedule_batch.get_parallel",
                return_value=SimpleNamespace(tp_rank=0),
            ):
                self.assertTrue(Scheduler._reject_mm_transport_failure(stub, req), mode)
            self.assertEqual(queued, [], mode)
            self.assertEqual(touched, [], mode)
            self.assertEqual(len(streamed), 1, mode)
            (out,) = streamed[0]
            fin = out.finished_reason
            self.assertIsInstance(fin, FINISH_ABORT)
            self.assertEqual(fin.status_code, HTTPStatus.INTERNAL_SERVER_ERROR)
            self.assertEqual(fin.err_type, "InternalServerError")
            self.assertIn("lost segment", fin.message)
            # coordinates survive so nothing downstream sees a None address
            self.assertEqual(
                (out.bootstrap_host, out.bootstrap_port, out.bootstrap_room),
                ("10.0.0.7", 8998, 4242),
            )


class _FakePoolProxy(CudaIpcTensorTransportProxy):
    """A CUDA pool slice without CUDA: records acknowledgements, forbids reconstruction."""

    def __init__(
        self,
    ):  # noqa: D401 - bypass the real constructor (needs tensors + a pool)
        self._consumer_acknowledged = False
        self.acks = []

    def acknowledge_consumption(self, consumer_count=1, consumer_rank=None):
        if self._consumer_acknowledged:
            return
        self.acks.append(consumer_count)
        self._consumer_acknowledged = True

    def reconstruct_on_target_device(self, *args, **kwargs):
        raise AssertionError("a rejected request must not reconstruct its features")


class _FakePackedView(_FakePoolProxy):
    """One typed view of a packed VMM transfer: acknowledges only through its owner."""

    def __init__(self, owner):
        super().__init__()
        self._packed_owner = owner

    def acknowledge_consumption(self, consumer_count=None):
        raise RuntimeError(
            "Packed CUDA VMM features must be reconstructed before release"
        )


class TestMixedTransportRejection(CustomTestCase):
    """A request can carry one image as a CUDA VMM/IPC pool slice and another as the
    CPU->SHM fallback. Rejecting it must release both: the SHM segment is
    unlinked and the GPU lease is acknowledged for this rank exactly once,
    without reconstructing anything and without double-releasing on a repeat."""

    def test_mixed_shm_and_vmm_rejection_releases_gpu_lease_exactly_once(self):
        vmm = _FakePoolProxy()
        owner = _FakePoolProxy()
        view_a, view_b = _FakePackedView(owner), _FakePackedView(owner)
        shm = ShmPointerMMData(torch.zeros(4))
        items = [
            MultimodalDataItem(modality=Modality.IMAGE, offsets=[(0, 2)], feature=vmm),
            MultimodalDataItem(
                modality=Modality.IMAGE,
                offsets=[(2, 4)],
                feature=view_a,
                model_specific_data={"image_grid_thw": view_b},
            ),
            MultimodalDataItem(modality=Modality.IMAGE, offsets=[(4, 6)], feature=shm),
        ]
        req = _tokenized_req(mm_inputs=MultimodalInputs(mm_items=items))
        # the SHM segment is gone: the real rejection precondition
        shared_memory.SharedMemory(name=shm.shm_name).unlink()

        discard_shm_features(req)

        self.assertEqual(vmm.acks, [1])  # this rank's slot, once
        self.assertEqual(
            owner.acks, [1]
        )  # one packed transfer, two views -> one release
        self.assertEqual(view_a.acks, [])
        self.assertEqual(view_b.acks, [])
        self.assertFalse(_segment_exists(shm.shm_name))
        # idempotent: a second discard (e.g. the consensus loop) releases nothing twice
        discard_shm_features(req)
        self.assertEqual(vmm.acks, [1])
        self.assertEqual(owner.acks, [1])
        # the marker still goes on afterwards, as the receiver does
        req.mm_inputs = MMInputsProcessError(message="lost segment")
        self.assertFalse(has_shm_features([req]))

    def test_release_skips_already_acknowledged_slices(self):
        proxy = _FakePoolProxy()
        proxy.acknowledge_consumption(1)
        item = MultimodalDataItem(
            modality=Modality.IMAGE, offsets=[(0, 1)], feature=proxy
        )
        self.assertEqual(item.release_transport_proxies(), 0)
        self.assertEqual(proxy.acks, [1])


if __name__ == "__main__":
    unittest.main()
