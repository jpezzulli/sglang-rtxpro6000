"""SM120 numerical regressions for GDN MTP RecoverSSM (--gdn-mtp-cache-mode=none).

Real-GPU tests: they run the FlashInfer BF16-state WY output-only verifier and the
accepted-state recovery against the full intermediate-state reference that full mode
uses, on Penny's active T=4 / T=8, H=16, HV=48, K=V=128 geometry. Coverage follows the
acceptance plan: outputs and committed state for every acceptance count 0..width-1 and
mixed per-row widths, padded (negative-slot) rows, the mamba-track boundary pass at
every residue a draft window can actually reach, repeated continuation drift, prefix
restore, and CUDA-graph capture/replay of the recovery.

ATOL/RTOL are provisional: they are the tolerances this port was first qualified at,
not a derived bound on accumulated recurrence error. If they move, say so in the
review instead of quietly widening them -- the recovery replaces the verifier's own
per-step states, so a real regression here changes decoded tokens, not just a tensor.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.utils import is_flashinfer_available
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=300, stage="base-b", runner_config="1-gpu-large")

H, HV, K, V = 16, 48, 128, 128
WIDTHS = (4, 8)  # W4 and W8 drafts
TRACK_INTERVAL = 64
DEVICE = "cuda"
ATOL = 4e-2
RTOL = 4e-2


def _make_pool(size: int, width: int, mode: str):
    """A real MambaPool on the real device, the way none mode allocates one."""
    from sglang.srt.configs.mamba_utils import (
        Mamba2CacheParams,
        Mamba2StateDType,
        Mamba2StateShape,
    )
    from sglang.srt.mem_cache.memory_pool import MambaPool

    shape = Mamba2StateShape.create(
        tp_world_size=1,
        intermediate_size=2 * HV,
        n_groups=1,
        num_heads=HV,
        head_dim=K,
        state_size=V,
        conv_kernel=4,
    )
    params = Mamba2CacheParams(
        shape=shape,
        dtype=Mamba2StateDType(conv=torch.bfloat16, temporal=torch.bfloat16),
        layers=[0],
    )
    return MambaPool(
        size=size,
        spec_state_size=size,
        cache_params=params,
        mamba_layer_ids=[0],
        device=DEVICE,
        speculative_num_draft_tokens=width,
        gdn_mtp_cache_mode=mode,
    )


def _sm_major() -> int:
    return torch.cuda.get_device_capability()[0] if torch.cuda.is_available() else 0


@unittest.skipUnless(
    torch.cuda.is_available() and is_flashinfer_available() and _sm_major() == 12,
    "SM120 with FlashInfer is required",
)
class TestRecoverSSMSM120(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from flashinfer.gdn_kernels import gated_delta_rule_mtp_wy_output_only
        from flashinfer.gdn_kernels.gdn_decode_bf16_state import gated_delta_rule_mtp

        cls.full_kernel = staticmethod(gated_delta_rule_mtp)
        cls.wy_kernel = staticmethod(gated_delta_rule_mtp_wy_output_only)

    # --- fixtures ------------------------------------------------------------
    @staticmethod
    def _case(batch_size: int, seed: int, width: int):
        gen = torch.Generator(device=DEVICE).manual_seed(seed)

        def rnd(*shape, dtype=torch.bfloat16):
            return torch.randn(*shape, device=DEVICE, dtype=dtype, generator=gen)

        pool_size = batch_size * 2 + 8
        # Slot 0 is the pool's reserved discard row, as in MambaPool (free_slots
        # starts at 1), which is what padded rows and masked boundary rows point at.
        slots = torch.arange(batch_size, device=DEVICE, dtype=torch.int32) + 1
        return {
            "T": width,
            "A_log": (rnd(HV, dtype=torch.float32) * 0.1).float(),
            "dt_bias": (rnd(HV, dtype=torch.float32) * 0.1).float(),
            "q": rnd(batch_size, width, H, K),
            "k": rnd(batch_size, width, H, K),
            "v": rnd(batch_size, width, HV, V),
            "a": rnd(batch_size, width, HV),
            "b": rnd(batch_size, width, HV),
            "state": rnd(pool_size, HV, V, K),
            "slots": slots,
        }

    @classmethod
    def _full(cls, case, state, slots=None):
        """Reference: state-writing verifier, h at every draft position.

        `slots` must be the mapping the run under test uses, or the reference is
        computed from different initial states than the code under verification.
        """
        batch_size, width = case["q"].shape[:2]
        slots = case["slots"] if slots is None else slots
        intermediate = torch.empty(
            batch_size, width, HV, V, K, device=DEVICE, dtype=torch.bfloat16
        )
        output = cls.full_kernel(
            A_log=case["A_log"],
            a=case["a"],
            dt_bias=case["dt_bias"],
            q=case["q"],
            k=case["k"],
            v=case["v"],
            b=case["b"],
            initial_state_source=state,
            initial_state_indices=slots,
            intermediate_states_buffer=intermediate,
            disable_state_update=True,
            use_qk_l2norm_in_kernel=True,
        )
        return output, intermediate

    @classmethod
    def _wy(cls, case, state):
        """None-mode verifier: outputs only, no per-draft states written."""
        return cls.wy_kernel(
            A_log=case["A_log"],
            a=case["a"],
            dt_bias=case["dt_bias"],
            q=case["q"],
            k=case["k"],
            v=case["v"],
            b=case["b"],
            initial_state_source=state,
            initial_state_indices=case["slots"],
            disable_state_update=True,
            use_qk_l2norm_in_kernel=True,
        )

    @classmethod
    def _recover(cls, case, state, accepted_steps, output_slots=None, slots=None):
        """None-mode commit: fold the accepted prefix into the state slot."""
        slots = case["slots"] if slots is None else slots
        output_slots = slots if output_slots is None else output_slots
        cls.full_kernel(
            A_log=case["A_log"],
            a=case["a"],
            dt_bias=case["dt_bias"],
            # q == k: the kernel l2-norms both, and only the state is kept.
            q=case["k"],
            k=case["k"],
            v=case["v"],
            b=case["b"],
            initial_state_source=state,
            initial_state_indices=slots,
            output_state_indices=output_slots,
            accepted_steps=accepted_steps,
            disable_state_update=False,
            disable_output=True,
            use_qk_l2norm_in_kernel=True,
        )

    @staticmethod
    def _commit_reference(state, slots, intermediate, accepted_steps):
        rows = torch.arange(slots.numel(), device=DEVICE)
        state[slots.long()] = intermediate[rows, accepted_steps.long()]

    # --- verifier parity ----------------------------------------------------
    def test_wy_output_and_recovered_state_all_widths_and_accept_lengths(self):
        for width in WIDTHS:
            for batch_size in (1, 2, 4):
                with self.subTest(width=width, batch_size=batch_size):
                    case = self._case(batch_size, 100 + batch_size + width, width)
                    output_full, intermediate = self._full(case, case["state"].clone())
                    output_wy = self._wy(case, case["state"].clone())
                    torch.testing.assert_close(
                        output_wy, output_full, atol=ATOL, rtol=RTOL
                    )

                    # accept counts 0..width-1, then a mixed per-row pattern.
                    patterns = [
                        torch.full(
                            (batch_size,), step, device=DEVICE, dtype=torch.int32
                        )
                        for step in range(width)
                    ]
                    if batch_size > 1:
                        patterns.append(
                            torch.arange(batch_size, device=DEVICE, dtype=torch.int32)
                            % width
                        )
                    for accepted_steps in patterns:
                        recovered = case["state"].clone()
                        self._recover(case, recovered, accepted_steps)
                        rows = torch.arange(batch_size, device=DEVICE)
                        torch.testing.assert_close(
                            recovered[case["slots"].long()],
                            intermediate[rows, accepted_steps.long()],
                            atol=ATOL,
                            rtol=RTOL,
                        )

    def test_padded_rows_do_not_touch_live_slots(self):
        """A short batch's padding rows carry slot -1 / step 0.

        What the contract promises, and what is checked: every VALID row commits its
        accepted state, and every slot outside the mapping keeps the bytes it had. The
        padded row's own destination is the reserved discard row (slot 0 is never
        handed to a request, since the allocator starts at 1), so its contents are not
        asserted -- the reference's intermediate row for a masked index is likewise
        not a defined value to compare against.
        """
        width = WIDTHS[0]
        case = self._case(3, 7, width)
        slots = torch.tensor([1, -1, 2], device=DEVICE, dtype=torch.int32)
        steps = torch.tensor([width - 1, 0, 0], device=DEVICE, dtype=torch.int32)
        # The reference runs the SAME mapping, so its valid rows start from the same
        # committed states the recovery sees.
        _, intermediate = self._full(case, case["state"].clone(), slots=slots)
        recovered = case["state"].clone()
        before = recovered.clone()
        # A live slot that this batch does not touch at all.
        bystander = int(slots.max()) + 2
        self.assertLess(bystander, case["state"].shape[0])

        self._recover(case, recovered, steps, slots=slots)

        for row, slot in ((0, 1), (2, 2)):
            torch.testing.assert_close(
                recovered[slot],
                intermediate[row, steps[row]],
                atol=ATOL,
                rtol=RTOL,
                msg=f"valid row {row} did not commit its accepted state",
            )
        untouched = [i for i in range(case["state"].shape[0]) if i not in (0, 1, 2)]
        torch.testing.assert_close(
            recovered[untouched], before[untouched], atol=0, rtol=0
        )
        torch.testing.assert_close(
            recovered[bystander], before[bystander], atol=0, rtol=0
        )

    # --- mamba-track boundary pass -----------------------------------------
    def test_boundary_prefixes_63_64_65(self):
        """The donor-qualified case: only the row at residue 63 can cross."""
        width = WIDTHS[0]
        case = self._case(3, 211, width)
        _, intermediate = self._full(case, case["state"].clone())
        recovered = case["state"].clone()

        track_slots = torch.tensor([4, 5, 6], device=DEVICE, dtype=torch.int32)
        track_steps = torch.tensor([0, -1, -1], device=DEVICE, dtype=torch.int32)
        crossed = track_steps >= 0
        normalized_slots = track_slots.clone().mul_(crossed)
        normalized_steps = track_steps.clamp(min=0)
        original_track = recovered[track_slots.long()].clone()

        self._recover(
            case,
            recovered,
            normalized_steps,
            output_slots=normalized_slots,
        )
        torch.testing.assert_close(
            recovered[track_slots[0].long()], intermediate[0, 0], atol=ATOL, rtol=RTOL
        )
        self.assertTrue(
            torch.equal(recovered[track_slots[1:].long()], original_track[1:]),
            "a non-crossing row must not clobber a live ping-pong checkpoint",
        )

        # And the working pass that follows still commits h_K in place.
        accepted_steps = torch.tensor([3, 1, 2], device=DEVICE, dtype=torch.int32)
        self._recover(case, recovered, accepted_steps)
        rows = torch.arange(3, device=DEVICE)
        torch.testing.assert_close(
            recovered[case["slots"].long()],
            intermediate[rows, accepted_steps.long()],
            atol=ATOL,
            rtol=RTOL,
        )

    def test_boundary_sweep_over_reachable_residues(self):
        """Crossing at residue r needs accept_lens >= 64-r, boundary step 63-r.

        So only r in [interval - width, interval - 1] is reachable by one draft
        window; every lower residue must arrive masked (step -1) and must not write.
        """
        for width in WIDTHS:
            # One row per reachable residue 64-width..63, plus one row for the
            # residue just below the reachable band (masked). Batch, slots and every
            # per-row vector below are derived from that same row count, and the
            # working slots (1..rows) are disjoint from the track slots
            # (rows+1..2*rows) -- the pool allocates 2*rows+8 rows, so both ranges fit.
            residues = list(range(TRACK_INTERVAL - width, TRACK_INTERVAL))
            residues = residues + [TRACK_INTERVAL - width - 1]
            batch_size = len(residues)
            case = self._case(batch_size, 313 + width, width)
            _, intermediate = self._full(case, case["state"].clone())
            recovered = case["state"].clone()
            rows = torch.arange(batch_size, device=DEVICE, dtype=torch.int32)
            slots = (rows + 1).to(torch.int32)
            track_slots = (rows + batch_size + 1).to(torch.int32)
            self.assertEqual(len(set(slots.tolist()) & set(track_slots.tolist())), 0)
            steps = torch.tensor(
                [TRACK_INTERVAL - 1 - r for r in residues],
                device=DEVICE,
                dtype=torch.int32,
            )
            unreachable = steps >= width
            self.assertTrue(bool(unreachable[-1]))
            crossed = ~unreachable
            track_steps = torch.where(
                crossed, steps, torch.full_like(steps, -1)
            ).contiguous()
            original_track = recovered[track_slots.long()].clone()

            self._recover(
                case,
                recovered,
                track_steps.clamp(min=0),
                output_slots=track_slots.clone().mul_(crossed),
                slots=slots,
            )
            for i, step in enumerate(track_steps.tolist()):
                if step < 0:
                    torch.testing.assert_close(
                        recovered[track_slots[i].long()],
                        original_track[i],
                        atol=0,
                        rtol=0,
                        msg=f"masked residue {residues[i]} wrote its track slot",
                    )
                    continue
                torch.testing.assert_close(
                    recovered[track_slots[i].long()],
                    intermediate[i, step],
                    atol=ATOL,
                    rtol=RTOL,
                    msg=f"residue {residues[i]} boundary state",
                )

    # --- drift, prefix reuse, graphs ---------------------------------------
    def test_prefix_restore_and_long_continuation(self):
        batch_size, width = 4, WIDTHS[0]
        reference = self._case(batch_size, 300, width)["state"]
        recovered = reference.clone()
        prefix_reference = None
        prefix_recovered = None
        replay_outputs = []

        def step(cycle, reference, recovered, recording):
            """One verify/commit cycle; returns the output the step was checked on."""
            case = self._case(batch_size, 301 + cycle, width)
            output_full, intermediate = self._full(case, reference)
            output_wy = self._wy(case, recovered)
            torch.testing.assert_close(output_wy, output_full, atol=ATOL, rtol=RTOL)
            accepted = (
                torch.arange(batch_size, device=DEVICE, dtype=torch.int32) + cycle
            ) % width
            self._commit_reference(reference, case["slots"], intermediate, accepted)
            self._recover(case, recovered, accepted)
            torch.testing.assert_close(
                recovered[case["slots"].long()],
                reference[case["slots"].long()],
                atol=ATOL,
                rtol=RTOL,
            )
            if recording:
                replay_outputs.append(output_wy.clone())
            return output_wy

        for cycle in range(24):
            step(cycle, reference, recovered, recording=cycle >= 16)
            if cycle == 15:
                prefix_reference = reference.clone()
                prefix_recovered = recovered.clone()

        # Simulate radix prefix reuse / retraction: restore the saved checkpoint into
        # fresh working pools and replay the continuation.
        # Same phase as the recording: the output a replayed step produced from the
        # state it STARTED from, not a later read of the pool it just advanced (that
        # would compare a pre-commit verify against a post-commit one).
        reference, recovered = prefix_reference, prefix_recovered
        for replay_index, cycle in enumerate(range(16, 24)):
            output_replay = step(cycle, reference, recovered, recording=False)
            torch.testing.assert_close(
                output_replay,
                replay_outputs[replay_index],
                atol=ATOL,
                rtol=RTOL,
            )

    def test_captured_recovery_graph_batches_and_replays(self):
        for width in WIDTHS:
            for batch_size in (1, 4):
                with self.subTest(width=width, batch_size=batch_size):
                    case = self._case(batch_size, 500 + batch_size, width)
                    _, intermediate = self._full(case, case["state"].clone())
                    state = case["state"].clone()
                    accepted = (
                        torch.arange(batch_size, device=DEVICE, dtype=torch.int32)
                        % width
                    )

                    # Warm compilation and all default-buffer allocation first, so
                    # the capture itself is JIT-free and alloc-free.
                    warm = state.clone()
                    self._recover(case, warm, accepted)
                    torch.cuda.synchronize()

                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        self._recover(case, state, accepted)

                    rows = torch.arange(batch_size, device=DEVICE)
                    for _ in range(2):
                        state.copy_(case["state"])
                        graph.replay()
                        torch.cuda.synchronize()
                        torch.testing.assert_close(
                            state[case["slots"].long()],
                            intermediate[rows, accepted.long()],
                            atol=ATOL,
                            rtol=RTOL,
                        )
                        # Same graph, different accepted widths: the indices are read
                        # from the captured buffers, not baked in.
                        accepted.copy_((accepted + 1) % width)

    # --- warmup capture ordering -------------------------------------------
    def _warmup_backend(self, case, pool):
        """Wire the real FlashInfer recovery path over a real none-mode pool.

        Every eligibility input is the production one, so nothing can exit early and
        pass vacuously: ``kernel_dispatcher.decode_kernel`` is a genuine
        ``FlashInferGDNKernel`` (SM120 -> ``use_state_pool`` True), so
        ``fi_recovery_kernel`` selects the WY recovery route; ``state_pool`` is a real
        ``MambaPool``, which is where the recovery event the HiCache write leg joins
        lives, and whose ``mamba2_layer_cache`` supplies the state rows; the stash and
        persistent conv-out buffer carry the layout ``_persist_kv`` reads.
        """
        from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
            HybridLinearAttnBackend,
        )
        from sglang.srt.layers.attention.linear.kernels.gdn_flashinfer import (
            FlashInferGDNKernel,
        )

        width = case["T"]
        batch_size = case["q"].shape[0]
        q_dim = k_dim = H * K
        v_dim = HV * V
        conv_dim = q_dim + k_dim + v_dim
        # Token-major persistent conv-out buffer: [rows, cache_steps, conv_dim] with k
        # at [q_dim : q_dim+k_dim] and v after it -- the slices _persist_kv takes as
        # strided views. case["k"]/["v"] are [B, T, H, D], so the head axes have to be
        # flattened into the column axis before they can land in the buffer.
        persist = torch.zeros(
            pool.size + 1, width, conv_dim, device=DEVICE, dtype=case["k"].dtype
        )
        persist[:batch_size, :, q_dim : q_dim + k_dim] = case["k"].reshape(
            batch_size, width, k_dim
        )
        persist[:batch_size, :, q_dim + k_dim : q_dim + k_dim + v_dim] = case[
            "v"
        ].reshape(batch_size, width, v_dim)

        temporal = pool.mamba2_layer_cache(0).temporal
        # Row 0 is the pool's reserved padding slot (rows are size+1), which is exactly
        # the discard row the warmup is allowed to write.
        self.assertEqual(
            tuple(temporal.shape),
            tuple(case["state"].shape),
            "the case must be built against this pool's geometry",
        )
        temporal.copy_(case["state"])

        backend = HybridLinearAttnBackend.__new__(HybridLinearAttnBackend)
        backend._recover_ssm = True
        backend._recovery_stream = None
        backend._recovery_event = None
        backend._recovery_event_pending = False
        backend._recovery_join_event = None
        backend._recovery_stream_ctx = None
        zeros = lambda: torch.zeros(  # noqa: E731 - four stable buffers, same shape
            pool.size + 1, dtype=torch.int32, device=DEVICE
        )
        backend._rec_state_idx_buf = zeros()
        backend._rec_acc_steps_buf = zeros()
        backend._rec_track_idx_buf = zeros()
        backend._rec_track_steps_buf = zeros()
        backend._rec_graphs = {}
        backend._rec_boundary_graphs = {}
        backend._rec_capture_bs = None
        # What the glue graph snapshots and reinstalls around the prep it records.
        backend.forward_metadata = SimpleNamespace(layer_id=0)
        backend.attn_backend_list = []
        sidecar = SimpleNamespace()
        sidecar.kernel_dispatcher = SimpleNamespace(decode_kernel=FlashInferGDNKernel())
        sidecar._no_cache_draft_token_num = width
        sidecar._conv_out_persist = {0: persist}
        sidecar._no_cache_stash = {
            0: {
                "a": case["a"],
                "b": case["b"],
                "A_log": case["A_log"],
                "A_log_f32": case["A_log"],
                "dt_bias": case["dt_bias"],
                "conv_dims": (q_dim, k_dim, v_dim, H, K, HV, V),
            }
        }
        sidecar.state_pool = pool
        sidecar.state_layer_cache = lambda layer_id: pool.mamba2_layer_cache(layer_id)
        backend.linear_attn_backend = sidecar
        return backend, temporal

    def _no_buffer(self):
        from sglang.srt.runtime_context import get_context

        return get_context().override_server_args(
            mamba_radix_cache_strategy="no_buffer"
        )

    def test_warmup_capture_waits_the_delayed_caller_stream_producer(self):
        """The capture's producer->consumer join, on hardware, with a slow producer.

        The four stable index/step buffers are zero-filled on the caller stream and
        the stash was written there; the first warmup launch runs on the recovery
        stream. Poison the working-index buffer with a LIVE slot and a non-zero
        accepted step, keep the caller stream busy behind that fill, and capture. The
        join must hold the launch until the fills land, so the only row the warmup can
        touch is the reserved discard row. Without it the recovery stream is idle, the
        launch reads the poison, and it commits an accepted prefix into the live
        request's row. The kernel is warm from a first capture pass, so a cold JIT
        cannot mask the missing dependency; the replay at the end proves the captured
        graph is a real recovery rather than a no-op.

        The CPU seam test (test_gdn_recovery_stream_order) pins the event order
        symbolically; this is the version that can fail on a real reordering.
        """
        width = WIDTHS[0]
        case = self._case(2, 909, width)
        pool = _make_pool(case["state"].shape[0] - 1, width, "none")
        backend, temporal = self._warmup_backend(case, pool)
        batch_size = case["q"].shape[0]
        live_slot = int(case["slots"][0])  # slot 1: a real request row
        before = temporal.clone()
        _, intermediate = self._full(case, case["state"].clone())

        with self._no_buffer():
            # Prewarm: JIT/arg-default population for this exact geometry.
            backend.capture_recovery_graphs([batch_size])
            self.assertEqual(backend._rec_capture_bs, [batch_size])
            backend._rec_graphs.clear()
            temporal.copy_(case["state"])

            # Producer work still in flight on the caller stream: a poison index and
            # step, then a ~100ms delay before the zero-fill the capture issues.
            backend._rec_state_idx_buf.fill_(live_slot)
            backend._rec_acc_steps_buf.fill_(width - 1)
            torch.cuda._sleep(200_000_000)
            backend.capture_recovery_graphs([batch_size])
            torch.cuda.synchronize()

            self.assertEqual(backend._rec_capture_bs, [batch_size])
            self.assertIn(batch_size, backend._rec_graphs)
            self.assertEqual(int(backend._rec_state_idx_buf[0]), 0)
            untouched = [i for i in range(temporal.shape[0]) if i != 0]
            torch.testing.assert_close(
                temporal[untouched],
                before[untouched],
                atol=0,
                rtol=0,
                msg="warmup recovery wrote a live slot: the capture did not join its "
                "caller-stream producer before the first launch",
            )

            # Not vacuous: the graph that was captured does the recovery. Point it at
            # the real slots and the last draft position and replay.
            backend._rec_state_idx_buf[:batch_size] = case["slots"]
            backend._rec_acc_steps_buf[:batch_size] = width - 1
            backend._rec_state_idx_buf[batch_size:] = 0
            backend._rec_acc_steps_buf[batch_size:] = 0
            backend._recovery_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(backend._recovery_stream):
                backend._rec_graphs[batch_size].replay()
            torch.cuda.synchronize()
            for row in range(batch_size):
                torch.testing.assert_close(
                    temporal[int(case["slots"][row])],
                    intermediate[row, width - 1],
                    atol=ATOL,
                    rtol=RTOL,
                    msg="captured recovery graph did not commit the accepted state",
                )

    def test_metadata_replay_cannot_order_the_recovery_the_seam_does(self):
        """Already-captured metadata replay orders nothing; only the seam does.

        A previous step's recovery is still running on the side stream (a delayed
        sentinel write into a live row) while this step's metadata prep is served by a
        glue-graph REPLAY, which runs no Python prep and issues no wait. Two phases,
        identical timing, one difference: the second calls drain_pending_recovery --
        the call ModelRunner._forward_raw now makes at the dispatch seam, before the
        cuda-graph branch returns -- before the step's own write to that row, and that
        write wins; the first skips it, which is what the graph-dispatch path used to
        do, and the delayed side stream lands last and wins. Same work, opposite
        result, so neither phase can pass vacuously.

        The key is captured BEFORE any producer is queued, deliberately:
        torch.cuda.graph's context enter calls torch.cuda.synchronize() (then GC and
        empty_cache) on a real capture, which drains every stream on the device. A
        cold capture inside the race would erase the race, and the control would then
        measure the capture rather than the replay -- and the replay is the only thing
        serving ever executes for a key after its first step.
        """
        from sglang.srt.model_executor.runner.metadata_glue_graph import (
            MetadataGlueGraph,
        )

        width = WIDTHS[0]
        case = self._case(2, 404, width)
        pool = _make_pool(case["state"].shape[0] - 1, width, "none")
        # The prep under test is the wrapper's own delegation loop, with no children.
        backend, temporal = self._warmup_backend(case, pool)
        row = int(case["slots"][0])
        event = pool.state_recovery_event
        self.assertIsInstance(event, torch.cuda.Event)
        glue = MetadataGlueGraph(device=DEVICE)
        glue.NUM_WARMUP = 0  # capture this key now, while nothing is in flight
        key = "target-verify"
        sentinel = lambda value: torch.full_like(temporal[row], value)  # noqa: E731

        def delayed_recovery(value):
            """The previous step's recovery: busy ~100ms, then commits `value`."""
            backend._ensure_recovery_stream()
            with torch.cuda.stream(backend._recovery_stream):
                torch.cuda._sleep(200_000_000)
                temporal[row].fill_(value)
                event.record()
            backend._recovery_event_pending = True

        with self._no_buffer():
            glue.run(backend, SimpleNamespace(), key)
            self.assertTrue(glue.is_captured(key), "the key must be captured up front")
            self.assertFalse(glue.disabled)
            self.assertEqual(glue.eager, 0)
            # The capture path replays once to do its own step's prep without counting
            # it as a replay, so count from here rather than assuming a value.
            replays_captured = glue.replays
            self.assertFalse(
                backend._recovery_event_pending,
                "nothing may be in flight while the key is captured",
            )

            # Control: the replay consumes nothing and orders nothing, so with no seam
            # drain the side stream's commit is the last writer to that row.
            delayed_recovery(1.0)
            glue.run(backend, SimpleNamespace(), key)
            self.assertTrue(
                backend._recovery_event_pending,
                "metadata replay consumed a join it never establishes",
            )
            self.assertEqual(glue.replays, replays_captured + 1)
            temporal[row].fill_(0.5)  # this step's own write, unjoined
            torch.cuda.synchronize()  # the only global sync in the phase
            torch.testing.assert_close(
                temporal[row],
                sentinel(1.0),
                atol=0,
                rtol=0,
                msg="control: the unjoined step won the row, so a missing join is "
                "not observable here and this test cannot fail",
            )

            # With the seam drain, the same write is ordered after the recovery.
            delayed_recovery(2.0)
            glue.run(backend, SimpleNamespace(), key)
            self.assertTrue(backend._recovery_event_pending)
            backend.drain_pending_recovery()
            self.assertFalse(backend._recovery_event_pending)
            temporal[row].fill_(3.0)
            torch.cuda.synchronize()
            torch.testing.assert_close(
                temporal[row],
                sentinel(3.0),
                atol=0,
                rtol=0,
                msg="the step's write must land after the joined recovery, per element "
                "(a row partially overwritten by both is corruption, not a tie)",
            )
            self.assertEqual(glue.replays, replays_captured + 2)
            self.assertEqual(glue.eager, 0, "both phases must use the captured replay")

    def test_state_pool_publishes_the_recovery_fence(self):
        """The pool, not the backend, owns the event the HiCache write leg joins.

        Checked on real hardware because it is allocated only for a CUDA state pool
        (see MambaPool.state_recovery_event and the CPU seam tests).
        """
        none_pool = _make_pool(2, WIDTHS[0], "none")
        full_pool = _make_pool(2, WIDTHS[0], "full")
        # The event the HiCache write leg joins exists exactly when the mode records
        # it, and it is the same object across every read of one pool.
        self.assertIsInstance(none_pool.state_recovery_event, torch.cuda.Event)
        self.assertIs(none_pool.state_recovery_event, none_pool.state_recovery_event)
        self.assertIsNone(full_pool.state_recovery_event)
        # Full mode keeps its per-draft snapshots; none mode must not pay for them
        # (the accounting the solver charges is this pool's allocation), and the conv
        # rollback windows stay in both.
        self.assertIsNotNone(full_pool.mamba_cache.intermediate_ssm)
        self.assertIsNone(none_pool.mamba_cache.intermediate_ssm)
        self.assertTrue(none_pool.mamba_cache.intermediate_conv_window)
        self.assertTrue(full_pool.mamba_cache.intermediate_conv_window)


if __name__ == "__main__":
    unittest.main()
