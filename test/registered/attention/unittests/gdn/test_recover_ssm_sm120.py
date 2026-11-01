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
    def _full(cls, case, state):
        """Reference: state-writing verifier, h at every draft position."""
        batch_size, width = case["q"].shape[:2]
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
            initial_state_indices=case["slots"],
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

        The recovery must leave every real slot alone for those rows and write only
        the reserved discard row, or a padded launch corrupts a live request.
        """
        width = WIDTHS[0]
        case = self._case(3, 7, width)
        slots = torch.tensor([1, -1, 2], device=DEVICE, dtype=torch.int32)
        _, intermediate = self._full(case, case["state"].clone())
        recovered = case["state"].clone()
        before = recovered.clone()
        steps = torch.tensor([width - 1, 0, 0], device=DEVICE, dtype=torch.int32)
        self._recover(case, recovered, steps, slots=slots)
        # Row 1 (padded) resolved to h_0 -> discard row 0; rows 0 and 2 committed.
        torch.testing.assert_close(
            recovered[1], before[0], atol=0, rtol=0, msg="padded row wrote a live slot"
        )
        for row, slot in ((0, 1), (2, 2)):
            torch.testing.assert_close(
                recovered[slot],
                intermediate[row, steps[row]],
                atol=ATOL,
                rtol=RTOL,
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
            case = self._case(width + 2, 313 + width, width)
            _, intermediate = self._full(case, case["state"].clone())
            recovered = case["state"].clone()
            residues = list(range(TRACK_INTERVAL - width, TRACK_INTERVAL))
            # One row per reachable residue, then rows for the residue below the
            # reachable band, which must be masked out.
            residues = residues + [TRACK_INTERVAL - width - 1]
            rows = torch.arange(len(residues), device=DEVICE, dtype=torch.int32)
            slots = (rows + 1).to(torch.int32)
            track_slots = (rows + width + 1).to(torch.int32)
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

        for cycle in range(24):
            step(cycle, reference, recovered, recording=cycle >= 16)
            if cycle == 15:
                prefix_reference = reference.clone()
                prefix_recovered = recovered.clone()

        # Simulate radix prefix reuse / retraction: restore the saved checkpoint into
        # fresh working pools and replay the continuation.
        reference, recovered = prefix_reference, prefix_recovered
        for replay_index, cycle in enumerate(range(16, 24)):
            step(cycle, reference, recovered, recording=False)
            torch.testing.assert_close(
                self._wy(self._case(batch_size, 301 + cycle, width), recovered),
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

    def test_state_pool_publishes_the_recovery_fence(self):
        """The pool, not the backend, owns the event the HiCache write leg joins.

        Checked on real hardware because it is allocated only for a CUDA state pool
        (see MambaPool.state_recovery_event and the CPU seam tests).
        """
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
        base = dict(
            size=2,
            spec_state_size=2,
            cache_params=params,
            mamba_layer_ids=[0],
            device="cuda",
            speculative_num_draft_tokens=WIDTHS[0],
        )
        none_pool = MambaPool(gdn_mtp_cache_mode="none", **base)
        full_pool = MambaPool(gdn_mtp_cache_mode="full", **base)
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
