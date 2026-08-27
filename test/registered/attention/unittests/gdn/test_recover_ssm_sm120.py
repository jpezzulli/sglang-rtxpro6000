"""SM120 numerical regressions for GDN MTP RecoverSSM.

These tests use Penny's active T=4, H=16, HV=48, K=V=128 geometry. They compare
the WY output-only verifier and accepted-state recovery against the full
intermediate-state reference, including mixed acceptance, radix-boundary
recovery, repeated continuation, simulated prefix restore, and CUDA graphs.
"""

import unittest

import torch

from sglang.srt.utils import is_flashinfer_available
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=240, stage="base-b", runner_config="1-gpu-large")

T, H, HV, K, V = 4, 16, 48, 128, 128
DEVICE = "cuda"
ATOL = 4e-2
RTOL = 4e-2


def _sm_major() -> int:
    return torch.cuda.get_device_capability()[0] if torch.cuda.is_available() else 0


@unittest.skipUnless(
    torch.cuda.is_available() and is_flashinfer_available() and _sm_major() == 12,
    "SM120 with FlashInfer is required",
)
class TestRecoverSSMSM120(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        from flashinfer.gdn_kernels import gated_delta_rule_mtp_wy_output_only
        from flashinfer.gdn_kernels.gdn_decode_bf16_state import gated_delta_rule_mtp

        cls.full_kernel = staticmethod(gated_delta_rule_mtp)
        cls.wy_kernel = staticmethod(gated_delta_rule_mtp_wy_output_only)

    @staticmethod
    def _case(batch_size: int, seed: int):
        gen = torch.Generator(device=DEVICE).manual_seed(seed)

        def rnd(*shape, dtype=torch.bfloat16):
            return torch.randn(*shape, device=DEVICE, dtype=dtype, generator=gen)

        pool_size = batch_size * 2 + 4
        slots = torch.arange(batch_size, device=DEVICE, dtype=torch.int32) + 1
        return {
            "A_log": (rnd(HV, dtype=torch.float32) * 0.1).float(),
            "dt_bias": (rnd(HV, dtype=torch.float32) * 0.1).float(),
            "q": rnd(batch_size, T, H, K),
            "k": rnd(batch_size, T, H, K),
            "v": rnd(batch_size, T, HV, V),
            "a": rnd(batch_size, T, HV),
            "b": rnd(batch_size, T, HV),
            "state": rnd(pool_size, HV, V, K),
            "slots": slots,
        }

    @classmethod
    def _full(cls, case, state):
        batch_size = case["q"].shape[0]
        intermediate = torch.empty(
            batch_size,
            T,
            HV,
            V,
            K,
            device=DEVICE,
            dtype=torch.bfloat16,
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
    def _recover(cls, case, state, accepted_steps, output_slots=None):
        output_slots = case["slots"] if output_slots is None else output_slots
        cls.full_kernel(
            A_log=case["A_log"],
            a=case["a"],
            dt_bias=case["dt_bias"],
            q=case["k"],
            k=case["k"],
            v=case["v"],
            b=case["b"],
            initial_state_source=state,
            initial_state_indices=case["slots"],
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

    def test_wy_and_recovered_state_all_batches_and_accept_lengths(self):
        for batch_size in range(1, 5):
            with self.subTest(batch_size=batch_size):
                case = self._case(batch_size, 100 + batch_size)
                output_full, intermediate = self._full(case, case["state"].clone())
                output_wy = self._wy(case, case["state"].clone())
                torch.testing.assert_close(output_wy, output_full, atol=ATOL, rtol=RTOL)

                patterns = [
                    torch.full((batch_size,), step, device=DEVICE, dtype=torch.int32)
                    for step in range(T)
                ]
                if batch_size > 1:
                    patterns.append(
                        torch.arange(batch_size, device=DEVICE, dtype=torch.int32) % T
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

    def test_extra_buffer_boundary_positions_63_64_65(self):
        case = self._case(3, 211)
        _, intermediate = self._full(case, case["state"].clone())
        recovered = case["state"].clone()

        # A T=4 verify beginning at committed lengths 63, 64, and 65 crosses
        # the interval-64 boundary only in row 0, at accepted step 0.
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
            recovered[track_slots[0].long()],
            intermediate[0, 0],
            atol=ATOL,
            rtol=RTOL,
        )
        self.assertTrue(
            torch.equal(recovered[track_slots[1:].long()], original_track[1:])
        )

        accepted_steps = torch.tensor([3, 1, 2], device=DEVICE, dtype=torch.int32)
        self._recover(case, recovered, accepted_steps)
        rows = torch.arange(3, device=DEVICE)
        torch.testing.assert_close(
            recovered[case["slots"].long()],
            intermediate[rows, accepted_steps.long()],
            atol=ATOL,
            rtol=RTOL,
        )

    def test_prefix_restore_and_long_continuation(self):
        batch_size = 4
        reference = self._case(batch_size, 300)["state"]
        recovered = reference.clone()
        prefix_reference = None
        prefix_recovered = None
        replay_outputs = []

        for cycle in range(24):
            case = self._case(batch_size, 301 + cycle)
            case["state"] = reference
            output_full, intermediate = self._full(case, reference)
            output_wy = self._wy(case, recovered)
            torch.testing.assert_close(output_wy, output_full, atol=ATOL, rtol=RTOL)
            accepted = (
                torch.arange(batch_size, device=DEVICE, dtype=torch.int32) + cycle
            ) % T
            self._commit_reference(reference, case["slots"], intermediate, accepted)
            self._recover(case, recovered, accepted)
            torch.testing.assert_close(
                recovered[case["slots"].long()],
                reference[case["slots"].long()],
                atol=ATOL,
                rtol=RTOL,
            )
            if cycle == 15:
                prefix_reference = reference.clone()
                prefix_recovered = recovered.clone()
            if cycle >= 16:
                replay_outputs.append(output_wy.clone())

        # Simulate radix prefix reuse / request retraction by restoring the saved
        # checkpoint into fresh working pools and replaying the continuation.
        reference = prefix_reference
        recovered = prefix_recovered
        for replay_index, cycle in enumerate(range(16, 24)):
            case = self._case(batch_size, 301 + cycle)
            case["state"] = reference
            output_full, intermediate = self._full(case, reference)
            output_wy = self._wy(case, recovered)
            torch.testing.assert_close(
                output_wy, replay_outputs[replay_index], atol=ATOL, rtol=RTOL
            )
            torch.testing.assert_close(output_wy, output_full, atol=ATOL, rtol=RTOL)
            accepted = (
                torch.arange(batch_size, device=DEVICE, dtype=torch.int32) + cycle
            ) % T
            self._commit_reference(reference, case["slots"], intermediate, accepted)
            self._recover(case, recovered, accepted)
            torch.testing.assert_close(
                recovered[case["slots"].long()],
                reference[case["slots"].long()],
                atol=ATOL,
                rtol=RTOL,
            )

    def test_captured_recovery_graph_batches_1_to_4(self):
        for batch_size in range(1, 5):
            with self.subTest(batch_size=batch_size):
                case = self._case(batch_size, 500 + batch_size)
                _, intermediate = self._full(case, case["state"].clone())
                state = case["state"].clone()
                accepted = (
                    torch.arange(batch_size, device=DEVICE, dtype=torch.int32) % T
                )

                # Warm compilation and all default-buffer allocation before capture.
                warm_state = state.clone()
                self._recover(case, warm_state, accepted)
                torch.cuda.synchronize()

                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    self._recover(case, state, accepted)

                state.copy_(case["state"])
                graph.replay()
                torch.cuda.synchronize()
                rows = torch.arange(batch_size, device=DEVICE)
                torch.testing.assert_close(
                    state[case["slots"].long()],
                    intermediate[rows, accepted.long()],
                    atol=ATOL,
                    rtol=RTOL,
                )

                accepted.copy_((accepted + 1) % T)
                state.copy_(case["state"])
                graph.replay()
                torch.cuda.synchronize()
                torch.testing.assert_close(
                    state[case["slots"].long()],
                    intermediate[rows, accepted.long()],
                    atol=ATOL,
                    rtol=RTOL,
                )


if __name__ == "__main__":
    unittest.main()
