# GDN MTP `--gdn-mtp-cache-mode none`: RecoverSSM and WY verifier parity

Candidate: base `28c2d24eab4f58132ec31dde76e63e145e07009d` (upstream `e3718813ec5f47df0ad368b9a4713fbd3ad0a3bf`).
Behaviour ported from the `feat/gdn-mtp-recover-ssm` donor (branch SHA
`284b00609830f450747407c3d675e3128ae5c682`, `real_impl.txt` SHA
`a7051d63d102839802686c8e2482cd205e219215d161ac2a0e666c123f032133`) onto current
upstream interfaces. No GPU work was run here; everything below is CPU-verified or
marked for Codex.

`none` mode deletes the per-step `intermediate_states_buffer` snapshot from decode-target
verify: the GDN verify kernel runs output-only through FlashInfer's `gated_delta_rule_mtp`
(WY, any prefix) and the target rewrites the recurrent state itself. Two facts make that
safe, and both are things the old design asserted without ever checking:

1. A suffix recurrence is exact in bf16 **when it is seeded with the exact state at that
   boundary** (`h_K`). Recovering `h_K` from `h_0` across the window costs 4.88% bit flips
   at `(2,128,1,128)`; seeding the kernel with the reference `h_K` costs 0.000% — the loss
   is entirely the recovery hop.
2. The consumed prefix length is a **known** integer (`accept_lens` minus one), not a
   distance measured from a delayed second launch.

So the port is: construct the accepted state, order it with a stream event, and fail loudly
at four runtime boundaries (the full list is in the review-findings section below).

## What this adds

* **`--gdn-mtp-cache-mode {full,none}`**, one new field in `arg_groups/fields/exec_.py`
  plus its `field_order.py` entry, one validation/resolution hook in `arg_groups/mamba_hook.py`
  and the new `arg_groups/resolution_hooks.py` module (registered in `arg_groups/pipeline.py`).
  Import-time resolution only; no env override; no `os.environ` read on any hot path.
* **Output-only WY verify routing** in `layers/attention/linear/gdn_backend.py` +
  `linear/kernels/gdn_flashinfer.py` (capability gate `can_target_verify(mode)`).
* **Output-only conv inputs** in `kernels/ops/mamba/causal_conv1d_triton.py` and
  `kernels/ops/attention/linear/fused_sigmoid_gating_recurrent.py`.
* **RecoverSSM commit** in `layers/attention/hybrid_linear_attn_backend.py` (side stream,
  captured recovery graph, mamba-track boundary pass, drain-before-deferred-COW/clear).
* **Per-request state-tracking slot** in `configs/mamba_utils.py`, `mem_cache/memory_pool.py`,
  `mem_cache/kv_cache_configurator.py`, `speculative/spec_utils.py` and
  `speculative/eagle_worker_v2.py` (`TRACK_NONE` masking, `track_slot[req]` publication).
* **NIXL whole-page state export** in `disaggregation/nixl/sender.py` / `receiver.py`, reusing
  the existing `state_indices`/`state_page_indices` fields: no new wire field, and V2-only
  receivers keep parsing the frame unchanged.
* **HiCache L2 ordering** in `managers/cache_controller.py` and `mem_cache/hi_mamba_cache.py`.
* **No pre-enqueue preverify restore and no async D2H copy worker.** Both are donor-side
  history: `_recover_ssm_track`, `mamba_preverify_restore` and `async_copy_server_request`
  are absent from base `28c2d24` as well as from this head, so the port's claim is that it
  does not reintroduce them (the launch-gate pair from the earlier draft is likewise gone).
  The net patch is additive: 29 files, +4070/-70 against `28c2d24`, deletions concentrated in
  the two attention backends and the doc regeneration. `docs/advanced_features/
  server_arguments.mdx` is regenerated.

## Tests

CPU (this host, `pytest` with `PYTHONPATH=python`):

* `test/registered/unit/server_args/test_gdn_mtp_cache_mode.py` (24 tests, 20 subtests) —
  mode parsing, every rejection, the MTP auto-downgrade, the kernel-capability gate and the
  SM120 routing table. It runs the real policy through both `_mamba_alias_rewrite` and
  `_resolve_gdn_mtp_cache_mode` rather than a copy, because the hooks are the shipped
  behaviour. There is no repo-wide argument-table test in this tree (nothing outside this
  file's README mentions `generate_args_as_tuple`), so the surface check for the new field is
  the whole directory: `test/registered/unit/server_args/` is 468 passed with 5 failures, and
  those 5 are identical on pristine `28c2d24` (444 passed, same 5) — plugin-parser, gRPC
  signature, multimodal-transport and lazy-reexport environment cases, plus a pre-existing
  lazy-import purity failure that my new `arg_groups/resolution_hooks.py` does not add to.
* `test/registered/unit/mem_cache/test_spec_mamba_buffer_accounting.py` — the per-request
  accounting: conv window always charged, SSM snapshots charged only for modes that allocate
  them (this is what makes the ~189 MB/GB and ~1.18 GB/req numbers reviewable), dedup across
  dense layouts, TP/PP sharding, the padding slot, PD-decode prealloc slots, the draft bound
  (incl. ` speculative_eagle_topk > 1`), and `mamba_cache_margin=1.2`. `_calculate_mamba_ratio`
  is read from the shipped module (`from ... import _calculate_mamba_ratio as cr`), and
  `_layer_num` comes from the published config, never a `server_args` proxy that would return
  `None` for `pp_size`.
* `test/registered/unit/layers/attention/test_gdn_recovery_stream_order.py` — the ordering
  rules, in the shape production uses: one drain per forward call at the post-forward seam;
  the recovery event is the pool-owned instance, and every consumer that captures the
  reference takes it from `pool.mamba2_layer_cache(0).temporal.device` on **every** call, so
  a second backend or a PD role switch cannot publish a stale event; a missing pool event is
  refused rather than replaced by a private one; masked residues cannot reach the
  `64 - accept_lens` arithmetic; a per-backend stream cannot fence the HiCache L2 legs.
  Plus three **constructor-level** regressions that run the real functions instead of
  tracing callees (review 1): `GDNAttnBackend.__init__` must forward the mode into
  `GDNKernelDispatcher` (four args, or startup is a `TypeError`),
  `HybridLinearAttnBackend.__init__` must create exactly one side stream and keep the
  pool-owned event (the old order touched `_recovery_stream` before it existed, and
  `_ensure_recovery_stream` discarded and recreated the pool event), and
  `ModelRunner.init_cuda_graphs` must still run the full post-capture tail (elastic-EP
  recapture, `finalize_cuda_graph_capture`, the `_finalize_elastic_ep_joiner` pass,
  `prebuild_deepep_v2_buffer`) with the recovery capture last.
  Also here: `TestWarmupCaptureJoinsTheCallerStream`, the seam test for the warmup-capture
  handoff (below).

GPU (for Codex; all skipped without CUDA — 12 tests collected, none executed here):

* `test/registered/attention/unittests/gdn/test_recover_ssm_sm120.py` (8): `full` vs `none`
  output/state equality at 4 widths and 7 shapes with 63/64/65 boundary crossings;
  `-0.0 == +0.0` and `NaN==NaN` as the **real** `torch.equal` (a `float('nan')` in the
  production wrapper short-circuits `nan != nan` to `False`, so a wrapper-level assertion is
  decoration); a bit-flip **and** a max-ULP report, with the statistic taken from the
  reference-vs-candidate **outputs** actually produced in the test body at a shape the test
  really uses, not from a hand-picked shape or a separately recomputed kernel call; padded
  rows leave live slots untouched and the discard row is the reserved one; the boundary
  residues reachable at each decode width (incl. `num_steps=8`), each with its own working
  slot and its own expected state; a tracked row restored from the published boundary state
  matching row-by-row, and a tracked mask that cannot reach the `64 - accept_lens`
  arithmetic; recovery inside a captured decode graph replaying against a pool restored from
  an external commit; and `test_warmup_capture_waits_the_delayed_caller_stream_producer`,
  which poisons the stable index buffers with a live slot behind a `torch.cuda._sleep` on the
  caller stream and requires the capture's join to keep the warmup launch out of that row.
* `test/manual/4-gpu-models/test_gdn_mtp_recover_ssm.py` (4): Qwen3.5-4B (`tp=1`), 16k
  contiguous shared-prefix requests, per-step accepted-lengths ledger, GSM8K with
  `max_tokens=4096`, and a `full`+ReplaySSM speedup case (absolute values, so no CI
  threshold). The acceptance budget lives in `_assert_accepted_lengths`, with the
  **measured** M2 accept-length mean as a floor so an environment that cannot reach the
  budget fails loudly instead of reporting zero accepted lengths.
  Launched with `--gdn-mtp-cache-mode none`; runtime evidence for the mode is the
  `[gdn_mtp_cache_mode]` line in the server log. The mode is also asserted through the
  server's own `/server_info` route, so a launch that silently dropped the flag fails.
* `test/registered/attention/unittests/gdn/README.md` (index) carries the launch recipe:
  pass `--gdn-mtp-cache-mode none`, and add `--field "gdn_mtp_cache_mode=$GDN_MTP_CACHE_MODE"`
  to `scripts/pennyroyal/derive_namespace.py`, so a mode flip produces a fresh cache
  namespace and stale NIXL state data from a `full` launch cannot be reused.

## Review findings and how each was closed

1. **`gdn_mtp_cache_mode` was never applied** (M2, the accepted target) and
   **`track_dsts` had no consumer** (M1). Both hooks now load the mode through
   `sglang.srt.runtime_context`, and the second call site (`_commit_mamba_track_boundary`)
   was reduced to `plan_spec_post_mamba_commit`, which takes `track_dsts` as an argument
  (that helper does not import `runtime_context`, which is why it is not left alone in a
  by-name search). `scripts/playground/` was excluded from the search: it contains
   generated `server_args.md`, and generated output is not integration.
2. **The recovery event was not pool-owned** at two of its three publication/retention
   points. `MambaPool` now owns it and `register_state_recovery_event` raises if a second
   backend tries to replace it; `gdn_backend` registers it (with a layer id), takes a
   constructor argument for the mode, and `HybridLinearAttnBackend` refuses a private
   fallback.
3. **The consumer joined nothing.** `_record_mamba_recovery` records on the producer stream
   and `wait_event`s on the consumer; both L2 legs wait the pool event before submitting their
   D2H/H2D, so the existing ACK-gated host publication is preserved. The capture path's
   handoff was added in review 3 (see item 6) — in the accepted source its only ordering was a
   synchronize *before* the launch, which protected the warm-compile pass; the launch inside
   the capture and the graph's own later sync did not need the added sync, so it was removed
   rather than kept as decoration.
4. **Falsifiability.** The event-identity test asserts against
   `pool.mamba2_layer_cache(0).temporal.device`; the track-mask test is a
   `(batch_size, num_decode_steps)` matrix carrying two masked rows and one row crossing at
   `TRACK_INTERVAL - accept_lens`. Both fail if the production code changes.
5. **Tests that cannot fail.** The real-kernel NaN test had a hand-computed expected
   trajectory (verified by hand) and now skips without FlashInfer; the NIXL integration test
   drives the real `_send_mamba_states` against a real `MambaPool` and real
   `Mamba2StateIPCLauncher`. A `test/registered/unit/managers/test_mamba_preverify_restore.py`
   deletion check exists; the old "zero launches" and "zero async copies" probes were dropped
   (they passed at zero), as was any "does the module import" test. The SM120 test's
   `torch.equal` assertion was also vacuous in a second way: a `float('nan')` passed in the
   test's own wrapper makes `nan != nan` false, so it never saw a NaN at all.
   Review 2 found the deletion guard itself was not deletion-proof (it used
   `find_spec("sglang.srt.managers.async_copy_server_request")`, which passes for a module
   living on disk but un-imported; the files are now deleted and the check asserts the
   attribute is absent).
   Correction found while writing this: an earlier draft of this README claimed the port
  deleted the donor's preverify-restore worker and its
  `test/registered/mem_cache/test_mamba_unittest.py` suite. Neither exists at base
  `28c2d24eab4f58132ec31dde76e63e145e07009d` (`git cat-file -e` fails for both paths), so
  there was nothing to delete and the deletion guards assert the absence of things that were
  already absent — still worth keeping as regressions, but described honestly.
  Review 2 also found a real crash on the SM120 file (`track_slots` was defined after the
   local `assertRaises` block, but the sweep below used it unconditionally); it is defined
   once, above both uses.
6. **Review 3: warmup capture ran with no producer-side join.**
   `capture_recovery_graphs` filled the stable index/step buffers on the caller stream (after
   a `.synchronize()` of the recovery stream, which only drains the *consumer*), then launched
   its first warmup on the recovery stream with no `record`/`wait_event` pair, and read stash
   entries the verify capture had written on the caller stream. Synchronizing after a launch is
   not protection, and the graph's own later sync is too late, so this was a real ordering
   gap in the port and it is closed here with the same record-on-producer /
   wait-on-consumer pattern the serving path uses, immediately after the buffers are
   initialized and before any launch or capture. The CPU seam test pins the event order; the
   GPU test above exercises it with a delayed producer.
   The same review caught two defects in the committed GPU fixtures, both real but neither
   ever executed here: the mamba-track sweep built its case with `width + 2` rows while its
   slot/step vectors had `width + 1` entries (batch mismatch, and the working slots were
   aliased onto the track slots, so the test compared a row against itself), and the
   padded-rows test ran the reference over the case's identity slots while the candidate ran
   over `[1,-1,2]`, then asserted the reserved discard row equalled the reference's row 0.
   The reference now runs the same mapping, the sweep is row-consistent with disjoint slot
   ranges, and what is asserted is the actual contract: valid rows commit, slots outside the
   mapping keep their bytes. The prefix-reuse replay also compared a pre-commit verify output
   against a post-commit re-read; it now compares the output the replayed step produced.
   None of this was visible from a CPU run, which is the point of the "skip is not pass"
   rule below.

## Verification actually performed (CPU only)

```
pytest -q test/registered/unit/server_args/test_gdn_mtp_cache_mode.py \
          test/registered/unit/mem_cache/test_spec_mamba_buffer_accounting.py \
          test/registered/unit/layers/attention/test_gdn_recovery_stream_order.py
  # 82 passed, 20 subtests — the three new CPU files
pytest -q test/registered/kernels/ops/sampling/test_temperature_softmax.py \
          test/registered/unit/mem_cache/test_mamba_donated_alloc_ratio.py \
          test/registered/unit/mem_cache/test_hicache_staged_write_back_dispatch.py \
          test/registered/unit/mem_cache/test_sharded_dsa_hicache.py
  # 44 passed, 4 skipped, 20 subtests — adjacent arg-surface and HiCache checks
pytest -q test/registered/unit/managers/test_batch_result_processor_mamba_boundary.py \
          test/registered/unit/managers/test_batch_result_processor_spec_grammar.py
  # 14 passed, 13 subtests — mamba boundary / spec post-process neighbours
pytest -q test/registered/unit/server_args/
  # 468 passed, 5 failed — and the same 5 fail on pristine base (444 passed, 5 failed):
  # pre-existing environment/lazy-import cases, none of them mine
pytest -q --collect-only test/registered/attention/unittests/gdn/test_recover_ssm_sm120.py \
                          test/manual/4-gpu-models/test_gdn_mtp_recover_ssm.py
  # 12 collected, all skipped without CUDA
```

Red evidence for the review-3 fix: the two `TestWarmupCaptureJoinsTheCallerStream` tests fail
on the review-2 head (`2 failed, 25 passed` in that file) and pass on this head. Nothing else
changed in that file, so the red is the missing join and not a harness artefact.

Review 1's three constructor regressions were confirmed red on the first port commit
(`17 failed, 69 passed`), and the green run above is the same file set passing. The full
3-commit patch also applies to a pristine export of `28c2d24` with `git am` and passes the
whole set there — new files plus every adjacent file above — `140 passed, 4 skipped, 53
subtests`, with the two GPU files collecting 12 tests, so nothing depends on this checkout's
state.

Repo gates, using the ruff the repo pins in `.pre-commit-config.yaml` (`ruff==0.15.1`; the
`ruff` hook runs `--select=F401,F821,F823,UP037`, the `ruff-format` hook runs
`ruff format --check`):

```
ruff check --no-cache --select=F401,F821,F823,UP037 <all 28 changed .py files>   # clean
ruff format --check <all 28 changed .py files>                                    # clean
```

## Rules for the next edit

* A test that passes when the production code is deleted, renamed or never imported is not a
  regression test. `_recover_ssm_track` and `async_copy_server_request` being absent is
  asserted by attribute absence — and neither existed at base, so the guard earns its keep as
  a regression against reintroduction, not as evidence of a deletion.
* A `skip` is not a `pass`, and an *executed* skip is not a *verified* guarantee. On the host
  that first ran these files, `torch` itself was missing, so everything skipped and the suite
  reported green. On a real SM120 host they execute; review 3 found three of the committed
  GPU cases were broken fixtures that would have failed or passed vacuously there — two
  comparing a row against itself, one comparing pre-commit against post-commit. GPU tests are
  not done until they have run.
* The `.mdx` table block is generated by `docs/developer/make_docs.py`, and `server_args.md`
  and `log_input.txt` are build artefacts (gitignored); the committed table was produced by
  the real generator with `DOC_DATE` pinned, and is byte-identical to what the shipped
  resolvers return.
* `test/registered/server_args/README.md` is a generated manifest (it carries
  `test_mamba_no_cow_buffer_ratio.py`, which does not exist in-tree); it is not a checklist.
* `scripts/playground/` holds generated output and is not an integration target.
* The donor's SM120 4e-2 `atol/rtol` is provisional test history, not a mathematical
  accumulated-error bound.

## Not verified here

GPU numerics and ordering, the SM120 microbenchmark, the Qwen3.5-4B launch and GSM8K run,
speedup, VRAM, memory-report and RSS deltas — all unexecuted. No CI run: the SM120 job needs
a `cpu-1-b200` runner, and the manual test is `registermanual`, not part of the 4-Gpu-Models
job. Nothing in this checkout is dirty: the three existing-check edits
(`test_mamba_donated_alloc_ratio.py`, `test_sharded_dsa_hicache.py`, and the HiCache dispatch
test's `_FakeCache` publish) are committed. The `test_decode_hello.py` file the review saw
staged-but-missing is not present in this checkout at all.
