# Limitations

- Flash-Next integration was produced immediately after model availability. It
  has extensive evidence on one RTX PRO 6000, but it is still early and may not
  be cleanly portable to another GPU, topology, checkpoint, or dependency set.
- An upstream architecture gate can mean “not selected or wired on SM120,” not
  “the kernel is incompatible.” This runtime opens only paths directly
  qualified on SM120; it does not globally remove architecture checks.
- Ordinary FlashInfer state-writing GDN target verification remains disabled on
  SM120. Only the `none`-mode WY output/recovery path is enabled. Tree/full-state
  verification remains Triton.
- FlashInfer HyperConnection Mix remains SM100-only; SM120 uses the qualified
  persistent Triton implementation. Shared-expert fusion is not claimed active.
- The Flash-Next QSA sparse-prefill correction covers the active unit-scale FP8
  checkpoint. The broader calibrated-scale path in open PR #36644 is not part
  of this release.
- SGLang's KV sizing estimator still reserves the intermediate GDN speculative
  pool even though `gdn_mtp_cache_mode=none` removes its physical allocation.
  The qualified `.981` memory fraction exposes the recovered budget. This is a
  documented estimator debt, not a general recommendation to raise memory
  fractions blindly.
- NIXL cleaner thresholds are whole-filesystem occupancy percentages, not an
  absolute cache-directory byte limit. Other data on the same filesystem can
  shift the effective cache capacity. Use a dedicated filesystem when strict
  occupancy semantics matter.
- Persistent cache is disposable. Representation namespaces reduce accidental
  reuse, but incomplete or missing state is allowed to become a later cache
  miss and recomputation.
- The final Flash-Next suite ran on source `64ecd64924`. The 27B/DFlash2 numbers
  are from the earlier qualified runtime-line source retained in this history;
  the later Flash-Next-specific commits were not all re-benchmarked on 27B.
- Results are measured observations, not performance guarantees. Flash-Next and
  27B results use different models, speculative mechanisms, state layouts, and
  memory allocations and must not be compared as if they were one test arm.
