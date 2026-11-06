# Penny Royal 3.1.6

3.1.6 speeds up prefill after long cached prefixes, adds an optional shared image
preprocessing cache, and fixes partial hybrid-cache prefetch handling.

On Flash-Next with 200,000 cached tokens and 4,000 new input tokens, fresh-input
prefill improved by about 26% for one request and 33% for four concurrent requests
versus 3.1.5. See [RESULTS.md](RESULTS.md) for the measured configurations and both
profiles' results.

The optional shared image preprocessing cache lets compatible tokenizer workers reuse
processed image artifacts. Enable it with `PENNY_MM_PREPROCESS_CACHE_DIR` and a
positive `PENNY_MM_PREPROCESS_CACHE_SIZE_MB` budget in MiB; both settings are available
in advanced setup. It is off by default. Use a fresh directory when changing its
budget.

Hybrid cache hits that cannot be restored completely now fall back to recomputation.

27B DFlash2 remains qualified; Flash-Next DFlash2 remains alpha. Both retain their
extra draft host-RAM requirement.

Thanks to [Jim Smith](https://github.com/jhsmith409) for identifying and benchmarking
the QSA optimisation and sharing his port; to upstream authors Shunkangz
([sgl-project/sglang#45791](https://github.com/sgl-project/sglang/pull/45791)) and
Zhangheng ([#48421](https://github.com/sgl-project/sglang/pull/48421)); and to
[lihong](https://github.com/leehom0123) for the shared image cache in
[PR #155](https://github.com/jpezzulli/sglang-rtxpro6000/pull/155).

The container is built and verified. Publication is pending;
[pennyroyal-v3.1.5](RELEASE_NOTES_3.1.5.md) remains the published download.
