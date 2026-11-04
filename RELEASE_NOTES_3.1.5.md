# Penny Royal 3.1.5

Penny Royal 3.1.5 reduces temporary GPU memory use when several requests
branch from a cached prefix. The runtime now copies Mamba checkpoint state in
chunks; each chunk targets 64 MiB and always fits at least one full state slot.

On the RTX A4000, copying four checkpoints reduced peak temporary allocation
from 576 to 144 MiB for the 27B state shape and from 216 to 54 MiB for
Flash-Next.

On the RTX PRO 6000, both profiles returned four correct answers and reused
9,344 GPU-cached tokens per request. Flash-Next processed all four in one
prefill batch. The concurrent 27B requests entered prefill separately.

Publication is pending: the image is built and verified at its build-SHA tag,
and the `v3.1.5` tag, its promotion and the release await approval; the
evidence is in [release-3.1.5.json](release-3.1.5.json). Until then the
[container guide](docker/pennyroyal/README.md) installs the published 3.1.4
release.

Thanks to [Tanish Malekar](https://github.com/tanish-malekar) and co-author
Claude Sonnet 5.5 for the upstream fix
([sgl-project/sglang#47984](https://github.com/sgl-project/sglang/pull/47984),
issue #47916), adapted here and merged in
[Penny Royal PR #156](https://github.com/jpezzulli/sglang-rtxpro6000/pull/156)
with the original attribution and sign-off preserved.
