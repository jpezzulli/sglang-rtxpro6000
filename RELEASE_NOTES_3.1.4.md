# Penny Royal 3.1.4

One fix: Claude Code histories with nested `tool_reference` blocks no longer
return HTTP 500. **Publication is pending** — the `pennyroyal-v3.1.4` tag and
the `v3.1.4` image do not exist yet; pending fields are marked in
[release-3.1.4.json](release-3.1.4.json). Until then,
[3.1.3](RELEASE_NOTES_3.1.3.md) remains the published release.

## Fix

- **Nested tool references are rendered safely (#137).** Such a history failed
  with HTTP 500 against the bundled Froggeric v22.5 template, which has no
  native reference expansion. The Anthropic adapter now renders recognized
  references as text — preserving names, order and IDs — and keeps native
  structured handling on capable templates. The reported reproduction returns
  HTTP 200, ordinary prompts send the same bytes, and unsupported blocks still
  return a clean HTTP 400. The chat template is unchanged. No setting, default
  or sampling behavior moves: the public width policy stays `tiered`, 27B
  DFlash2 remains qualified and only Flash-Next DFlash2 stays alpha, and the
  draft's host-RAM note stands — our 27B launch measured **24.83 GB of draft
  cache** beyond the 96 GB target/Mamba budget, with no allocator change.

## Validation (reused, not rerun)

Accepted head `7a5e449c`, tested on the published 3.1.3 base `177697b3`:
three sequential reviews, final clean; 145 CPU tests with 68 subtests passed
(one unrelated uncached external-tokenizer test deselected); Contributor CI
run 40745879894 green; 22 live checks passed across both 27B FP8/DFlash2 and
Flash-Next NVFP4/native NEXTN — the original 500→200, streaming, mixed
references, token counts, expected 400s, generation and parallel tools — with
no Xid in the window. The 3.1.3 candidate `ef3258749b` was restored afterwards
and passed 5/5 restoration checks. Image verification and the host-native
banner check are open publication gates.

## Update

When published, container users pull `ghcr.io/jpezzulli/sglang-rtxpro6000:v3.1.4`
and restart with their existing configuration; native users follow
[BUILD.md](BUILD.md). An explicit image pin survives on purpose — update it
yourself. Until then the published download pair stays `pennyroyal-v3.1.3`
source with the `v3.1.3` image ([container guide](docker/pennyroyal/README.md)).

## Thanks

- [@BoneGoat](https://github.com/BoneGoat) — the report and reproduction.
- [Penny Royal](mailto:Pennyroyal@agentmail.to) — the fix and release docs;
  the commit's original attribution is preserved.
