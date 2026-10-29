# Penny Royal 3.1.4

Penny Royal 3.1.4 improves Claude Code and Codex compatibility for tool
histories, structured arguments and image results, and makes container dependency
versions predictable. Publication is pending; 3.1.3 remains published.

## Fixes

- Anthropic histories with nested `tool_reference` blocks no longer fail with
  HTTP 500 (#137, #138, reported by @BoneGoat). Recognized references render
  safely with names, order and IDs preserved, and the bundled Froggeric template
  is unchanged.
- `/v1/responses` accepts namespaced and custom tools, inline tool inventories,
  replay of chronological history and image-containing tool results, with one
  identity across declarations, selections, replay and outputs. Tool namespaces
  survive validation on the tested pydantic 2.13.4 and 2.14.1 pairs (#146).
- Qwen tool arguments keep schema-accepted arrays and objects and explicit
  strings such as a literal `"null"`.
- Omitted output-token budgets are computed after image expansion, so image
  requests without an explicit limit stop failing the context check.

## Packaging

Images install the exact dependency versions qualified on the native host, gated
around the FlashInfer compile and again in the final image (#150); compiled
artifacts restore only when their recorded inputs match (#147, #148). Custom-tool
grammar is prompt guidance, not enforced decoding; AutoReview stays optional with
your settings unchanged; features not yet qualified are in
[LIMITATIONS.md](LIMITATIONS.md).

## Validation

The namespace correction passed 384 tests with 303 subtests on each tested
dependency pair, and Flash-Next passed live API checks plus the real uncoached
Codex MCP workflow, replay and image callback. Earlier two-profile evidence (165
tests with 189 subtests, plus the 27B and Flash-Next runs) is reused as dated
results; the [release record](release-3.1.4.json) carries the build history.

## Update

When published, pull `ghcr.io/jpezzulli/sglang-rtxpro6000:v3.1.4` and restart
with your existing configuration; update any saved image pin. Native setup:
[BUILD.md](BUILD.md).

## Thanks

[@BoneGoat](https://github.com/BoneGoat) for the report and reproduction, and
Penny Royal for the fixes. Upstream lineage: [PROVENANCE.md](PROVENANCE.md).
