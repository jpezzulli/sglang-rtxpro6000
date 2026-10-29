# Penny Royal 3.1.4

Penny Royal 3.1.4 improves Claude Code and Codex compatibility for tool
histories, structured arguments and image results, and makes container
dependency versions predictable. Publication and container verification are
pending; 3.1.3 remains published.

## Fixes

- Anthropic histories with nested `tool_reference` blocks no longer fail with
  HTTP 500 (#137, #138, reported by @BoneGoat). Recognized references render
  safely with names, order and IDs preserved, and the bundled Froggeric chat
  template is unchanged.
- `/v1/responses` accepts namespaced and custom tools, inline tool inventories,
  faithful replay of chronological history and image-containing tool results,
  with one consistent identity across declarations, selections, replay and
  streamed outputs. Tool namespaces survive request validation on every
  supported dependency version (#146).
- Qwen tool arguments keep schema-accepted arrays and objects and explicit
  string values such as a literal `"null"`.
- Omitted output-token budgets are computed after image expansion, so picture
  requests without an explicit limit stop failing the context check.

## Packaging

Images pin the dependency versions validated on the host and reuse the compiled
FlashInfer artifact only when its recorded inputs match (#147, #148), so image
and host stay in step. The pin set is still landing, and publication waits for
it.

Custom-tool grammar support is prompt guidance, not enforced grammar decoding.
Optional AutoReview was checked and remains optional, and your saved approval
settings are unchanged. Features not yet qualified are listed in
[LIMITATIONS.md](LIMITATIONS.md).

## Validation

The final source passed 165 tests with 189 subtests, and both maintained profiles
(27B FP8/DFlash2, Flash-Next NVFP4/native NEXTN) completed the original
uncoached real Codex MCP workflow with actual tool execution and callbacks and a
5/5 ordinary generation/parallel-tool smoke each. Overlapping earlier runs are
recorded separately, not summed; accepted model-generation mismatches are in the
[release record](release-3.1.4.json).

## Update

When published, pull `ghcr.io/jpezzulli/sglang-rtxpro6000:v3.1.4` and restart
with your existing configuration; update any saved image pin. Native
instructions: [BUILD.md](BUILD.md).

## Thanks

[@BoneGoat](https://github.com/BoneGoat) for the report and reproduction, and
Penny Royal for the fixes. Upstream lineage: [PROVENANCE.md](PROVENANCE.md).
