# Penny Royal 3.1.4

Penny Royal 3.1.4 improves Claude Code and Codex compatibility for tool
histories, structured arguments and image results. Publication and container
verification are pending.

## Fixes

- Anthropic histories with nested `tool_reference` blocks no longer fail with
  HTTP 500 (#137, #138, reported by @BoneGoat). Recognized references render
  safely with names, order and IDs preserved, and the bundled Froggeric chat
  template is unchanged.
- `/v1/responses` accepts namespaced and custom tools, inline tool
  inventories, faithful replay of chronological history, and tool results
  containing images. One consistent identity spans declarations, selections,
  replay and streamed outputs, including retired-name collision handling.
- Qwen tool arguments keep schema-accepted arrays and objects and explicit
  string values such as a literal `"null"`.
- Omitted output-token budgets are computed after image expansion, so picture
  requests without an explicit limit stop failing the context check.

Custom-tool grammar support is prompt guidance, not enforced grammar decoding.
Optional AutoReview was checked and remains optional; your saved approval
settings are unchanged. Features not yet qualified for this release are listed
in [LIMITATIONS.md](LIMITATIONS.md).

## Validation

The final source passed 165 tests with 189 subtests. Both maintained profiles —
27B FP8/DFlash2 and Flash-Next NVFP4/native NEXTN — completed the original
uncoached real Codex MCP workflow with actual tool execution and callbacks,
and a 5/5 ordinary generation/parallel-tool smoke each. Overlapping earlier
runs are recorded separately, not summed, and accepted model-generation
mismatches are detailed in the [release record](release-3.1.4.json).

## Update

When published, pull `ghcr.io/jpezzulli/sglang-rtxpro6000:v3.1.4` and restart
with your existing configuration; update any saved image pin. Native
instructions: [BUILD.md](BUILD.md).

## Thanks

[@BoneGoat](https://github.com/BoneGoat) for the report and reproduction, and
Penny Royal for the fixes. Upstream lineage: [PROVENANCE.md](PROVENANCE.md).
