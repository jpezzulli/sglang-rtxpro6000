# Penny Royal 3.1.4

Penny Royal 3.1.4 is accepted and awaits publication. It makes Claude Code
histories with nested `tool_reference` blocks safe (#137, #138, reported by
@BoneGoat) and adds the accepted Codex-style `/v1/responses` work on top of
merged #138 and #140: namespaced function and custom tools with inline
inventories, chronological replay and history, and image-containing tool
results, with one reversible tool identity shared by the prompt, selections,
replay and full/SSE outputs. Colliding names that leave the live inventory
keep their retired identity and stay non-callable. Qwen tool arguments now
keep schema-accepted arrays and objects and explicit string values such as a
literal `"null"`, and omitted completion-token budgets are deferred until
after image expansion, so picture requests stop failing the context check.
Optional AutoReview and the narrow plaintext agent-message handling behave as
before. The bundled Froggeric template, defaults, settings and sampling are
unchanged, and this is not full Codex parity.

Live evidence: both 27B FP8/DFlash2 and Flash-Next NVFP4/native NEXTN passed
an uncached real Codex MCP workflow with actual tool execution and a 5/5
generation/parallel-tool smoke; the final accepted source passed 165 tests
with 189 subtests. Flash-Next parser spot-checks passed as accepted, with
known model-generation mismatches listed in the
[release record](release-3.1.4.json) — not every payload is exact.

Publication is held. The earlier candidate build 40757589184 is cancelled and
obsolete; no new image exists, so the [3.1.3](RELEASE_NOTES_3.1.3.md) image is
still the current published release and the manual install examples stay on
it. When `v3.1.4` publishes, pull the new tag and restart with your existing
configuration; native instructions are in [BUILD.md](BUILD.md).

Thanks to @BoneGoat for the report and reproduction, to Penny Royal for the
fixes, and to the upstream authors whose accepted changes were adapted —
lineage is in [PROVENANCE.md](PROVENANCE.md).
