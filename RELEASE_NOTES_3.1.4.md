# Penny Royal 3.1.4

Fixes Claude Code conversations that failed with HTTP 500 when their history
contained nested `tool_reference` blocks
([#137](https://github.com/jpezzulli/sglang-rtxpro6000/issues/137),
[#138](https://github.com/jpezzulli/sglang-rtxpro6000/pull/138)).

Publication and container verification are pending.

## Fix

The Anthropic adapter renders recognized references safely while preserving
tool names, ordering and IDs. Templates that support structured references keep
that handling. The bundled Froggeric chat template is unchanged; ordinary
requests keep identical prompts, and unsupported blocks still return HTTP 400.

## Validation

145 CPU tests and 68 subtests passed, plus 22 live checks across 27B FP8/DFlash2
and Flash-Next NVFP4/native NEXTN. The original reproduction now returns HTTP
200. One unrelated tokenizer-download test was excluded.
[Release record](release-3.1.4.json).

## Update

Once published, pull `ghcr.io/jpezzulli/sglang-rtxpro6000:v3.1.4` and restart
with your existing configuration. Update any saved image pin. Native
instructions: [BUILD.md](BUILD.md).

## Thanks

[@BoneGoat](https://github.com/BoneGoat) for the report and reproduction, and
[Penny Royal](mailto:Pennyroyal@agentmail.to) for the fix.
