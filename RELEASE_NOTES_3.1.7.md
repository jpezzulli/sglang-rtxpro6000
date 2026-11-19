# Penny Royal 3.1.7

Claude Code can send `thinking.display: "updates"` to `/v1/messages` with
adaptive or enabled thinking ([issue #171](https://github.com/jpezzulli/sglang-rtxpro6000/issues/171)).
Penny continues to stream ordinary reasoning.
Thanks to JustinTong0323 for the
[upstream fix](https://github.com/sgl-project/sglang/pull/45129); Penny wrote the
backport and the launcher change.

Flash-Next launchers now default CUDA expandable segments off when `TP_SIZE`
is greater than one. This addresses the reported CUDA IPC failure during
graph capture. See [configuration](CONFIGURATION.md#gpus-tp-and-media) for
the allocator override and [issue #161](https://github.com/jpezzulli/sglang-rtxpro6000/issues/161)
for the field report.

TP2 validation is pending. If you have two RTX PRO 6000s, please report
whether 3.1.7 starts, captures its CUDA graphs, and serves a request with
`TP_SIZE=2`. Include your launch settings and any startup errors in #161.

Use source tag `pennyroyal-v3.1.7` and image
`ghcr.io/jpezzulli/sglang-rtxpro6000:v3.1.7` once [release-3.1.7.json](release-3.1.7.json)
records the published pair; until then `pennyroyal-v3.1.6` with `v3.1.6` is what is
published, and passing `--image` keeps these launch files on it. Update existing
image pins and launch files together; see the [container guide](docker/pennyroyal/README.md).
