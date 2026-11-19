# Penny Royal 3.1.7

Claude Code can send `thinking.display: "updates"` to `/v1/messages` with
adaptive or enabled thinking. Penny continues to stream ordinary reasoning.
Thanks to JustinTong0323 for the
[upstream fix](https://github.com/sgl-project/sglang/pull/45129).

Flash-Next launchers now default CUDA expandable segments off when `TP_SIZE`
is greater than one. This addresses the reported CUDA IPC failure during
graph capture. See [configuration](CONFIGURATION.md#gpus-tp-and-media) for
the allocator override and [issue #161](https://github.com/jpezzulli/sglang-rtxpro6000/issues/161)
for the field report.

TP2 validation is pending. If you have two RTX PRO 6000s, please report
whether 3.1.7 starts, captures its CUDA graphs, and serves a request with
`TP_SIZE=2`. Include your launch settings and any startup errors in #161.
