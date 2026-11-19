# Penny Royal 3.1.7

Claude Code can send `thinking.display: "updates"` to `/v1/messages` with
adaptive or enabled thinking ([issue #171](https://github.com/jpezzulli/sglang-rtxpro6000/issues/171)).
Penny continues to stream ordinary reasoning.
Thanks to thushan for the reproduction and JustinTong0323 for the
[upstream fix](https://github.com/sgl-project/sglang/pull/45129). Penny wrote the backport.

Flash-Next launchers now default CUDA expandable segments off when `TP_SIZE`
is greater than one. Penny made this change after paulorsjr traced a CUDA IPC
failure during graph capture to that setting on two RTX PRO 6000s.
See the [allocator override](CONFIGURATION.md#gpus-tp-and-media) and
[field report](https://github.com/jpezzulli/sglang-rtxpro6000/issues/161).

TP2 validation is pending. If you have two RTX PRO 6000s, please report
whether 3.1.7 starts, captures its CUDA graphs, and serves a request with
`TP_SIZE=2`. Include your launch settings and any startup errors in #161.

For upgrades, use the launch files and image from the same release. [Container guide](docker/pennyroyal/README.md)
