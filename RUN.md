# Run

The measured service used one RTX PRO 6000 Blackwell 96 GB at a 450 W board
limit, tensor parallelism 1, and an OpenAI-compatible endpoint. The generalized
launcher is [configs/pennyroyal/serve-qwen38-dflash2.sh](configs/pennyroyal/serve-qwen38-dflash2.sh).

## Runtime shape

| Area | Qualified value |
|---|---|
| Target | `orcarouter/Qwen3.8-27B-Uncensored-FP8` |
| Draft | `incoai/Qwen3.8-27B-DFlash2` |
| GPU / topology | one RTX PRO 6000 Blackwell 96 GB; SM120; TP=1 |
| Compute | BF16 |
| Target / draft KV | FP8 E4M3 / FP8 E4M3 |
| Context admission | 524,288 tokens |
| RoPE | YaRN factor 2 from native 262,144; separate target and draft overrides |
| Concurrent requests | 4 maximum |
| GPU memory fraction | 0.94 |
| Full-KV token pool | 1,194,496 tokens at startup |
| KV page size | 64 tokens |
| Prefill chunk / maximum | 2,048 / 2,048 tokens |
| Target prefill | FlashInfer |
| Target decode / verify | TRTLLM-MHA with XQA on SM120 |
| Draft attention | FlashInfer |
| Speculation | DFlash2, 8 draft tokens, 2,048-token draft window, decode mode |
| CUDA graphs | active during qualified decode and agentic samples |
| Mamba GPU pool | 16 slots |
| Mamba retention | `extra_buffer_lazy`; maximum 3 states per radix path |
| Mamba dtypes | FP32 SSM state; BF16 convolution state |
| Thinking default | enabled and preserved; `reasoning_effort=medium` |
| Reasoning / tools | Qwen3 reasoning parser; Qwen3 Coder tool parser |
| Idle behavior | `--sleep-on-idle` |

The Mamba three-state path cap is a correctness setting for retained-prefix
copy-on-write behavior, not a transient benchmark tweak. Runs without the cap
exhausted all 16 slots and failed the tested transition.

## HiCache and NIXL

HiCache/NIXL is part of the deployment but is not the source of GPU decode
speed. It provides reusable-prefix movement and persistence:

```text
GPU target KV + Mamba/GDN state + DFlash2 state
    -> HiCache host RAM
    -> NIXL POSIX FILE storage on NVMe/XFS
```

| Setting | Value |
|---|---|
| HiCache configured size | 96 decimal GB |
| Host mode | cache |
| Write policy | write-through |
| GPU/host I/O | kernel |
| Host layout | page-first |
| Storage backend | NIXL |
| Prefetch policy | timeout |
| FILE path | POSIX + io_uring + O_DIRECT |
| Cleaner watermarks | 68% high / 65% low whole-filesystem occupancy |

The 96 GB setting is not a total hybrid-state ceiling. Startup allocated
78.42 GB target KV, 17.70 GB Mamba/GDN state, and 24.51 GB DFlash2 state in
host memory: 120.63 GB total. The qualified host had roughly 224 GiB of RAM.

Persistent roots are representation-specific. The included namespace helper
hashes checkpoint content identity, source revision and tracked diff, context,
topology, page geometry, dtypes, speculative shape, attention choices, Mamba
dtypes, Torch version, and SM architecture. An identical configuration selects
the same directory after restart; a representation change selects another.

Mooncake is not a fallback. It was rejected and removed from the deployed path.
The fallback is this same target/DFlash2 runtime with hierarchical RAM and SSD
caching disabled.

## Prepare directories

The example assumes the repository is cloned at `/opt/sglang/src` and its venv
is `/opt/sglang/.venv`. Adjust the parent paths through environment variables if
needed.

```bash
mkdir -p /srv/cache/sglang-runtime/{huggingface,torch,torchinductor,triton,cuda,flashinfer,sglang/jit}
mkdir -p /srv/cache/sglang_nixl
```

Choose cleaner watermarks from whole-filesystem occupancy on the selected SSD.
The included 68/65 values produced an approximate 455-515 GB NIXL operating
band on the qualified 1.8 TiB `/srv` filesystem; they are not portable byte
limits.

## Launch

```bash
export SGLANG_ROOT=/opt/sglang
export TARGET_MODEL=/path/to/orcarouter-Qwen3.8-27B-Uncensored-FP8
export DRAFT_MODEL=/path/to/incoai-Qwen3.8-27B-DFlash2
export CACHE_BASE=/srv/cache/sglang-runtime
export NIXL_STORAGE_BASE=/srv/cache/sglang_nixl

/opt/sglang/src/configs/pennyroyal/serve-qwen38-dflash2.sh
```

For a like-for-like controlled xhigh decode, override reasoning effort in the
request. The 1x/4x qualification used temperature 1.0, top-p 0.95, a fixed
seed per stream, thinking enabled, and a 1,024-token output ceiling.

## Startup checks

Before measuring, confirm all of the following in logs and API behavior:

- target and draft checkpoints match the recorded revisions;
- target and draft KV caches are both FP8 E4M3;
- DFlash2 initialized with eight draft tokens and a 2,048-token window;
- target decode/verification selected TRTLLM-MHA/XQA;
- draft attention selected FlashInfer;
- Mamba allocation is 16 with the three-state path cap;
- full-KV capacity is 1,194,496 tokens;
- CUDA graph capture completed and decode reports `cuda graph: True`;
- the selected NIXL FILE namespace matches the printed representation identity;
- ordinary thinking and tool-call smoke requests succeed.

Perform several thinking-enabled warmups before collecting throughput. Do not
stop a client with signals to cancel an inference request; use protocol-level
cancellation or let it finish.
