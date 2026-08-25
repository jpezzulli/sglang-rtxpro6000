# Publication hygiene scan

Scan date: 2026-08-24 EDT

The proposed tracked and untracked publication set was scanned before any
remote repository was created. The scan covered 8,218 files from the complete
frozen source plus the new documentation/evidence.

## Results

- No AWS, GitHub, Hugging Face, OpenAI, Slack, or Google credential-shaped
  value was found in the added publication material or seven-commit local
  source delta.
- No operator username, private hostname, personal email address, or HPE
  identifier was found in the proposed working tree.
- HTTP access-log lines containing the private client address were removed from
  the agentic log. No private address remains in added publication files.
- No model-weight extension and no file larger than 100 MB is present outside
  Git object storage.
- No model weights, cache payloads, request bodies, prompt corpus, conversation
  text, hidden reasoning stream, venv, JIT cache, Atlas index, or compiler build
  output is included.

The complete upstream-derived source contains expected security test fixtures
and documentation examples: one test checks for the textual presence of a
private-key header, another uses a synthetic `sk-test-*` API key, and upstream
examples contain private-address placeholders. These are inherited public test
fixtures, not local credentials. The four private-address hits in the local
seven-commit diff are synthetic request-statistics unit-test inputs.

## Deliberate exclusions

- Model weights and Hugging Face cache payloads: replaced by immutable revisions
  and SHA-256 identities in `evidence/models/checkpoints.json`.
- HiCache/NIXL FILE data and all other cache contents: disposable runtime state,
  very large, and potentially derived from private prompts.
- `/opt/sglang/.venv`, built wheels except their hashes, JIT caches, Rust target
  output, Atlas indexes, and other build products: reconstructed by `BUILD.md`.
- Full local validation request/SSE artifacts: may contain hidden reasoning or
  test prompts; compact numeric summaries and the public validation repository
  revision are supplied instead.
- The original machine-bound launcher and systemd unit: the performance-
  relevant shape is preserved in a generalized launcher without local service
  wiring.
- NIXL source: it is a separate upstream project; its exact commit and build
  options are recorded rather than vendored.
- The earlier 199-line agentic excerpt: it is contained within the larger
  1,181-line sample, so publishing both would duplicate requests.
