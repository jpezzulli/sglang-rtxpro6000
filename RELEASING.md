# Penny Royal release checklist

Use this checklist on the release's existing GitHub Project item. Mark each
item complete with a short result or link. Identify reused checks by source
and date; discuss exceptions rather than silently skipping them. This is a
maintainer checklist, separate from the public release announcement.

An explicitly accepted reuse or exception satisfies the relevant item when
its scope and date are recorded. This checklist does not reopen completed
qualification or reverse John's decision to reuse the 27B checks for 3.0.

## Validation

- [ ] 27B: full reasoning, tool, long-context and media suites completed.
- [ ] 27B: prefill/decode speed and time to first token recorded.
- [ ] 27B: cache restore checked, including correct answers and actual cache reuse.
- [ ] Flash-Next: full reasoning, tool, long-context and media suites completed.
- [ ] Flash-Next: prefill/decode speed and time to first token recorded.
- [ ] Flash-Next: cache restore checked, including correct answers and actual cache reuse.
- [ ] New fixes have focused regression checks and completed review.
- [ ] Results identify the tested source, model and settings. Reused results remain dated; failures are resolved or explicitly discussed.

## Setup and packaging

- [ ] Beta configurator updated for every accepted user-facing change, including defaults, supported combinations and generated startup settings. Record the checked revision on the release item.
- [ ] Every new user setting appears where needed: native instructions, host container startup files, environment examples and configurator.
- [ ] Configurator saves, reloads and passes settings to the launched process.
- [ ] Defaults agree across native launchers, container startup files and generated configurations; 3.1 selects the native-MTP Flash-Next profile with FR-Spec by default, keeps the optional DFlash2 profile's fixed width routing (block 6 at one request, block 4 at two or more) with FlashInfer verification of accepted draft tokens, FR-Spec retained, automatic KV pool sizing, and the native-MTP recipes maintained without a new public optimization toggle.
- [ ] The narrowed GPU build is integrated: RTX PRO 6000 Blackwell inference at TP1/TP2, with Ampere/Ada retained for sidecar image processing. Record actual image identity and size; do not invent a size reduction.
- [ ] Fresh-install and upgrade instructions fetch the correct files, including setup updates made after a runtime release.
- [ ] Version references, image defaults and compatibility hashes are updated.
- [ ] Update the user-visible Penny Royal version for the release. Before publication, verify its actual output in both the native release environment and built candidate container; record the commands or endpoints and output on the release item. Identify any upstream SGLang package version separately.
- [ ] Capture the native dependency baseline before the expensive container build: from the actual qualified native release environment, record the interpreter path, source and date, the Python version, and every installed package version, direct and transitive, with exact build identifiers and immutable source identity for locally built packages. A post-build `pip freeze` alone does not satisfy this gate.
- [ ] Pin every shared runtime dependency to its qualified baseline version so the image build installs that version instead of freshly selecting from broad requirements. Deliberate upgrades are accepted and validated natively first, then refresh the baseline and pins before the release build.
- [ ] Identify native-only development tools and container-only build or platform packages separately, with exact versions for container additions, and do not install native-only development tools into the image. Separating them is not permission to silently waive runtime parity.
- [ ] Local diagnostics remain off in the published defaults and are not exposed as a public setup option.

## README and documentation

- [ ] Configuration reference completed after release scope and defaults are agreed: all supported settings and overrides, defaults and recommendations for each profile, where to set them, interactions, and restart/cache effects (see [CONFIGURATION.md](CONFIGURATION.md)). Record the checked revision on the release item.
- [ ] README and affected documentation updated for the final included features, settings and measured results. Record the checked revision on the release item.
- [ ] Human-readable wording; no defensive qualification boilerplate.
- [ ] Search terms, descriptive headings and canonical links preserved.
- [ ] Current results shown for both models; older results linked as history.
- [ ] `llms.txt` and machine-readable release notes updated.
- [ ] Penny Royal name, approved logo assets and light/dark presentation are consistent.
- [ ] Native SGLang mean/peak measurements follow `RESULTS.md`; client wall-time rates stay separately labelled.
- [ ] README, setup-guide links and section anchors checked.
- [ ] Release notes follow the agreed format: problem, technical fix, user effect, credits and update instructions.
- [ ] Named platform fixes, such as WSL2, are explicitly mentioned.

## Board, issues and pull requests

- [ ] Board matches what shipped, what remains and what was deferred.
- [ ] Review related issues for replies or closure, and carry out the agreed actions.
- [ ] Every release change has an associated PR. Before submission, each maintainer-submitted PR covers one functional item with its related tests and documentation; record any explicitly approved exception on the release item.
- [ ] Every included PR has a disposition: merged, incorporated with changes, or deferred.
- [ ] Accepted contributor work goes through its original PR wherever possible. Necessary corrections are made there before merge.
- [ ] Release acknowledgements name direct Penny Royal contributors and community interaction; source notices and upstream lineage stay in their existing records.
- [ ] Contributor authorship and credit are preserved. Any adaptation has an agreed explanation and disposition for the original PR.
- [ ] Issue replies link to the actual fix or release.

## Publication

When the container build or packaging changes, build the candidate image and
boot it for a quick generation/tool check before publishing the release. Reuse
the completed runtime qualification, and promote the checked image digest
without rebuilding it.

- [ ] README/documentation and configurator gates above are both complete for this candidate; a runtime test pass does not complete these gates.
- [ ] John approves the final release notes and publication.
- [ ] Publish the reviewed source and immutable release tag; never move an already published tag.
- [ ] Publish release notes and start the container build.
- [ ] Confirm build checks pass and the versioned image points to the correct source. Until checked, its status is pending.
- [ ] Before publication, compare the candidate image's actual installed versions against the approved native baseline and pins. Missing, extra or mismatched runtime dependencies block release; only explained development/build/platform-only differences are allowed. Record the baseline identity, pin-file revision and parity result on the release item.
- [ ] Verify the public README, release and installation links.

## Return to everyday use

Do this immediately after validation, before returning the service to normal use.

- [ ] Bring the normal native checkout to the published main branch and install from it. Use stable environment and launcher paths; no live dependency on candidate or investigation directories. Keep private diagnostics explicitly separate from published defaults.
- [ ] Restore Flash-Next with the agreed settings and local diagnostics left off, the operator's current decision; a private diagnostic overlay stays separate and is not restored.
- [ ] Confirm API readiness and the diagnostic state the operator selected, off for this release; a launcher flag alone is insufficient.
- [ ] Retire completed local release worktrees, candidate environments and temporary build artifacts after checking live references and uncommitted work. Preserve ongoing investigations and deliberate qualification results; name any cleanup still pending.
- [ ] Report publication/build status and any remaining work.
