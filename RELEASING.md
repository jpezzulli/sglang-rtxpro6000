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
- [ ] Defaults agree across native launchers, container startup files and generated configurations; 3.0 selects FR-Spec plus C1-only adaptive MTP, retaining FR-Spec with fixed W4 at C2+, without a new public optimization toggle.
- [ ] The narrowed GPU build is integrated: RTX PRO 6000 Blackwell inference at TP1/TP2, with Ampere/Ada retained for sidecar image processing. Record actual image identity and size; do not invent a size reduction.
- [ ] Fresh-install and upgrade instructions fetch the correct files, including setup updates made after a runtime release.
- [ ] Version references, image defaults and compatibility hashes are updated.
- [ ] Local diagnostics remain off in the published defaults and are not exposed as a public setup option.

## README and documentation

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
- [ ] Verify the public README, release and installation links.

## Return to everyday use

Do this immediately after validation, before returning the service to normal use.

- [ ] Restore Flash-Next with the agreed settings and local diagnostics enabled.
- [ ] Confirm API readiness and diagnostic initialization; a launcher flag alone is insufficient.
- [ ] Report publication/build status and any remaining work.
