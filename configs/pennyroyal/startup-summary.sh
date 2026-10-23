#!/usr/bin/env bash
# Product identity: what release is starting and which source it came from.
# Every launch path calls this before its first preflight step, so a launch that
# dies later still says what it was. The release comes from the version asset
# beside this file, never from a Git tag (a shallow SHA-based container checkout
# has none) and never from the SGLang package version, whose 0.5.x number is
# upstream's own. The revision is the checkout's HEAD, and locally modified
# source is named as such instead of claiming a clean release.
pennyroyal_version_banner() {
  local root="${REPO_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}"
  local version revision state asset
  # The asset ships beside this file, so a REPO_ROOT that points elsewhere (or
  # at nothing) still reports the release the launcher itself came with.
  asset="$root/configs/pennyroyal/pennyroyal.version"
  [[ -r "$asset" ]] || asset="$(dirname -- "${BASH_SOURCE[0]}")/pennyroyal.version"
  version="$(sed -n '1p' "$asset" 2>/dev/null || true)"
  [[ -n "$version" ]] || version='unknown'
  revision="$(git -C "$root" rev-parse --short=10 HEAD 2>/dev/null || true)"
  if [[ -z "$revision" ]]; then
    state='source revision unavailable'
  else
    state="source $revision"
    if [[ -n "$(git -C "$root" status --porcelain --untracked-files=no 2>/dev/null || true)" ]]; then
      state="$state, locally modified"
    fi
  fi
  printf 'Penny Royal %s | %s\n' "$version" "$state"
}

# Display the arguments we will actually exec, not a second set of defaults.
# Allocation/admission can still be clamped during SGLang initialization.
pennyroyal_startup_summary() {
  local -A setting=()
  local arg key value profile frspec=off ple='not applicable' online='not applicable'
  local media="${SGLANG_MM_PREPROCESS_DEVICE:-automatic}" media_label
  while (( $# )); do
    arg="$1"
    case "$arg" in
      --*=*) key="${arg%%=*}"; value="${arg#*=}" ;;
      --model-path|--speculative-draft-model-path|--speculative-algorithm|--speculative-token-map|--context-length|--max-total-tokens|--max-running-requests|--max-mamba-cache-size|--kv-cache-dtype|--speculative-draft-kv-cache-dtype|--hicache-size|--hicache-storage-backend|--image-processor-backend|--tp|--speculative-num-steps|--speculative-eagle-topk|--speculative-num-draft-tokens|--speculative-adaptive-config)
        key="$arg"; value="${2:-unspecified}"; if (( $# > 1 )); then shift; fi ;;
      --ple-offload-embedding|--enable-hierarchical-cache|--speculative-adaptive) key="$arg"; value=true ;;
      *) shift; continue ;;
    esac
    setting["$key"]="$value"
    shift
  done
  case "${setting[--speculative-algorithm]:-none}" in
    NEXTN)
      # Native MTP is the maintained default launch path; the summary names
      # it the same way the recipes do.
      profile='Qwen3.8 Flash-Next / native NEXTN MTP'
      ;;
    DFLASH)
      # Two DFlash2 launches run this same algorithm against different targets,
      # and the path alone does not say which is which, so the recipe names
      # itself. The default keeps the 27B report for the recipe that predates
      # the label.
      profile="${PENNYROYAL_PROFILE_LABEL:-Qwen3.8-27B / DFlash2 (DFLASH, no FR-Spec)}"
      ;;
    *) profile="${setting[--speculative-algorithm]:-no speculation}" ;;
  esac
  [[ -z ${setting[--speculative-token-map]:-} ]] || frspec=on
  # Resident FP8 and PLE placement belong to the launch, not to one speculation
  # algorithm: the Flash-Next DFlash2 path runs the same rowwise-FP8 candidate
  # head and the same PLE offload as the NEXTN paths, so both are reported from
  # what this launch resolved or asked for rather than from its algorithm.
  # Report the effective behavior, not a manufactured default: a saved explicit
  # choice is named as such, otherwise the eligibility the recipes resolved
  # before namespace derivation decides.
  case "${SGLANG_SM120_ONLINE_MXFP8:-}" in
    true) online='rowwise FP8 (explicit)' ;;
    false) online='off (explicit opt-out)' ;;
    *)
      if [[ "${ONLINE_FP8_PRECISION:-false}" == rowwise_fp8 ]]; then
        online='rowwise FP8 (automatic)'
      else
        online='off (automatic: eligibility not met)'
      fi
      ;;
  esac
  if [[ ${setting[--ple-offload-embedding]:-false} == true ]]; then
    ple='host RAM'
  elif [[ ${PENNY_PLE_BACKEND:-} == nvme ]]; then
    ple='NVMe (prepared model overlay)'
  else
    ple='no host offload requested'
  fi
  case "$media" in
    cpu) media_label=CPU ;;
    cuda:0) media_label='main GPU (cuda:0)' ;;
    cuda:*) media_label="secondary GPU ($media)" ;;
    *) media_label="$media" ;;
  esac
  printf '\nPennyroyal startup — requested settings\n'
  printf '  Profile: %s | TP: %s\n' "$profile" "${setting[--tp]:-automatic}"
  printf '  Model: %s\n' "${setting[--model-path]:-unspecified}"
  if [[ -n ${setting[--speculative-draft-model-path]:-} ]]; then
    printf '  Draft model: %s\n' "${setting[--speculative-draft-model-path]}"
  fi
  printf '  FR-Spec: %s | Online FP8: %s | KV dtype: %s\n' \
    "$frspec" "$online" "${setting[--kv-cache-dtype]:-automatic}"
  if [[ -n ${setting[--speculative-draft-kv-cache-dtype]:-} ]]; then
    printf '  Draft KV dtype: %s\n' "${setting[--speculative-draft-kv-cache-dtype]}"
  fi
  if [[ -n ${setting[--speculative-token-map]:-} ]]; then
    printf '  FR-Spec token map: %s\n' "${setting[--speculative-token-map]}"
  fi
  if [[ ${setting[--speculative-adaptive]:-false} == true ]]; then
    case "${setting[--speculative-algorithm]:-}" in
      DFLASH)
        # DFLASH adaptation is fixed batch-size width routing, not the
        # acceptance-driven MTP loop: the launch line carries the widest routed
        # width and the policy file picks the width per batch size.
        printf '  Speculation: DFLASH fixed-width routing (launch width steps=%s | draft tokens=%s; policy %s)\n' \
          "${setting[--speculative-num-steps]:-not set}" \
          "${setting[--speculative-num-draft-tokens]:-not set}" \
          "${setting[--speculative-adaptive-config]:-built-in table}"
        ;;
      *)
        # The launch line names the policy maximum; the runtime picks the width
        # per batch from the adaptive config file, so do not report a static one.
        # The chosen width policy is named because one setting now selects this
        # whole line: the file and the width travel together.
        printf '  Speculation: adaptive MTP up to steps=%s | top-k=%s | draft tokens=%s (policy %s)\n' \
          "${setting[--speculative-num-steps]:-not set}" \
          "${setting[--speculative-eagle-topk]:-not set}" \
          "${setting[--speculative-num-draft-tokens]:-not set}" \
          "${setting[--speculative-adaptive-config]:-built-in table}"
        if [[ -n ${PENNY_MTP_POLICY:-} ]]; then
          printf '  MTP width policy: %s (PENNY_MTP_POLICY)\n' "$PENNY_MTP_POLICY"
        fi
        ;;
    esac
  else
    printf '  Speculation: steps=%s | top-k=%s | draft tokens=%s\n' \
      "${setting[--speculative-num-steps]:-not set}" \
      "${setting[--speculative-eagle-topk]:-not set}" \
      "${setting[--speculative-num-draft-tokens]:-not set}"
  fi
  printf '  Context: %s tokens | KV cap: %s\n' \
    "${setting[--context-length]:-automatic}" "${setting[--max-total-tokens]:-automatic}"
  printf '  Max running requests: %s | Mamba slots: %s\n' \
    "${setting[--max-running-requests]:-automatic}" "${setting[--max-mamba-cache-size]:-automatic}"
  # SGLang sizes the host pool at size * 1e9 bytes (decimal GB, not GiB).
  printf '  PLE: %s | HiCache: %s | Host tier: %s GB\n' \
    "$ple" "${setting[--enable-hierarchical-cache]:-false}" "${setting[--hicache-size]:-automatic}"
  printf '  Storage backend: %s | NIXL location: %s\n' \
    "${setting[--hicache-storage-backend]:-none}" "${SGLANG_HICACHE_NIXL_BACKEND_STORAGE_DIR:-not set}"
  printf '  Media preprocessing: %s (%s); vision encoder stays on model GPU\n' \
    "$media_label" "${setting[--image-processor-backend]:-automatic}"
  printf '  Actual KV capacity and request admission are reported by SGLang after profiling.\n\n'
}
