#!/usr/bin/env bash
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
      --model-path|--speculative-draft-model-path|--speculative-algorithm|--speculative-token-map|--context-length|--max-total-tokens|--max-running-requests|--max-mamba-cache-size|--kv-cache-dtype|--speculative-draft-kv-cache-dtype|--hicache-size|--hicache-storage-backend|--image-processor-backend|--tp|--speculative-num-steps|--speculative-eagle-topk|--speculative-num-draft-tokens)
        key="$arg"; value="${2:-unspecified}"; if (( $# > 1 )); then shift; fi ;;
      --ple-offload-embedding|--enable-hierarchical-cache) key="$arg"; value=true ;;
      *) shift; continue ;;
    esac
    setting["$key"]="$value"
    shift
  done
  case "${setting[--speculative-algorithm]:-none}" in
    NEXTN)
      profile='Qwen3.8 Flash-Next / native NEXTN MTP'
      # Report the effective behavior, not a manufactured default: a saved
      # explicit choice is named as such, otherwise the eligibility the
      # recipes resolved before namespace derivation decides.
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
      ;;
    DFLASH) profile='Qwen3.8-27B / DFlash2' ;;
    *) profile="${setting[--speculative-algorithm]:-no speculation}" ;;
  esac
  [[ -z ${setting[--speculative-token-map]:-} ]] || frspec=on
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
  printf '  Speculation: steps=%s | top-k=%s | draft tokens=%s\n' \
    "${setting[--speculative-num-steps]:-not set}" \
    "${setting[--speculative-eagle-topk]:-not set}" \
    "${setting[--speculative-num-draft-tokens]:-not set}"
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
