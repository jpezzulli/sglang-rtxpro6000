# Shared Pennyroyal launcher helper for the optional image preprocessing cache;
# source before launch_args (see CONFIGURATION.md). One pair of settings, one
# argument array, every launch path: the four native recipes and the four mounted
# startup files read the settings here, so no route can accept what another
# refuses.
#
# PENNY_MM_PREPROCESS_CACHE_DIR is a private directory the tokenizer workers
# share, and PENNY_MM_PREPROCESS_CACHE_SIZE_MB is its budget in MiB
# (1 MiB = 1024*1024 bytes). Blank on both is off: no flag, no directory, nothing
# created. A budget alone is the per-worker cache each worker already keeps,
# where 0 turns that cache off. A directory needs a positive budget, because the
# budget is what bounds the retained artifacts and what the directory's own index
# records, so reusing a directory under a different budget needs a fresh
# directory. The pair is refused here, before the model loads, rather than by the
# server afterwards.
#
# In a container the directory is the path inside it: /cache/<name> and
# /dev/shm/<name> are writable with the mounts and --shm-size run.sh uses.
PENNY_MM_PREPROCESS_CACHE_ARGS=()
pennyroyal_mm_preprocess_cache_args() {
  local directory="${PENNY_MM_PREPROCESS_CACHE_DIR:-}"
  local budget="${PENNY_MM_PREPROCESS_CACHE_SIZE_MB:-}"
  if [[ "$directory" == *[$'\n\r\v\f\034\035\036\u0085\u2028\u2029']* ]]; then
    echo "PENNY_MM_PREPROCESS_CACHE_DIR must not contain line breaks or NUL" >&2
    exit 1
  fi
  # One value, exactly as written: a path with padding is a different path, and the
  # configurator refuses the same pair, so neither side launches what the other
  # would not.
  if [[ "$directory" =~ [[:space:]]$ || "$directory" =~ ^[[:space:]] ]]; then
    echo "PENNY_MM_PREPROCESS_CACHE_DIR must not have leading or trailing spaces, got '$directory'" >&2
    exit 1
  fi
  if [[ "$budget" =~ [[:space:]]$ || "$budget" =~ ^[[:space:]] ]]; then
    echo "PENNY_MM_PREPROCESS_CACHE_SIZE_MB must not have leading or trailing spaces, got '$budget'" >&2
    exit 1
  fi
  if [[ -n "$directory" ]]; then
    if [[ "$directory" != /* ]]; then
      echo "PENNY_MM_PREPROCESS_CACHE_DIR must be an absolute directory, got '$directory'" >&2
      exit 1
    fi
    if [[ -z "$budget" ]]; then
      echo "PENNY_MM_PREPROCESS_CACHE_DIR needs a positive PENNY_MM_PREPROCESS_CACHE_SIZE_MB budget in MiB, got 'nothing'" >&2
      exit 1
    fi
    if [[ ! "$budget" =~ ^[0-9]+$ ]]; then
      echo "PENNY_MM_PREPROCESS_CACHE_SIZE_MB must be a whole number of MiB (1 MiB = 1048576 bytes), got '$budget'" >&2
      exit 1
    fi
    # Positivity by string, not by (( )): bash arithmetic is signed 64-bit and
    # would turn a budget too large to count into a refusal or a wrap.
    if [[ "$budget" =~ ^0+$ ]]; then
      echo "PENNY_MM_PREPROCESS_CACHE_DIR needs a positive PENNY_MM_PREPROCESS_CACHE_SIZE_MB budget in MiB, got '$budget'" >&2
      exit 1
    fi
    PENNY_MM_PREPROCESS_CACHE_ARGS=(
      --mm-preprocess-cache-dir "$directory"
      --mm-preprocess-cache-size-mb "$budget"
    )
  elif [[ -n "$budget" ]]; then
    if [[ ! "$budget" =~ ^[0-9]+$ ]]; then
      echo "PENNY_MM_PREPROCESS_CACHE_SIZE_MB must be a whole number of MiB (1 MiB = 1048576 bytes), got '$budget'" >&2
      exit 1
    fi
    PENNY_MM_PREPROCESS_CACHE_ARGS=(--mm-preprocess-cache-size-mb "$budget")
  fi
}
