# Shared Flash-Next request/state capacity; source before deriving the namespace.
# More admitted requests also need recurrent-state slots and graph headroom.
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-4}"
MAX_MAMBA_CACHE_SIZE="${MAX_MAMBA_CACHE_SIZE:-24}"
for capacity_name in MAX_RUNNING_REQUESTS MAX_MAMBA_CACHE_SIZE; do
  if [[ ! ${!capacity_name} =~ ^[1-9][0-9]*$ ]]; then
    echo "$capacity_name must be a positive integer" >&2
    exit 1
  fi
done
