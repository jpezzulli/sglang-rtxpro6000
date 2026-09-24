# Shared Pennyroyal launcher reasoning-effort convenience (PR#18); source
# before launch_args. This is launcher-only policy: the server keeps its own
# behavior, and an explicit per-request reasoning_effort still wins at
# serving time through the request-first precedence in
# OpenAIServingChat._process_messages.
#
# Builds the qualified --default-chat-template-kwargs JSON for every recipe
# from one place: enable_thinking/preserve_thinking stay pinned,
# reasoning_effort defaults to medium, and PENNY_REASONING_EFFORT (unset,
# empty, or whitespace-only keeps medium; an accepted tier rewrites just that
# key) selects the tier. Accepted values are the OpenAI tiers
# none|minimal|low|medium|high|xhigh|max -- no server-side env handling and
# no float extension here. An invalid value exits non-zero at launch rather
# than booting a server that rejects every request.
PENNY_REASONING_EFFORT_TIERS="none minimal low medium high xhigh max"
PENNY_REASONING_EFFORT="${PENNY_REASONING_EFFORT:-}"
# Trim only the ends, like the tier comparison below expects; internal
# whitespace stays invalid.
PENNY_REASONING_EFFORT_TRIMMED="$(printf '%s' "$PENNY_REASONING_EFFORT" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
PENNY_REASONING_EFFORT_NORMALIZED="medium"
if [[ -n "$PENNY_REASONING_EFFORT_TRIMMED" ]]; then
  PENNY_REASONING_EFFORT_LOWER="$(printf '%s' "$PENNY_REASONING_EFFORT_TRIMMED" | tr '[:upper:]' '[:lower:]')"
  for tier in $PENNY_REASONING_EFFORT_TIERS; do
    if [[ "$PENNY_REASONING_EFFORT_LOWER" == "$tier" ]]; then
      PENNY_REASONING_EFFORT_NORMALIZED="$tier"
      break
    fi
  done
  if [[ "$PENNY_REASONING_EFFORT_NORMALIZED" == "medium" && "$PENNY_REASONING_EFFORT_LOWER" != "medium" ]]; then
    echo "PENNY_REASONING_EFFORT must be one of: $PENNY_REASONING_EFFORT_TIERS; got '$PENNY_REASONING_EFFORT_TRIMMED'" >&2
    exit 1
  fi
fi
# Plain single-quoted heredoc: no expansion surprises, one JSON line, and
# byte-for-byte the qualified default when no tier was requested.
DEFAULT_CHAT_TEMPLATE_KWARGS="$(cat <<EOF
{"enable_thinking":true,"preserve_thinking":true,"reasoning_effort":"${PENNY_REASONING_EFFORT_NORMALIZED}"}
EOF
)"
