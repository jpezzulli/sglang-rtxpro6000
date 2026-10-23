# Shared native-MTP width-policy selector: source it before the launch line.
#
# One public setting, PENNY_MTP_POLICY, picks BOTH the runtime policy file and
# the launch width that can carry it, because a candidate wider than the launch
# width is refused, not resized, under captured graphs. Keeping the two halves
# in this one file is what stops the native recipes and the mounted container
# startup copies from drifting apart. Unset keeps the qualified default, so an
# existing launch is untouched by the existence of the other choices.
#
#   tiered    the current default: W8 (steps 7) at one running request and W4
#             (steps 3) at every batch of two or more, one candidate per tier,
#             so the width is a batch-size lookup the acceptance EMA cannot move
#   adaptive  the v3.0.2 policy: C1 may pick W4 or W8 under the existing 40/80
#             batch hold and backoff, C2+ stays W4
#   fixed-w4  W4 (steps 3, 4 draft tokens) at every concurrency
#
# An invalid value is refused here rather than guessed, and a missing policy
# file is named. There is no checksum gate on the policy content: the hash of
# the file actually selected goes into the NIXL namespace below, which is what
# keeps a different or edited policy out of another selection's cache identity
# instead of turning a bundled asset into a boot refusal. Where the value comes
# from is the ordinary precedence: a key saved by ./configure-penny wins over the
# shell, a key nothing saves keeps an exported value, and a key neither states is
# the default below. A saved container setup therefore writes the choice into its
# startup script's own settings block, where the lines beside it already live.
PENNY_MTP_POLICY="${PENNY_MTP_POLICY:-tiered}"
case "$PENNY_MTP_POLICY" in
  tiered)
    PENNY_MTP_POLICY_FILE=adaptive-next.json
    PENNY_MTP_SPEC_STEPS=7
    PENNY_MTP_DRAFT_TOKENS=8
    ;;
  adaptive)
    PENNY_MTP_POLICY_FILE=adaptive-next-v3.0.2.json
    PENNY_MTP_SPEC_STEPS=7
    PENNY_MTP_DRAFT_TOKENS=8
    ;;
  fixed-w4)
    PENNY_MTP_POLICY_FILE=adaptive-next-fixed-w4.json
    PENNY_MTP_SPEC_STEPS=3
    PENNY_MTP_DRAFT_TOKENS=4
    ;;
  *)
    echo "PENNY_MTP_POLICY must be tiered, adaptive or fixed-w4; got '$PENNY_MTP_POLICY'" >&2
    exit 1
    ;;
esac
PENNY_MTP_POLICY_ASSET_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ADAPTIVE_CONFIG="$PENNY_MTP_POLICY_ASSET_DIR/$PENNY_MTP_POLICY_FILE"
[[ -r "$ADAPTIVE_CONFIG" ]] || {
  echo "MTP width policy is missing or unreadable: $ADAPTIVE_CONFIG" >&2
  exit 1
}
read -r ADAPTIVE_SHA _ < <(sha256sum "$ADAPTIVE_CONFIG")
export PENNY_MTP_POLICY
