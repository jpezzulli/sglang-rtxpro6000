# Shared, pinned chat template for both Pennyroyal model profiles.
# Its hash is also part of each launcher's persistent-cache identity.
CHAT_TEMPLATE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/templates/froggeric-v22.5.jinja"
read -r CHAT_TEMPLATE_SHA _ < <(sha256sum "$CHAT_TEMPLATE")
if [[ "$CHAT_TEMPLATE_SHA" != e57684bae4156211a55473c5a63be976a405a37ab5be5ae0e5abf1df5349c4b2 ]]; then
  echo "Froggeric v22.5 template checksum mismatch: $CHAT_TEMPLATE" >&2
  exit 1
fi
