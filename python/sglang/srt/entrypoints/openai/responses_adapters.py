# SPDX-License-Identifier: Apache-2.0
"""Translations between Responses-API wire shapes and SGLang's chat internals.

Two things the Responses API expresses that the chat-completions path has no
representation for:

* ``custom`` tools, whose payload is freeform text rather than JSON-object
  arguments. Each is surfaced to the model as a function tool with a single
  string property, and the resulting call is translated back into a
  ``custom_tool_call``.
* ``reasoning.encrypted_content``, the opaque blob a ``store=false`` client
  replays to hand a reasoning trace back to the server.

The chat function-call path has one flat name space, so a namespaced member
needs a derived single name. Callers document that name with the generic MCP
convention (``mcp__server__tool``) or, where a template spells it with a dot,
with the separator it advertises, and the derivation keeps an explicit
reversible binding for every declaration: prompt, named and required
``tool_choice``, parsed calls and replayed history all read the same table.
Ordinary flat names are never rewritten, and a name the model produced that is
not in the table is passed through instead of being guessed apart.
"""

from __future__ import annotations

import base64
import json
import zlib
from typing import Any, Dict, Iterable, Iterator, Optional, Set, Tuple

from sglang.srt.entrypoints.openai.protocol import ResponseTool

CUSTOM_TOOL_INPUT_KEY = "input"

#: Default model-visible join for a namespace: the generic MCP convention.
IDENTITY_SEPARATOR = "__"

WireIdentity = Tuple[Optional[str], str]


def _escape_part(value: str) -> str:
    """Escape everything a derived spelling can be split back apart on.

    Percent goes first, which keeps this injective for names that already
    contain literal escapes.
    """
    return value.replace("%", "%25").replace(".", "%2E").replace("__", "%5F%5F")


def identity_spelling(
    name: str,
    namespace: Optional[str] = None,
    collision_level: int = 0,
    separator: str = IDENTITY_SEPARATOR,
) -> str:
    """One model-visible spelling for one declared identity.

    Level 0 is the documented convention. A contested level-0 spelling
    escalates to the escaped ``namespace.leaf`` form and then re-escapes it per
    level, so every level denotes exactly one identity.
    """
    if collision_level == 0:
        return name if namespace is None else f"{namespace}{separator}{name}"
    spelling = _escape_part(name)
    if namespace is not None:
        spelling = f"{_escape_part(namespace)}.{spelling}"
    for _ in range(collision_level - 1):
        spelling = _escape_part(spelling)
    return spelling


class ToolIdentities:
    """Declaration-derived bindings between wire identities and model names.

    Replayed history can still name a tool the request no longer declares, so
    those identities are bound to a spelling too: retired tools cannot collapse
    onto one shared name. History bindings stay history-only -- never
    selectable and never the destination of a parsed call -- because appearing
    in a transcript does not make a tool callable again.
    """

    def __init__(
        self,
        declarations: Iterable[Tuple[Optional[str], str, str]] = (),
        replayed: Iterable[WireIdentity] = (),
        separator: str = IDENTITY_SEPARATOR,
    ):
        # ``type`` only distinguishes custom shims from function calls, and only
        # callable declarations carry one.
        declared: Dict[WireIdentity, str] = {}
        for namespace, name, tool_type in declarations:
            declared[(namespace, name)] = tool_type
        replay_only = [wire for wire in dict.fromkeys(replayed) if wire not in declared]
        self._separator = separator
        self._declared = set(declared)
        self._types: Dict[str, str] = {}
        self._wire_of_name: Dict[str, WireIdentity] = {}
        self._name_of_wire: Dict[WireIdentity, str] = {}
        # Callables claim first -- flat declarations before namespace members, so
        # an ordinary name survives no matter what a namespace flattens to -- and
        # retired tools land in the spellings left over. Both passes are keyed by
        # identity, never by the order an inventory or a history arrived in.
        for wires, callable_ in ((declared, True), (replay_only, False)):
            for wire in sorted(
                wires, key=lambda i: (i[0] is not None, i[0] or "", i[1])
            ):
                namespace, name = wire
                level = 0
                spelling = identity_spelling(name, namespace, level, self._separator)
                while spelling in self._wire_of_name:
                    level += 1
                    spelling = identity_spelling(
                        name, namespace, level, self._separator
                    )
                self._wire_of_name[spelling] = wire
                self._name_of_wire[wire] = spelling
                if callable_:
                    self._types[spelling] = declared[wire]

    def model_name(self, name: str, namespace: Optional[str] = None) -> str:
        """Spelling bound to ``namespace``/``name``, callable or history-only.

        An identity this object never learned about still renders as one
        deterministic name and cannot take a spelling something else already
        holds.
        """
        known = self._name_of_wire.get((namespace, name))
        if known is not None:
            return known
        level = 0
        spelling = identity_spelling(name, namespace, level, self._separator)
        while spelling in self._wire_of_name:
            level += 1
            spelling = identity_spelling(name, namespace, level, self._separator)
        return spelling

    def declares(self, name: str, namespace: Optional[str] = None) -> bool:
        return (namespace, name) in self._declared

    def wire_identity(self, model_name: str) -> Optional[Dict[str, str]]:
        """Wire ``name``/``namespace`` of a model-emitted name, or None when this
        request never declared that tool callable: no identity is guessed out of
        a string, and a name that only survives in history routes nowhere.
        """
        if model_name not in self._types:
            return None
        namespace, name = self._wire_of_name[model_name]
        return {"name": name, "namespace": namespace} if namespace else {"name": name}

    def is_custom(self, model_name: str) -> bool:
        return self._types.get(model_name) == "custom"

    @property
    def custom_names(self) -> Set[str]:
        """Spellings that carry a freeform payload, not JSON arguments."""
        return {spelling for spelling, kind in self._types.items() if kind == "custom"}

    @property
    def names(self) -> Dict[str, Dict[str, str]]:
        """Callable spellings as the ``response_tool_name`` mapping shape."""
        return {
            spelling: identity
            for spelling, identity in (
                (spelling, self.wire_identity(spelling)) for spelling in self._types
            )
            if identity is not None
        }


def tool_identities(
    tools: Any,
    replayed: Iterable[WireIdentity] = (),
    separator: str = IDENTITY_SEPARATOR,
) -> ToolIdentities:
    """Build the bindings for one set of Responses tool declarations.

    ``replayed`` takes the explicit wire identities of replayed call items, so a
    retired tool keeps a spelling of its own without becoming callable.
    """
    declarations = []
    for tool in tools or []:
        if tool.type == "namespace" and tool.name:
            for member_type, member in namespace_members(tool):
                declarations.append((tool.name, member["name"], member_type))
        elif tool.type in ("function", "custom") and tool.name:
            declarations.append((None, tool.name, tool.type))
    return ToolIdentities(declarations, replayed, separator)


def expand_tool_namespaces(
    tools: Any, identities: Optional[ToolIdentities] = None
) -> list:
    """Flatten every namespace declaration onto the callable it binds to.

    The chat function-call path has one flat name space, so a namespaced member
    is offered under its bound spelling for every parser, with the namespace
    description kept on it. Ordinary declarations pass through, renamed only if
    the table had to move them.
    """
    if identities is None:
        identities = tool_identities(tools)
    expanded = []
    for tool in tools or []:
        if tool.type == "namespace" and tool.name:
            for member_type, definition in namespace_members(tool):
                child = ResponseTool.model_validate({**definition, "type": member_type})
                child.name = identities.model_name(child.name, tool.name)
                child.description = (
                    "\n\n".join(
                        text for text in (tool.description, child.description) if text
                    )
                    or None
                )
                expanded.append(child)
        elif tool.type in ("function", "custom") and tool.name:
            expanded.append(
                tool.model_copy(update={"name": identities.model_name(tool.name)})
            )
    return expanded


def response_tool_name(name, names):
    return names.get(name, {"name": name})


_SIMPLE_ESCAPES = {
    '"': '"',
    "\\": "\\",
    "/": "/",
    "b": "\b",
    "f": "\f",
    "n": "\n",
    "r": "\r",
    "t": "\t",
}


def custom_tool_parameters() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            CUSTOM_TOOL_INPUT_KEY: {
                "type": "string",
                "description": "The freeform text payload for this tool.",
            }
        },
        "required": [CUSTOM_TOOL_INPUT_KEY],
        "additionalProperties": False,
    }


def custom_tool_description(
    description: Optional[str], tool_format: Optional[Dict[str, Any]]
) -> Optional[str]:
    """Fold a custom tool's description and its input format into one blob.

    A ``grammar`` format is described to the model rather than enforced by the
    grammar backend: the payload is emitted inside the model's own tool-call
    envelope, so constrained decoding cannot be scoped to it.
    """
    parts = []
    if description:
        parts.append(description)
    if isinstance(tool_format, dict) and tool_format.get("type") == "grammar":
        syntax = tool_format.get("syntax") or "lark"
        definition = tool_format.get("definition") or ""
        parts.append(
            f"The payload must conform to this {syntax} grammar:\n{definition}"
        )
    return "\n\n".join(parts) or None


def namespace_members(tool: Any) -> Iterator[Tuple[str, Dict[str, Any]]]:
    """Yield ``(member_type, definition)`` for the flat and OpenAI-wrapped
    members accepted by templates that render namespace tools."""
    for member in tool.tools or []:
        member_type = member.get("type", "function")
        function = member.get("function", member)
        if isinstance(function, dict) and function.get("name"):
            yield member_type, function


def encode_custom_tool_input(payload: str) -> str:
    return json.dumps({CUSTOM_TOOL_INPUT_KEY: payload}, ensure_ascii=False)


def decode_custom_tool_input(arguments: str) -> str:
    """Recover the payload from a completed shim call's arguments.

    Falls back to the raw string so a model that ignored the wrapper and
    emitted bare text still produces a usable call.
    """
    try:
        parsed = json.loads(arguments)
    except ValueError:
        return arguments
    if isinstance(parsed, dict):
        value = parsed.get(CUSTOM_TOOL_INPUT_KEY)
        if isinstance(value, str):
            return value
        if len(parsed) == 1:
            only = next(iter(parsed.values()))
            if isinstance(only, str):
                return only
    return arguments


def _payload_start(buffer: str) -> Optional[int]:
    # Skip complete top-level fields with the standard decoder. Searching for a
    # key substring would mistake nested input fields or a later key's quotes
    # after a non-string input value for the custom payload.
    decoder = json.JSONDecoder()

    def skip_space(index):
        while index < len(buffer) and buffer[index] in " \t\r\n":
            index += 1
        return index

    index = skip_space(0)
    if index >= len(buffer) or buffer[index] != "{":
        return None
    index = skip_space(index + 1)
    while index < len(buffer):
        try:
            key, index = decoder.raw_decode(buffer, index)
        except ValueError:
            return None
        index = skip_space(index)
        if not isinstance(key, str) or index >= len(buffer) or buffer[index] != ":":
            return None
        index = skip_space(index + 1)
        if (
            key == CUSTOM_TOOL_INPUT_KEY
            and index < len(buffer)
            and buffer[index] == '"'
        ):
            return index + 1
        try:
            _, index = decoder.raw_decode(buffer, index)
        except ValueError:
            return None
        index = skip_space(index)
        if index >= len(buffer) or buffer[index] != ",":
            return None
        index = skip_space(index + 1)
    return None


def _unicode_escape(buffer: str, i: int) -> Optional[Tuple[str, int]]:
    if i + 6 > len(buffer):
        return None
    try:
        code = int(buffer[i + 2 : i + 6], 16)
    except ValueError:
        return None
    if not 0xD800 <= code <= 0xDBFF:
        return chr(code), i + 6
    if i + 12 > len(buffer) or buffer[i + 6 : i + 8] != "\\u":
        return None
    try:
        low = int(buffer[i + 8 : i + 12], 16)
    except ValueError:
        return None
    if not 0xDC00 <= low <= 0xDFFF:
        return None
    return chr(0x10000 + ((code - 0xD800) << 10) + (low - 0xDC00)), i + 12


def decode_custom_tool_input_prefix(buffer: str) -> str:
    """Longest fully decodable prefix of the payload in a partial argument buffer.

    Streaming tool-call parsers hand out argument JSON in fragments, so the
    payload is un-escaped incrementally to keep ``custom_tool_call_input``
    deltas in step with the value the item finally reports.
    """
    start = _payload_start(buffer)
    if start is None:
        return ""
    out = []
    i, n = start, len(buffer)
    while i < n:
        ch = buffer[i]
        if ch == '"':
            break
        if ch != "\\":
            out.append(ch)
            i += 1
            continue
        if i + 1 >= n:
            break
        simple = _SIMPLE_ESCAPES.get(buffer[i + 1])
        if simple is not None:
            out.append(simple)
            i += 2
            continue
        if buffer[i + 1] == "u":
            decoded = _unicode_escape(buffer, i)
            if decoded is None:
                break
            out.append(decoded[0])
            i = decoded[1]
            continue
        break
    return "".join(out)


DEVELOPER_BLOCK_LABEL = "Developer instructions:"


def label_developer_content(content: Any) -> Any:
    """Prefix a ``developer`` message's text with its tier label.

    Chat templates recognize system/user/assistant/tool, so a developer message
    collapses to ``system``. Labelling the block keeps the tier legible instead
    of flattening it into the surrounding instructions, where a model can read
    it as one more conversational turn and let a later user message win.
    """
    if isinstance(content, str):
        return f"{DEVELOPER_BLOCK_LABEL}\n{content}"
    if isinstance(content, list):
        for index, part in enumerate(content):
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                labelled = dict(part)
                labelled["text"] = f"{DEVELOPER_BLOCK_LABEL}\n{part['text']}"
                return [*content[:index], labelled, *content[index + 1 :]]
        return [{"type": "input_text", "text": DEVELOPER_BLOCK_LABEL}, *content]
    return content


_REASONING_STATE_PREFIX = "sglang-reasoning-v1."


def encode_reasoning_state(text: str) -> str:
    """Pack a reasoning trace into the blob ``reasoning.encrypted_content`` carries.

    A ``store=false`` client replays reasoning items verbatim, so the trace has
    to survive the round trip without server-side state. SGLang holds no key,
    so this is an encoded -- not cryptographically protected -- payload: treat
    it as opaque to clients, not as a confidentiality boundary.
    """
    packed = base64.urlsafe_b64encode(zlib.compress(text.encode("utf-8")))
    return _REASONING_STATE_PREFIX + packed.decode("ascii")


def decode_reasoning_state(blob: Any) -> Optional[str]:
    if not isinstance(blob, str) or not blob.startswith(_REASONING_STATE_PREFIX):
        return None
    try:
        raw = base64.urlsafe_b64decode(blob[len(_REASONING_STATE_PREFIX) :])
        return zlib.decompress(raw).decode("utf-8")
    except (ValueError, zlib.error, UnicodeDecodeError):
        return None
