# SPDX-License-Identifier: Apache-2.0
"""Responses tools mapped onto the existing chat function-call path.

Custom input shims are adapted from upstream PR38690 (Dovis01, JustinTong0323).
Namespace inventory handling follows PR41178 (harmya).

The chat function-call path has one flat name space, so a namespaced member
needs a derived single name. Callers document that name with the generic MCP
convention (``mcp__server__tool``), so the derivation prefers that spelling and
keeps an explicit reversible binding for every declaration: prompt, named and
required ``tool_choice``, parsed calls and replayed history all read the same
table. Ordinary flat names are never rewritten, and a name the model produced
that is not in the table is passed through instead of being guessed apart.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional, Tuple

from sglang.srt.entrypoints.openai.protocol import ResponseTool

CUSTOM_TOOL_INPUT_KEY = "input"

#: Generic MCP naming convention: server, two underscores, tool.
IDENTITY_SEPARATOR = "__"

WireIdentity = Tuple[Optional[str], str]


def _escape_part(value: str) -> str:
    """Escape everything a derived spelling can be split back apart on.

    Percent goes first, which keeps this injective for names that already
    contain literal escapes.
    """
    return value.replace("%", "%25").replace(".", "%2E").replace("__", "%5F%5F")


def identity_spelling(
    name: str, namespace: Optional[str] = None, collision_level: int = 0
) -> str:
    """One model-visible spelling for one declared identity.

    Level 0 is the generic convention. A contested level-0 spelling escalates
    to the escaped ``namespace.leaf`` form and then re-escapes it per level, so
    every level denotes exactly one identity.
    """
    if collision_level == 0:
        return name if namespace is None else f"{namespace}{IDENTITY_SEPARATOR}{name}"
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
    onto one shared name. History bindings stay history-only -- never selectable
    and never the destination of a parsed call -- because appearing in a
    transcript does not make a tool callable again.
    """

    def __init__(
        self,
        declarations: Iterable[Tuple[Optional[str], str, str]] = (),
        replayed: Iterable[WireIdentity] = (),
    ):
        # ``type`` only distinguishes custom shims from function calls, and only
        # callable declarations carry one.
        declared: Dict[WireIdentity, str] = {}
        for namespace, name, tool_type in declarations:
            declared[(namespace, name)] = tool_type
        replay_only = [wire for wire in dict.fromkeys(replayed) if wire not in declared]
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
                spelling = identity_spelling(name, namespace, level)
                while spelling in self._wire_of_name:
                    level += 1
                    spelling = identity_spelling(name, namespace, level)
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
        spelling = identity_spelling(name, namespace, level)
        while spelling in self._wire_of_name:
            level += 1
            spelling = identity_spelling(name, namespace, level)
        return spelling

    def declares(self, name: str, namespace: Optional[str] = None) -> bool:
        return (namespace, name) in self._declared

    def wire_identity(self, model_name: str) -> Optional[dict[str, str]]:
        """Wire ``name``/``namespace`` of a model-emitted name, or None when this
        request never declared that tool callable: no identity is guessed out of a
        string, and a name that only survives in history routes nowhere.
        """
        if model_name not in self._types:
            return None
        namespace, name = self._wire_of_name[model_name]
        return {"name": name, "namespace": namespace} if namespace else {"name": name}

    def is_custom(self, model_name: str) -> bool:
        return self._types.get(model_name) == "custom"


def tool_identities(tools, replayed: Iterable[WireIdentity] = ()) -> ToolIdentities:
    """Build the bindings for one set of Responses tool declarations.

    ``replayed`` takes the explicit wire identities of replayed call items, so a
    retired tool keeps a spelling of its own without becoming callable.
    """
    declarations = []
    for tool in tools:
        if tool.type == "namespace":
            declarations.extend(
                (tool.name, child["name"], child["type"]) for child in tool.tools
            )
        elif tool.type in ("function", "custom"):
            declarations.append((None, tool.name, tool.type))
    return ToolIdentities(declarations, replayed)


def expand_tool_namespaces(tools, identities: Optional[ToolIdentities] = None):
    if identities is None:
        identities = tool_identities(tools)
    expanded: List[Any] = []
    for tool in tools:
        if tool.type == "namespace":
            for definition in tool.tools:
                child = ResponseTool.model_validate(definition)
                child.name = identities.model_name(child.name, tool.name)
                child.description = (
                    "\n\n".join(
                        text for text in (tool.description, child.description) if text
                    )
                    or None
                )
                expanded.append(child)
        elif tool.type in ("function", "custom"):
            expanded.append(
                tool.model_copy(update={"name": identities.model_name(tool.name)})
            )
    return expanded


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
