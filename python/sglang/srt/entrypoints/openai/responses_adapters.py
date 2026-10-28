# SPDX-License-Identifier: Apache-2.0
"""Responses tools mapped onto the existing chat function-call path.

Custom input shims are adapted from upstream PR38690 (Dovis01, JustinTong0323).
Namespace inventory handling follows PR41178 (harmya), with escaped internal
names so a top-level name cannot alias a namespaced member.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional, Tuple

CUSTOM_TOOL_INPUT_KEY = "input"


def internal_tool_name(name: str, namespace: Optional[str] = None) -> str:
    """Keep ordinary names intact and reserve unescaped dots as separators.

    Escaping percent first makes this injective even for a flat ``a.b`` tool,
    namespace ``a`` / member ``b``, and names containing literal escapes.
    The mapping is stateless, so replay needs only the wire identity.
    """

    def escape(value):
        return value.replace("%", "%25").replace(".", "%2E")

    leaf = escape(name)
    return f"{escape(namespace)}.{leaf}" if namespace is not None else leaf


def tool_call_identity(name: str) -> dict[str, str]:
    def unescape(value):
        return value.replace("%2E", ".").replace("%25", "%")

    if "." in name:
        namespace, leaf = name.split(".", 1)
        return {"name": unescape(leaf), "namespace": unescape(namespace)}
    return {"name": unescape(name)}


def expand_tool_namespaces(tools):
    # Import here: protocol's named-choice translation uses internal_tool_name.
    from sglang.srt.entrypoints.openai.protocol import ResponseTool

    expanded = []
    for tool in tools:
        if tool.type == "namespace":
            for definition in tool.tools:
                child = ResponseTool.model_validate(definition)
                child.name = internal_tool_name(child.name, tool.name)
                child.description = (
                    "\n\n".join(
                        text for text in (tool.description, child.description) if text
                    )
                    or None
                )
                expanded.append(child)
        elif tool.type in ("function", "custom"):
            expanded.append(
                tool.model_copy(update={"name": internal_tool_name(tool.name)})
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
