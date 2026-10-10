import json
import logging
import re
from typing import Any, List, Optional

from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.environ import envs
from sglang.srt.function_call.base_format_detector import BaseFormatDetector
from sglang.srt.function_call.core_types import (
    StreamingParseResult,
    ToolCallItem,
    _GetInfoFunc,
)
from sglang.srt.function_call.utils import (
    TOOL_PARSER_ACCEPTED,
    TOOL_PARSER_ARGUMENT_CONVERSION_FAILED,
    TOOL_PARSER_INCOMPLETE,
    TOOL_PARSER_PARSE_ERROR,
    TOOL_PARSER_UNKNOWN_TOOL,
    infer_type_from_json_schema,
    record_tool_parser_event,
    safe_literal_eval,
)

logger = logging.getLogger(__name__)


class Qwen3CoderDetector(BaseFormatDetector):
    def __init__(self):
        super().__init__()

        # Sentinel tokens
        self.tool_call_start_token: str = "<tool_call>"
        self.tool_call_end_token: str = "</tool_call>"
        self.tool_call_prefix: str = "<function="
        self.function_end_token: str = "</function>"
        self.parameter_prefix: str = "<parameter="
        self.parameter_end_token: str = "</parameter>"

        # Parameter conversion remains compatible with the non-streaming fallback.
        self.tool_call_parameter_regex = re.compile(
            r"<parameter=(.*?)(?:</parameter>|(?=<parameter=)|(?=</function>)|$)",
            re.DOTALL,
        )

        # Streaming State
        # Base class already initializes _buffer, we just use it directly
        # No need to check with hasattr - we control the lifecycle through inheritance

        # Index pointing to the next character to be processed in buffer
        self.parsed_pos: int = 0
        # Parameter count inside the current tool being processed, used to determine whether to add comma
        self.current_tool_param_count: int = 0
        # Flag indicating whether current tool has already sent '{'
        self.json_started: bool = False

        # [FIX] New state flag: mark whether inside tool_call structure block
        self.is_inside_tool_call: bool = False

        # After a tool closes, whitespace is ambiguous until the next tag or
        # genuine text arrives. Keep it out of SSE content while arguments
        # from the same increment may still be waiting to be emitted (#37408).
        self._pending_tool_separator: Optional[str] = None

        # Initialize attributes that were missing in the original PR
        self.current_func_name: Optional[str] = None
        self._stream_string: Optional[dict] = None

        # Hold the wrapper until its name is validated. Rejected examples must
        # survive as literal text, including markup received in earlier chunks.
        self._pending_tool_prefix: Optional[str] = None
        self._tool_wrapper_prefix = ""
        self._suppress_current_call = False
        self._rejected_function_depth = 0
        self._rejected_tool_depth = 0
        self._preserve_tool_wrapper = False

        # Markdown fenced examples are literal output, even when their body is
        # valid tool syntax. Streaming holds only a possibly structural line
        # prefix; ordinary body content is forwarded immediately.
        self._code_fence_marker: Optional[str] = None
        self._code_fence_length = 0
        self._code_fence_line_can_close = True
        self._stream_at_line_start = True

        # Per-parameter flag for aggregate conversion-fallback accounting.
        self._conversion_fallback = False
        # Guards aggregate incomplete accounting against repeated flushes.
        self._incomplete_recorded = False
        self._superseded_incomplete_calls = 0

    def has_tool_call(self, text: str) -> bool:
        return self.tool_call_start_token in text

    def _get_arguments_config(
        self, func_name: str, tools: Optional[list[Tool]]
    ) -> dict:
        """Extract argument configuration for a function."""
        if tools is None:
            return {}
        for config in tools:
            try:
                config_type = config.type
                config_function = config.function
                config_function_name = config_function.name
            except AttributeError:
                continue

            if config_type == "function" and config_function_name == func_name:
                try:
                    params = config_function.parameters
                except AttributeError:
                    return {}

                if isinstance(params, dict) and "properties" in params:
                    return params["properties"]
                elif isinstance(params, dict):
                    return params
                else:
                    return {}
        logger.warning(f"Tool '{func_name}' is not defined in the tools list.")
        return {}

    def _get_param_type(self, param_schema: Any) -> str:
        """Infer the parser conversion type from a JSON schema parameter."""
        inferred_type = infer_type_from_json_schema(param_schema)
        if inferred_type is None:
            return "string"
        return str(inferred_type).strip().lower()

    def _is_declared_tool(self, func_name: str, tools: Optional[List[Tool]]) -> bool:
        if not tools or envs.SGLANG_FORWARD_UNKNOWN_TOOLS.get():
            return True
        return self._tool_name_known(func_name, tools)

    @staticmethod
    def _tool_name_known(func_name: str, tools: Optional[List[Tool]]) -> bool:
        """Name declared or not, ignoring the unknown-tool forwarding switch.

        Only used for aggregate accounting, so a forwarded unknown tool stays
        visible as an unknown_tool event.
        """
        return any(
            tool.type == "function" and tool.function.name == func_name
            for tool in tools or []
        )

    def _convert_param_value(
        self, param_value: str, param_name: str, param_config: dict, func_name: str
    ) -> Any:
        """Convert a parameter value and record one aggregate fallback event.

        A value that fails several strategies (json.loads then literal_eval)
        still counts once for that parameter.
        """
        self._conversion_fallback = False
        converted = self._convert_param_value_typed(
            param_value, param_name, param_config, func_name
        )
        if self._conversion_fallback:
            self._conversion_fallback = False
            record_tool_parser_event(TOOL_PARSER_ARGUMENT_CONVERSION_FAILED)
        return converted

    def _convert_param_value_typed(
        self, param_value: str, param_name: str, param_config: dict, func_name: str
    ) -> Any:
        """Convert parameter value based on its type in the schema."""
        # Handle null value for any type
        if param_value.lower() == "null":
            return None

        if param_name not in param_config:
            if param_config != {}:
                logger.warning(
                    f"Parsed parameter '{param_name}' is not defined in the tool "
                    f"parameters for tool '{func_name}', directly returning the string value."
                )
            return param_value

        param_type = self._get_param_type(param_config[param_name])
        if param_type in ["string", "str", "text", "varchar", "char", "enum"]:
            return param_value
        elif (
            param_type.startswith("int")
            or param_type.startswith("uint")
            or param_type.startswith("long")
            or param_type.startswith("short")
            or param_type.startswith("unsigned")
        ):
            try:
                param_value = int(param_value)
            except Exception:
                logger.warning(
                    f"Parsed value '{param_value}' of parameter '{param_name}' is not an integer in tool "
                    f"'{func_name}', degenerating to string."
                )
                self._conversion_fallback = True
            return param_value
        elif param_type.startswith("num") or param_type.startswith("float"):
            try:
                maybe_convert = (
                    False if "." in param_value or "e" in param_value.lower() else True
                )
                param_value: float = float(param_value)
                if maybe_convert and param_value.is_integer():
                    param_value = int(param_value)
            except Exception:
                logger.warning(
                    f"Parsed value '{param_value}' of parameter '{param_name}' is not a float in tool "
                    f"'{func_name}', degenerating to string."
                )
                self._conversion_fallback = True
            return param_value
        elif param_type in ["boolean", "bool", "binary"]:
            param_value = param_value.lower()
            if param_value not in ["true", "false"]:
                logger.warning(
                    f"Parsed value '{param_value}' of parameter '{param_name}' is not a boolean (`true` of `false`) in tool '{func_name}', degenerating to false."
                )
                self._conversion_fallback = True
            return param_value == "true"
        else:
            if (
                param_type in ["object", "array", "arr"]
                or param_type.startswith("dict")
                or param_type.startswith("list")
            ):
                try:
                    param_value = json.loads(param_value)
                    return param_value
                except Exception:
                    logger.warning(
                        f"Parsed value '{param_value}' of parameter '{param_name}' cannot be parsed with json.loads in tool "
                        f"'{func_name}', will try other methods to parse it."
                    )
                    self._conversion_fallback = True
            try:
                param_value = safe_literal_eval(param_value)
            except Exception:
                logger.warning(
                    f"Parsed value '{param_value}' of parameter '{param_name}' cannot be converted via Python `ast.literal_eval()` in tool '{func_name}', degenerating to string."
                )
                self._conversion_fallback = True
            return param_value

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        """One-shot parsing for non-streaming scenarios."""
        if self.tool_call_start_token not in text:
            return StreamingParseResult(normal_text=text)

        calls = []
        try:
            raw_tool_calls = list(self._iter_unquoted_tool_spans(text))
            # Preserve the legacy incomplete-wrapper fallback only when there
            # are no complete wrappers; do not promote a truncated trailing one.
            complete_tool_calls = [
                span for span in raw_tool_calls if span[1] != span[2]
            ]
            if complete_tool_calls:
                # A trailing unfinished wrapper stays prose when earlier calls
                # are complete. Still account for a genuine unfinished function
                # inside it without promoting that function into a tool call.
                for tool_start, tool_end, content_end in raw_tool_calls:
                    if tool_end != content_end:
                        continue
                    content = text[
                        tool_start + len(self.tool_call_start_token) : content_end
                    ]
                    for start, end, body_end in self._iter_structure_spans(
                        content, self.tool_call_prefix, self.function_end_token
                    ):
                        if end != body_end:
                            continue
                        name, separator, _ = content[
                            start + len(self.tool_call_prefix) : body_end
                        ].partition(">")
                        if separator and name:
                            if tools and not self._tool_name_known(name, tools):
                                record_tool_parser_event(TOOL_PARSER_UNKNOWN_TOOL)
                            if self._is_declared_tool(name, tools):
                                record_tool_parser_event(TOOL_PARSER_INCOMPLETE)
            raw_tool_calls = complete_tool_calls or raw_tool_calls

            tool_idx = 0
            normal_text_chunks = []
            text_pos = 0
            for tool_start, tool_end, content_end in raw_tool_calls:
                self._append_normal_text(text[text_pos:tool_start], normal_text_chunks)
                tool_content = text[
                    tool_start + len(self.tool_call_start_token) : content_end
                ]
                # Find function calls
                funcs = list(
                    self._iter_structure_spans(
                        tool_content, self.tool_call_prefix, self.function_end_token
                    )
                )
                accepted_spans = []
                for func_start, func_end, body_end in funcs:
                    func_body = tool_content[
                        func_start + len(self.tool_call_prefix) : body_end
                    ]
                    if ">" not in func_body:
                        continue

                    name_end = func_body.index(">")
                    func_name = func_body[:name_end]
                    if not self._is_declared_tool(func_name, tools):
                        logger.warning(
                            "Model attempted to call undefined function: %s", func_name
                        )
                        record_tool_parser_event(TOOL_PARSER_UNKNOWN_TOOL)
                        continue
                    params_str = func_body[name_end + 1 :]

                    param_config = self._get_arguments_config(func_name, tools)
                    parsed_params = {}

                    for p_match in self.tool_call_parameter_regex.findall(params_str):
                        if ">" not in p_match:
                            continue
                        p_idx = p_match.index(">")
                        p_name = p_match[:p_idx]
                        p_val = p_match[p_idx + 1 :]
                        # Remove prefixing and trailing \n
                        if p_val.startswith("\n"):
                            p_val = p_val[1:]
                        if p_val.endswith("\n"):
                            p_val = p_val[:-1]

                        parsed_params[p_name] = self._convert_param_value(
                            p_val, p_name, param_config, func_name
                        )

                    calls.append(
                        ToolCallItem(
                            tool_index=tool_idx,
                            name=func_name,
                            parameters=json.dumps(parsed_params, ensure_ascii=False),
                        )
                    )
                    tool_idx += 1
                    accepted_spans.append((func_start, func_end))
                    record_tool_parser_event(TOOL_PARSER_ACCEPTED)
                    if func_end == body_end == len(tool_content):
                        # Accepted call whose own closing markup never arrived
                        # (its wrapper or the response ended first); the
                        # streaming path records the same shape at termination.
                        record_tool_parser_event(TOOL_PARSER_INCOMPLETE)
                    if tools and not self._tool_name_known(func_name, tools):
                        # Undeclared name accepted because unknown-tool
                        # forwarding is on; overlaps accepted by design.
                        record_tool_parser_event(TOOL_PARSER_UNKNOWN_TOOL)

                if accepted_spans and len(accepted_spans) == len(funcs):
                    self._pending_tool_separator = ""
                else:
                    # Remove only accepted functions from a mixed wrapper; an
                    # undeclared call is prose, not permission to drop content.
                    literal = text[tool_start:tool_end]
                    offset = len(self.tool_call_start_token)
                    for start, end in reversed(accepted_spans):
                        literal = literal[: offset + start] + literal[offset + end :]
                    self._append_normal_text(literal, normal_text_chunks)
                text_pos = tool_end

            self._append_normal_text(text[text_pos:], normal_text_chunks)
            normal_text = "".join(normal_text_chunks)
            self._pending_tool_separator = None

            return StreamingParseResult(normal_text=normal_text, calls=calls)

        except Exception as e:
            logger.error(f"Error in detect_and_parse: {e}")
            record_tool_parser_event(TOOL_PARSER_PARSE_ERROR)
            return StreamingParseResult(normal_text=text)

    @staticmethod
    def _iter_structure_spans(text: str, start_token: str, end_token: str):
        """Yield outer structure spans without promoting nested literal markup.

        An unknown function owns its entire body, including nested examples.
        A lazy first-close regex would expose later nested functions as siblings.
        Incomplete outer structures keep the existing end-of-input fallback.
        """
        tokens = re.compile(re.escape(start_token) + "|" + re.escape(end_token))
        pos = 0
        while (start := text.find(start_token, pos)) != -1:
            depth = 1
            end = body_end = len(text)
            for match in tokens.finditer(text, start + len(start_token)):
                depth += 1 if match.group() == start_token else -1
                if depth == 0:
                    body_end, end = match.start(), match.end()
                    break
            yield start, end, body_end
            pos = end

    @staticmethod
    def _find_fence_open(text: str, pos: int = 0, *, require_newline: bool = False):
        """Find the next ordinary Markdown fence opening line."""
        line_end = r"(?:\r\n|\r|\n)" if require_newline else r"(?:\r\n|\r|\n|$)"
        pattern = re.compile(
            rf"(?:(?<=\n)|(?<=\r)|\A) {{0,3}}"
            rf"(?P<marker>`{{3,}}|~{{3,}})(?P<info>[^\r\n]*){line_end}",
        )
        while match := pattern.search(text, pos):
            marker = match.group("marker")
            # CommonMark forbids backticks in an info string for a backtick
            # fence. Tilde fences have no equivalent restriction.
            if marker[0] != "`" or "`" not in match.group("info"):
                return match
            pos = match.end()
        return None

    @staticmethod
    def _find_fence_close(text: str, pos: int, marker: str, length: int):
        return re.compile(
            rf"(?:(?<=\n)|(?<=\r)|\A) {{0,3}}"
            rf"{re.escape(marker)}{{{length},}}[ \t]*(?:\r\n|\r|\n|$)",
        ).search(text, pos)

    def _iter_unquoted_tool_spans(self, text: str):
        """Yield wrappers outside fenced code, treating wrappers as opaque.

        A genuine wrapper that starts first owns its whole body, so Markdown in
        a real argument does not change tool parsing. Conversely, a fence that
        starts first owns all apparent wrappers through its close or EOF.
        """
        pos = 0
        while pos < len(text):
            tool_start = text.find(self.tool_call_start_token, pos)
            fence = self._find_fence_open(text, pos)
            if fence is not None and (tool_start == -1 or fence.start() < tool_start):
                marker = fence.group("marker")
                close = self._find_fence_close(
                    text, fence.end(), marker[0], len(marker)
                )
                pos = close.end() if close is not None else len(text)
                continue
            if tool_start == -1:
                break

            relative_span = next(
                self._iter_structure_spans(
                    text[tool_start:],
                    self.tool_call_start_token,
                    self.tool_call_end_token,
                )
            )
            start, end, body_end = relative_span
            yield (
                tool_start + start,
                tool_start + end,
                tool_start + body_end,
            )
            pos = tool_start + end

    def _advance_stream(self, length: int) -> None:
        consumed = self._buffer[self.parsed_pos : self.parsed_pos + length]
        self.parsed_pos += length
        if consumed:
            self._stream_at_line_start = consumed.endswith(("\r", "\n"))

    @staticmethod
    def _stream_line_end(text: str) -> Optional[int]:
        """Return the end offset of the first LF, CRLF, or bare-CR line."""
        cr = text.find("\r")
        lf = text.find("\n")
        starts = [index for index in (cr, lf) if index != -1]
        if not starts:
            return None
        start = min(starts)
        if text[start : start + 2] == "\r\n":
            return start + 2
        return start + 1

    def _stream_fence_event(self, text: str):
        """Return the next complete opening or an incomplete candidate."""
        search_pos = 0
        while opening := self._find_fence_open(text, search_pos, require_newline=True):
            if opening.start() != 0 or self._stream_at_line_start:
                return "open", opening.start(), opening
            search_pos = opening.end()

        line_start = max(text.rfind("\n"), text.rfind("\r")) + 1
        if line_start == 0 and not self._stream_at_line_start:
            return None
        tail = text[line_start:]
        if re.fullmatch(r" {1,3}", tail):
            return "potential", line_start, None
        candidate = re.match(r" {0,3}(?P<marker>`+|~+)(?P<info>.*)$", tail)
        if candidate is None:
            return None
        marker = candidate.group("marker")
        info = candidate.group("info")
        if len(marker) < 3 and info:
            return None
        if marker[0] == "`" and "`" in info:
            return None
        return "potential", line_start, None

    def _is_stream_fence_close(self, line: str) -> bool:
        content = line.removesuffix("\n").removesuffix("\r")
        return (
            re.fullmatch(
                rf" {{0,3}}{re.escape(self._code_fence_marker or '')}"
                rf"{{{self._code_fence_length},}}[ \t]*",
                content,
            )
            is not None
        )

    def _could_be_stream_fence_close(self, line: str) -> bool:
        """Whether an unfinished line can still become the closing fence."""
        match = re.match(r" {0,3}", line)
        assert match is not None
        pos = match.end()
        if pos == len(line):
            return True
        if line[pos] != self._code_fence_marker:
            return False

        marker_end = pos
        while marker_end < len(line) and line[marker_end] == self._code_fence_marker:
            marker_end += 1
        marker_count = marker_end - pos
        if marker_end == len(line):
            return True
        if marker_count < self._code_fence_length:
            return False
        return all(char in " \t\r" for char in line[marker_end:])

    def _emit_string_parameter(self, calls: List[ToolCallItem]) -> bool:
        """Emit a declared string while retaining possible closing markup."""
        state = self._stream_string
        value = self._buffer[self.parsed_pos :]
        if state["leading"] and value:
            state["leading"] = False
            if value.startswith("\n"):
                self._advance_stream(1)
                value = value[1:]

        delimiters = (
            (self.parameter_end_token, len(self.parameter_end_token)),
            (self.parameter_prefix, 0),
            (self.function_end_token, 0),
        )
        endings = [
            (value.find(token), consume)
            for token, consume in delimiters
            if token in value
        ]
        complete = bool(endings)
        if complete:
            end, consume = min(endings)
            chunk = value[:end]
            if chunk.endswith("\n"):
                chunk = chunk[:-1]
            advance = end + consume
        else:
            # A trailing LF may be stripped on completion; a possible "null"
            # and any split delimiter must also remain undecided until then.
            keep = max(len(token) for token, _ in delimiters) + len("null")
            advance = max(0, len(value) - keep)
            if not advance:
                return False
            chunk = value[:advance]

        fragment = ""
        if not state["opened"]:
            if not self.json_started:
                fragment += "{"
                self.json_started = True
            if self.current_tool_param_count:
                fragment += ", "
            fragment += json.dumps(state["name"]) + ": "
            if complete and chunk.lower() == "null":
                fragment += "null"
            else:
                fragment += '"' + json.dumps(chunk, ensure_ascii=False)[1:-1]
                if complete:
                    fragment += '"'
            state["opened"] = True
        else:
            fragment += json.dumps(chunk, ensure_ascii=False)[1:-1]
            if complete:
                fragment += '"'

        if fragment:
            calls.append(
                ToolCallItem(tool_index=self.current_tool_id, parameters=fragment)
            )
        self._advance_stream(advance)
        if complete:
            self.current_tool_param_count += 1
            self._stream_string = None
        return complete

    def parse_streaming_increment(
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        """Record one parse_error event, then keep the original propagation."""
        try:
            return self._parse_streaming_increment_impl(new_text, tools)
        except Exception:
            record_tool_parser_event(TOOL_PARSER_PARSE_ERROR)
            raise

    def _parse_streaming_increment_impl(
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        """
        Robust cursor-based streaming parser.
        """
        self._buffer += new_text

        # Guard against empty buffer
        if not self._buffer:
            return StreamingParseResult()

        calls = []
        normal_text_chunks = []

        while True:
            if self._stream_string is not None:
                if self._emit_string_parameter(calls):
                    continue
                break

            # Working text slice
            current_slice = self._buffer[self.parsed_pos :]

            # Optimization: If almost empty, wait for more
            if not current_slice:
                break

            if self._code_fence_marker is not None:
                if not self._code_fence_line_can_close:
                    line_end = self._stream_line_end(current_slice)
                    emit_length = len(current_slice) if line_end is None else line_end
                    self._append_normal_text(
                        current_slice[:emit_length], normal_text_chunks
                    )
                    self._advance_stream(emit_length)
                    if line_end is not None:
                        self._code_fence_line_can_close = True
                    continue

                line_end = self._stream_line_end(current_slice)
                if line_end is None:
                    if self._could_be_stream_fence_close(current_slice):
                        # Keep only an ambiguous closing-line tail buffered.
                        break
                    self._code_fence_line_can_close = False
                    continue
                line = current_slice[:line_end]
                closes_fence = self._is_stream_fence_close(line)
                self._append_normal_text(line, normal_text_chunks)
                self._advance_stream(len(line))
                if closes_fence:
                    self._code_fence_marker = None
                    self._code_fence_length = 0
                self._code_fence_line_can_close = True
                continue

            # A real wrapper takes precedence over Markdown-like content in
            # its arguments. Outside wrappers, stop before an opening fence so
            # apparent tool syntax in the block is emitted literally.
            if not self.is_inside_tool_call:
                fence_event = self._stream_fence_event(current_slice)
                if fence_event is not None:
                    event, start, opening = fence_event
                    if start == 0:
                        if event == "potential":
                            break
                        marker = opening.group("marker")
                        opening_line = current_slice[: opening.end()]
                        self._append_normal_text(opening_line, normal_text_chunks)
                        self._advance_stream(len(opening_line))
                        self._code_fence_marker = marker[0]
                        self._code_fence_length = len(marker)
                        self._code_fence_line_can_close = True
                        continue
                    current_slice = current_slice[:start]

            # Rejected bodies are literal until their own function closes.
            # Handle nested tags before callable-tag recognition: otherwise a
            # declared function quoted inside an unknown tool's parameter can
            # reenter the accepted path, or an inner wrapper can reset rejection.
            if self._suppress_current_call:
                literal_tag_len = 0
                if current_slice.startswith(self.tool_call_prefix):
                    end_angle = current_slice.find(">")
                    if end_angle == -1:
                        break
                    self._rejected_function_depth += 1
                    literal_tag_len = end_angle + 1
                elif current_slice.startswith(self.function_end_token):
                    self._rejected_function_depth -= 1
                    if self._rejected_function_depth == 0:
                        self._suppress_current_call = False
                        self._rejected_tool_depth = 0
                    literal_tag_len = len(self.function_end_token)
                elif current_slice.startswith(self.tool_call_start_token):
                    self._rejected_tool_depth += 1
                    literal_tag_len = len(self.tool_call_start_token)
                elif current_slice.startswith(self.tool_call_end_token):
                    if self._rejected_tool_depth:
                        self._rejected_tool_depth -= 1
                        literal_tag_len = len(self.tool_call_end_token)
                    else:
                        # An outer wrapper close also bounds a malformed,
                        # unfinished rejected function; a later wrapper is new.
                        self._suppress_current_call = False
                        self._rejected_function_depth = 0
                if literal_tag_len:
                    self._append_normal_text(
                        current_slice[:literal_tag_len], normal_text_chunks
                    )
                    self._advance_stream(literal_tag_len)
                    continue

            # -------------------------------------------------------
            # 1. Priority detection: check if it's the start of Tool Call
            # -------------------------------------------------------
            if current_slice.startswith(self.tool_call_start_token):
                if self._pending_tool_prefix is not None:
                    self._append_normal_text(
                        self._pending_tool_prefix, normal_text_chunks
                    )
                self._pending_tool_prefix = self.tool_call_start_token
                self._tool_wrapper_prefix = self.tool_call_start_token
                self._suppress_current_call = False
                self._preserve_tool_wrapper = False
                self._advance_stream(len(self.tool_call_start_token))
                self.is_inside_tool_call = True
                continue

            # -------------------------------------------------------
            # 2. Function Name: <function=name>
            # -------------------------------------------------------
            # Bare function/parameter markup in prose is not a call (#38624).
            if self.is_inside_tool_call and current_slice.startswith(
                self.tool_call_prefix
            ):
                end_angle = current_slice.find(">")
                if end_angle != -1:
                    func_name = current_slice[len(self.tool_call_prefix) : end_angle]

                    # A later function can replace a still-open call. Remember
                    # it before the native parser resets that state, but emit
                    # incomplete outcomes only when the response terminates.
                    if self.current_func_name is not None:
                        self._superseded_incomplete_calls += 1

                    if not self._is_declared_tool(func_name, tools):
                        logger.warning(
                            "Model attempted to call undefined function: %s", func_name
                        )
                        record_tool_parser_event(TOOL_PARSER_UNKNOWN_TOOL)
                        self._suppress_current_call = True
                        self._rejected_function_depth = 1
                        self._rejected_tool_depth = 0
                        self.current_func_name = None
                        wrapper_prefix = ""
                        if not self._preserve_tool_wrapper:
                            wrapper_prefix = (
                                self._pending_tool_prefix or self._tool_wrapper_prefix
                            )
                        self._append_normal_text(
                            wrapper_prefix + current_slice[: end_angle + 1],
                            normal_text_chunks,
                        )
                        self._preserve_tool_wrapper = True
                        self._pending_tool_prefix = None
                        self._advance_stream(end_angle + 1)
                        continue

                    if self._pending_tool_prefix is not None:
                        # A later rejected sibling still needs the opening
                        # wrapper and its whitespace, even after a valid call.
                        self._tool_wrapper_prefix = self._pending_tool_prefix
                        self._pending_tool_prefix = None
                    self._pending_tool_separator = None
                    self._suppress_current_call = False

                    self.current_tool_id += 1
                    self.current_tool_name_sent = True
                    self.current_tool_param_count = 0
                    self.json_started = False
                    self.current_func_name = func_name

                    calls.append(
                        ToolCallItem(
                            tool_index=self.current_tool_id,
                            name=func_name,
                            parameters="",
                        )
                    )
                    # One logical call here; the parameter fragments streamed
                    # later for this call are not counted again.
                    record_tool_parser_event(TOOL_PARSER_ACCEPTED)
                    if tools and not self._tool_name_known(func_name, tools):
                        # Undeclared name accepted because unknown-tool
                        # forwarding is on; overlaps accepted by design.
                        record_tool_parser_event(TOOL_PARSER_UNKNOWN_TOOL)

                    self._advance_stream(end_angle + 1)
                    continue
                else:
                    # Incomplete tag
                    break

            # -------------------------------------------------------
            # 3. Parameter: <parameter=name>value...
            # -------------------------------------------------------
            if (
                self.is_inside_tool_call
                and self.current_func_name is not None
                and current_slice.startswith(self.parameter_prefix)
            ):
                name_end = current_slice.find(">")
                if name_end != -1:
                    param_name = current_slice[
                        len(self.parameter_prefix) : name_end
                    ]
                    param_config = self._get_arguments_config(
                        self.current_func_name, tools
                    )
                    schema = param_config.get(param_name)
                    if isinstance(schema, dict) and schema.get("type") == "string":
                        self._stream_string = {
                            "name": param_name,
                            "opened": False,
                            "leading": True,
                        }
                        self._advance_stream(name_end + 1)
                        continue

                    value_start_idx = name_end + 1
                    rest_of_slice = current_slice[value_start_idx:]

                    # A parameter can end in multiple ways:
                    # 1. [Normal] Encounter </parameter>
                    # 2. [Abnormal] Encounter next <parameter=
                    # 3. [Abnormal] Encounter </function>
                    # So we need to find the smallest one as the parameter end position.
                    cand_end_param = rest_of_slice.find(self.parameter_end_token)
                    cand_next_param = rest_of_slice.find(self.parameter_prefix)
                    cand_end_func = rest_of_slice.find(self.function_end_token)

                    candidates = []
                    if cand_end_param != -1:
                        candidates.append(
                            (cand_end_param, len(self.parameter_end_token))
                        )
                    if cand_next_param != -1:
                        candidates.append((cand_next_param, 0))
                    if cand_end_func != -1:
                        candidates.append((cand_end_func, 0))

                    if candidates:
                        best_cand = min(candidates, key=lambda x: x[0])
                        end_pos = best_cand[0]
                        end_token_len = best_cand[1]

                        param_name = current_slice[
                            len(self.parameter_prefix) : name_end
                        ]
                        raw_value = rest_of_slice[:end_pos]

                        # Cleanup value
                        if raw_value.startswith("\n"):
                            raw_value = raw_value[1:]
                        if raw_value.endswith("\n"):
                            raw_value = raw_value[:-1]

                        # JSON Construction
                        if not self.json_started:
                            calls.append(
                                ToolCallItem(
                                    tool_index=self.current_tool_id, parameters="{"
                                )
                            )
                            self.json_started = True

                        param_config = self._get_arguments_config(
                            self.current_func_name, tools
                        )
                        converted_val = self._convert_param_value(
                            raw_value, param_name, param_config, self.current_func_name
                        )

                        # Construct JSON fragment: "key": value
                        # Note: We must be careful with json.dumps to ensure valid JSON streaming
                        json_key_val = f"{json.dumps(param_name)}: {json.dumps(converted_val, ensure_ascii=False)}"

                        if self.current_tool_param_count > 0:
                            fragment = f", {json_key_val}"
                        else:
                            fragment = json_key_val

                        calls.append(
                            ToolCallItem(
                                tool_index=self.current_tool_id, parameters=fragment
                            )
                        )
                        self.current_tool_param_count += 1

                        # Advance cursor
                        total_len = (name_end + 1) + end_pos + end_token_len
                        self._advance_stream(total_len)
                        continue

                # Incomplete parameter tag or value
                break

            # -------------------------------------------------------
            # 4. Function End: </function>
            # -------------------------------------------------------
            if self.is_inside_tool_call and current_slice.startswith(
                self.function_end_token
            ):
                if self.current_func_name is None:
                    self._append_unparsed_text(
                        self.function_end_token, normal_text_chunks
                    )
                    self._suppress_current_call = False
                    self._advance_stream(len(self.function_end_token))
                    continue
                if not self.json_started:
                    calls.append(
                        ToolCallItem(tool_index=self.current_tool_id, parameters="{")
                    )
                    self.json_started = True

                calls.append(
                    ToolCallItem(tool_index=self.current_tool_id, parameters="}")
                )
                self._advance_stream(len(self.function_end_token))
                self.current_func_name = None
                continue

            # -------------------------------------------------------
            # 5. Tool Call End: </tool_call>
            # -------------------------------------------------------
            if self.is_inside_tool_call and current_slice.startswith(
                self.tool_call_end_token
            ):
                if self._pending_tool_prefix is not None:
                    self._append_normal_text(
                        self._pending_tool_prefix + self.tool_call_end_token,
                        normal_text_chunks,
                    )
                    self._pending_tool_prefix = None
                elif self._preserve_tool_wrapper:
                    self._append_normal_text(
                        self.tool_call_end_token, normal_text_chunks
                    )
                else:
                    self._pending_tool_separator = ""
                self._advance_stream(len(self.tool_call_end_token))
                self.is_inside_tool_call = False  # [FIX] Exit tool call region
                self._suppress_current_call = False
                self._preserve_tool_wrapper = False
                continue

            # -------------------------------------------------------
            # 6. Handling content / whitespace / normal text
            # -------------------------------------------------------
            # If current position is not the start of a tag (i.e., doesn't start with <), it might be plain text,
            # or a newline between two tags.
            # But we need to be careful not to output truncated tags like "<fun" as text.

            next_open_angle = current_slice.find("<")

            if next_open_angle == -1:
                # This entire segment is plain text
                self._append_unparsed_text(current_slice, normal_text_chunks)
                # [FIX] If inside tool call, discard this text (usually \n), don't append
                self._advance_stream(len(current_slice))
                continue

            elif next_open_angle == 0:
                # Looks like a Tag, but doesn't match any known Tag above

                possible_tags = [
                    self.tool_call_start_token,
                    self.tool_call_end_token,
                    self.tool_call_prefix,
                    self.function_end_token,
                    self.parameter_prefix,
                    self.parameter_end_token,
                ]

                is_potential_tag = False
                for tag in possible_tags:
                    if tag.startswith(current_slice):
                        is_potential_tag = True
                        break

                if is_potential_tag:
                    break  # Wait for more
                else:
                    # Just a plain '<' symbol
                    self._append_unparsed_text("<", normal_text_chunks)
                    self._advance_stream(1)
                    continue

            else:
                # '<' is in the middle
                text_segment = current_slice[:next_open_angle]
                self._append_unparsed_text(text_segment, normal_text_chunks)
                # [FIX] If inside tool call, discard whitespace/text before Tag
                self._advance_stream(next_open_angle)
                continue

        # Memory Cleanup: Slice the buffer
        # Keep unparsed part, discard parsed part
        if self.parsed_pos > 0:
            self._buffer = self._buffer[self.parsed_pos :]
            self.parsed_pos = 0

        normal_text = "".join(normal_text_chunks) if normal_text_chunks else ""
        return StreamingParseResult(calls=calls, normal_text=normal_text)

    def _append_unparsed_text(self, text: str, chunks: List[str]) -> None:
        if self._pending_tool_prefix is not None:
            self._pending_tool_prefix += text
        elif (
            not self.is_inside_tool_call
            or self._suppress_current_call
            or (self._preserve_tool_wrapper and self.current_func_name is None)
        ):
            self._append_normal_text(text, chunks)
        elif self.current_func_name is None:
            self._tool_wrapper_prefix += text

    def finish(self, tools: List[Tool]) -> StreamingParseResult:
        # Only flush unrecognized syntax. An unfinished accepted call must not
        # silently become prose or have fabricated argument-closing delimiters.
        # Response termination is the only place an incomplete call is counted,
        # and only for a call attempt: a bare wrapper with no function is
        # returned as prose and is not an attempt, a partially received opening
        # marker is still ambiguous markup, and a rejected unknown call was
        # already counted as unknown_tool.
        if not self._incomplete_recorded:
            self._incomplete_recorded = True
            incomplete_calls = self._superseded_incomplete_calls + int(
                self.current_func_name is not None
            )
            for _ in range(incomplete_calls):
                record_tool_parser_event(TOOL_PARSER_INCOMPLETE)
        chunks = []
        if self._code_fence_marker is not None:
            self._append_normal_text(self._buffer, chunks)
            self._code_fence_marker = None
            self._code_fence_length = 0
            self._code_fence_line_can_close = True
            self._buffer = ""
        elif self.current_func_name is None:
            self._append_normal_text(
                (self._pending_tool_prefix or "") + self._buffer, chunks
            )
            self._pending_tool_prefix = None
            self._buffer = ""
        return StreamingParseResult(normal_text="".join(chunks))

    def _append_normal_text(self, text: str, chunks: List[str]) -> None:
        if self._pending_tool_separator is not None:
            self._pending_tool_separator += text
            if not self._pending_tool_separator.strip():
                return
            # Preserve genuine prose, including spaces split into their own
            # increments. Filtering every whitespace-only result loses them.
            text = self._pending_tool_separator
            self._pending_tool_separator = None
        chunks.append(text)

    def supports_structural_tag(self) -> bool:
        return True

    def structure_info(self) -> _GetInfoFunc:
        raise NotImplementedError

    def get_structural_tag_name(self) -> str:
        return "qwen_3_coder"
