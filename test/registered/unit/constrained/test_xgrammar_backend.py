from unittest.mock import Mock, call

import pytest

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.constrained.xgrammar_backend import XGrammarGrammar
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _make_grammar():
    matcher = Mock()
    matcher.is_terminated.return_value = False
    matcher.accept_token.return_value = True
    grammar = XGrammarGrammar(
        matcher=matcher,
        vocab_size=128,
        ctx=Mock(),
        override_stop_tokens=None,
        key_string="test-grammar",
    )
    return grammar, matcher


def _accept(grammar, tokens):
    for token in tokens:
        grammar.accept_token(token)


def test_repeated_speculative_rollback_updates_accepted_tokens_in_place():
    grammar, matcher = _make_grammar()
    _accept(grammar, [10, 11])
    accepted_tokens = grammar.accepted_tokens

    _accept(grammar, [20, 21, 22])
    grammar.rollback(3)
    assert grammar.accepted_tokens is accepted_tokens
    assert accepted_tokens == [10, 11]

    _accept(grammar, [30, 31])
    grammar.rollback(1)
    _accept(grammar, [40])
    assert grammar.accepted_tokens is accepted_tokens
    assert accepted_tokens == [10, 11, 30, 40]
    assert matcher.rollback.call_args_list == [call(3), call(1)]


@pytest.mark.parametrize("rollback_count", [0, 3, 8])
def test_rollback_boundary_clears_the_same_accepted_tokens_list(rollback_count):
    grammar, matcher = _make_grammar()
    _accept(grammar, [1, 2, 3])
    accepted_tokens = grammar.accepted_tokens

    grammar.rollback(rollback_count)

    matcher.rollback.assert_called_once_with(rollback_count)
    assert grammar.accepted_tokens is accepted_tokens
    assert accepted_tokens == []


def test_rollback_error_leaves_debug_tokens_unchanged():
    grammar, matcher = _make_grammar()
    _accept(grammar, [4, 5, 6])
    accepted_tokens = grammar.accepted_tokens
    matcher.rollback.side_effect = RuntimeError("rollback rejected")

    with pytest.raises(RuntimeError, match="rollback rejected"):
        grammar.rollback(2)

    assert grammar.accepted_tokens is accepted_tokens
    assert accepted_tokens == [4, 5, 6]


def test_grammar_instances_keep_independent_accepted_token_lists():
    first, first_matcher = _make_grammar()
    second, second_matcher = _make_grammar()
    _accept(first, [7, 8, 9])
    _accept(second, [70, 80])

    first.rollback(2)

    assert first.accepted_tokens == [7]
    assert second.accepted_tokens == [70, 80]
    assert first.accepted_tokens is not second.accepted_tokens
    first_matcher.rollback.assert_called_once_with(2)
    second_matcher.rollback.assert_not_called()


def test_rejected_token_does_not_enter_debug_history():
    grammar, matcher = _make_grammar()
    _accept(grammar, [12])
    matcher.accept_token.return_value = False

    with pytest.raises(ValueError, match="Tokens not accepted: 13"):
        grammar.accept_token(13)

    assert grammar.accepted_tokens == [12]
