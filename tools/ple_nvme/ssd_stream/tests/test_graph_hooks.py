from types import SimpleNamespace

from sglang_ssd_stream.graph import after_load_batch, around_execute


def test_graph_hook_runs_between_load_and_replay():
    calls = []

    class Module:
        def prepare_ssd_stream_graph_replay(self, input_ids, forward_batch, tokens):
            calls.append((input_ids, forward_batch, tokens))

    runner = SimpleNamespace(
        model_runner=SimpleNamespace(model=SimpleNamespace(modules=lambda: [Module()])),
        _replay_graph_key=SimpleNamespace(size=8),
        ragged_verify_mode=False,
        bs=2,
        captured_req_width=4,
    )
    batch = SimpleNamespace(input_ids="ids")

    def replay(self):
        after_load_batch(None, self, batch)
        return "ok"

    assert around_execute(replay, runner) == "ok"
    assert calls == [("ids", batch, 8)]


def test_fixed_window_replay_uses_captured_width_and_stays_inert_outside_execute():
    calls = []

    class Module:
        def prepare_ssd_stream_graph_replay(self, input_ids, forward_batch, tokens):
            calls.append(tokens)

    def runner(ragged_verify_mode, replay_graph_key_size):
        return SimpleNamespace(
            model_runner=SimpleNamespace(
                model=SimpleNamespace(modules=lambda: [Module()])
            ),
            _replay_graph_key=SimpleNamespace(size=replay_graph_key_size),
            ragged_verify_mode=ragged_verify_mode,
            bs=2,
            captured_req_width=4,
        )

    batch = SimpleNamespace(input_ids="ids")

    def replay(self):
        after_load_batch(None, self, batch)
        return "ok"

    # A load_batch outside the execute window (capture, an unrelated caller)
    # must not stage rows for a replay that is not happening.
    after_load_batch(None, runner(False, 999), batch)
    assert calls == []

    # Fixed speculation window: bs * captured_req_width rows, never the ragged key.
    assert around_execute(replay, runner(False, 999)) == "ok"
    assert calls == [8]
    calls.clear()
    # Ragged verify keeps the replay graph key as the row budget.
    assert around_execute(replay, runner(True, 12)) == "ok"
    assert calls == [12]
