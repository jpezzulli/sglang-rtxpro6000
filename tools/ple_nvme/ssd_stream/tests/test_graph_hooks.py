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
