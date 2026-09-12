"""CUDA graph replay integration for host-staged PLE adapters."""

from __future__ import annotations

_REPLAY_ACTIVE_ATTR = "_ssd_stream_execute_active"
_REPLAY_HOOKS_ATTR = "_ssd_stream_replay_hooks"


def around_execute(original_fn, runner, *args, **kwargs):
    """Mark the narrow region in which load_batch precedes graph replay."""
    setattr(runner, _REPLAY_ACTIVE_ATTR, True)
    try:
        return original_fn(runner, *args, **kwargs)
    finally:
        setattr(runner, _REPLAY_ACTIVE_ATTR, False)


def after_load_batch(result, runner, forward_batch, *args, **kwargs):
    """Stage dynamic PLE rows after graph inputs are loaded and before replay."""
    if not getattr(runner, _REPLAY_ACTIVE_ATTR, False):
        return result

    hooks = getattr(runner, _REPLAY_HOOKS_ATTR, None)
    if hooks is None:
        hooks = tuple(
            hook
            for module in runner.model_runner.model.modules()
            if callable(
                hook := getattr(module, "prepare_ssd_stream_graph_replay", None)
            )
        )
        setattr(runner, _REPLAY_HOOKS_ATTR, hooks)

    if not hooks:
        return result

    graph_physical_tokens = (
        runner._replay_graph_key.size
        if runner.ragged_verify_mode
        else runner.bs * runner.captured_req_width
    )
    for hook in hooks:
        hook(forward_batch.input_ids, forward_batch, graph_physical_tokens)
    return result
