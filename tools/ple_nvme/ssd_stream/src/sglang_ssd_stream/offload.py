"""Qwen expert selection for SGLang's grouped CPU offloader."""

from __future__ import annotations


def _experts(layer):
    return layer.mlp.experts


def _expert_weight_names(module) -> list[str]:
    return [
        name
        for name in (
            "w13_weight",
            "w2_weight",
            "w13_blockscale_swizzled",
            "w2_blockscale_swizzled",
        )
        if hasattr(module, name)
    ]


def around_make_layers(original_fn, *args, **kwargs):
    from .config import grouped_cpu_offload_enabled

    if grouped_cpu_offload_enabled():
        if kwargs.get("offloader_kwargs") is not None:
            raise ValueError("Qwen grouped CPU offload was configured twice")
        kwargs["offloader_kwargs"] = {
            "submodule_accessor": _experts,
            "whitelist_param_names_creator": _expert_weight_names,
        }
    return original_fn(*args, **kwargs)
