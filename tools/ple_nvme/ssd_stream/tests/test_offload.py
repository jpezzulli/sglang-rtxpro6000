from sglang_ssd_stream import config
from sglang_ssd_stream.offload import around_make_layers


class Experts:
    w13_weight = object()
    w2_weight = object()
    w13_blockscale_swizzled = object()


class Layer:
    class MLP:
        experts = Experts()

    mlp = MLP()


def test_qwen_grouped_offload_selects_only_runtime_expert_weights(monkeypatch):
    captured = {}

    def original(*args, **kwargs):
        captured.update(kwargs)
        return args

    monkeypatch.setattr(config, "grouped_cpu_offload_enabled", lambda: True)
    around_make_layers(original, 48, object())
    options = captured["offloader_kwargs"]
    experts = options["submodule_accessor"](Layer())

    assert experts is Layer.MLP.experts
    assert options["whitelist_param_names_creator"](experts) == [
        "w13_weight",
        "w2_weight",
        "w13_blockscale_swizzled",
    ]
