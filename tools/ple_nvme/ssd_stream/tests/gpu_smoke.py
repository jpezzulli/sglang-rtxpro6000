"""GPU smoke tests for the Qwen4 SSD Stream implementation."""

import hashlib
import json
import tempfile
from pathlib import Path

import torch
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
from sglang_ssd_stream.backend import SSDStreamEmbedding
from sglang_ssd_stream.config import load_manifest


def _embedding(dtype: torch.dtype, rows: int = 1024, width: int = 160):
    embedding = VocabParallelEmbedding(
        rows,
        width,
        params_dtype=dtype,
        org_num_embeddings=rows,
        enable_tp=False,
    )
    embedding.register_buffer("weight_scale", torch.ones((), dtype=torch.float32))
    return embedding


def _meta_embedding(dtype: torch.dtype, rows: int = 1024, width: int = 160):
    with torch.device("meta"):
        embedding = VocabParallelEmbedding(
            rows,
            width,
            params_dtype=dtype,
            org_num_embeddings=rows,
            enable_tp=False,
        )
    embedding.register_buffer("weight_scale", torch.ones((), dtype=torch.float32))
    return embedding


def _source(dtype: torch.dtype, rows: int = 1024, width: int = 160):
    if dtype == torch.float8_e4m3fn:
        raw = (
            torch.arange(rows * width, dtype=torch.int64)
            .remainder(63)
            .to(torch.uint8)
            .reshape(rows, width)
        )
        return raw.view(torch.float8_e4m3fn)
    return (
        torch.arange(rows * width, dtype=torch.float32)
        .remainder(257)
        .reshape(rows, width)
        .to(dtype)
    )


def _run_case(dtype: torch.dtype) -> None:
    source = _source(dtype)
    with tempfile.TemporaryDirectory() as backing_dir:
        root = Path(backing_dir)
        table = root / "ple" / "table.bin"
        table.parent.mkdir()
        table.write_bytes(source.view(torch.uint8).numpy().tobytes())
        digest = hashlib.sha256(table.read_bytes()).hexdigest()
        manifest = root / "ssd-stream.json"
        manifest.write_text(
            json.dumps(
                {
                    "format": "sglang-ssd-stream",
                    "version": 1,
                    "tables": [
                        {
                            "layer": 0,
                            "path": "ple/table.bin",
                            "sha256": digest,
                            "dtype": str(dtype).removeprefix("torch."),
                            "rows": source.shape[0],
                            "columns": source.shape[1],
                            "row_start": 0,
                            "bytes": table.stat().st_size,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        config = load_manifest(manifest)
        staged = SSDStreamEmbedding(_embedding(dtype), config=config, ple_layer_index=0)
        assert not list(staged.parameters())
        assert not hasattr(staged, "weight")

        meta_staged = SSDStreamEmbedding(
            _meta_embedding(dtype), config=config, ple_layer_index=0
        )
        assert not list(meta_staged.parameters())

        ids = torch.tensor([0, 127, 0, 511, 900], device="cuda")
        expected = source[ids.cpu()].to(torch.bfloat16)
        output = staged.allocate_output((*ids.shape, source.shape[1]), ids.device)
        prefetch_stream = torch.cuda.Stream()
        ticket = staged.begin_gather(
            ids,
            out=output,
            stream=prefetch_stream,
            producer_stream=torch.cuda.current_stream(),
        )
        ticket.wait_for_launch()
        torch.cuda.current_stream().wait_stream(prefetch_stream)
        torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
        if dtype == torch.float8_e4m3fn:
            # A decode graph can wait on an event recorded by SSD staging
            # immediately before replay, while reading a static output buffer.
            graph_ids = torch.tensor([9, 81, 512, 9], device="cuda")
            graph_expected = source[graph_ids.cpu()].to(torch.bfloat16)
            graph_output = staged.allocate_output(
                (*graph_ids.shape, source.shape[1]), graph_ids.device
            )
            graph_result = torch.empty_like(graph_output)
            graph_ready = torch.cuda.Event(external=True)
            graph_ready.record(torch.cuda.current_stream())
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                torch.cuda.current_stream().wait_event(graph_ready)
                graph_result.copy_(graph_output)

            graph_ticket = staged.begin_gather(
                graph_ids,
                out=graph_output,
                stream=prefetch_stream,
                producer_stream=torch.cuda.current_stream(),
                completion_event=graph_ready,
            )
            graph_ticket.wait_for_launch()
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(
                graph_result.cpu(), graph_expected, rtol=0, atol=0
            )

        invalid = torch.tensor([source.shape[0] + 10], device="cuda")
        assert torch.count_nonzero(staged.gather(invalid)).item() == 0
        assert hashlib.sha256(table.read_bytes()).hexdigest() == digest

        reused = SSDStreamEmbedding(_embedding(dtype), config=config, ple_layer_index=0)
        torch.testing.assert_close(reused.gather(ids).cpu(), expected, rtol=0, atol=0)
        torch.testing.assert_close(
            meta_staged.gather(ids).cpu(), expected, rtol=0, atol=0
        )


if __name__ == "__main__":
    _run_case(torch.float8_e4m3fn)
    _run_case(torch.bfloat16)
    print("SSD Stream tests passed")
