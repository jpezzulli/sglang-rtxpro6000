"""Build a deterministic draft vocabulary from non-evaluation text sources."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def select_ids(counts, valid_ids, special_ids, size, base_count):
    valid = set(valid_ids)
    special = set(special_ids)
    if not special <= valid:
        raise ValueError("special IDs must belong to the model vocabulary")
    if not 0 <= base_count <= size <= len(valid):
        raise ValueError("invalid draft vocabulary size")
    chosen = special | set(sorted(valid)[:base_count])
    if len(chosen) > size:
        raise ValueError("mandatory IDs exceed draft vocabulary size")
    for token_id in sorted(counts, key=lambda token_id: (-counts[token_id], token_id)):
        if len(chosen) == size:
            break
        if token_id in valid:
            chosen.add(token_id)
    for token_id in sorted(valid):
        if len(chosen) == size:
            break
        chosen.add(token_id)
    return sorted(chosen)


def main():
    import torch
    from tokenizers import Tokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--corpus-root", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, default=65536)
    parser.add_argument("--base-count", type=int, default=32768)
    parser.add_argument("--max-bytes", type=int, default=16 * 1024 * 1024)
    parser.add_argument("--per-file-bytes", type=int, default=64 * 1024)
    args = parser.parse_args()
    manifest_path = args.output.with_suffix(".manifest.json")
    if args.output.exists() or manifest_path.exists():
        raise FileExistsError("use a fresh map output path")
    config = json.loads((args.model / "config.json").read_text())
    vocab_size = config.get("text_config", config)["vocab_size"]
    tokenizer_path = args.model / "tokenizer.json"
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    tokenizer_data = json.loads(tokenizer_path.read_text())
    valid = {i for i in tokenizer.get_vocab().values() if 0 <= i < vocab_size}
    special = {
        entry["id"] for entry in tokenizer_data["added_tokens"] if entry["special"]
    }
    excluded = {"test", "tests", "validation", "results", "benchmarks", ".git"}
    paths = sorted(
        {
            path.resolve()
            for root in args.corpus_root
            for path in root.rglob("*")
            if path.is_file()
            and not path.is_symlink()
            and path.suffix in {".py", ".md", ".json", ".sh"}
            and not excluded.intersection(path.parts)
        }
    )
    counts, corpus, total_bytes = Counter(), [], 0
    for path in paths:
        if total_bytes >= args.max_bytes:
            break
        with path.open("rb") as stream:
            raw = stream.read(min(args.per_file_bytes, args.max_bytes - total_bytes))
        if not raw:
            continue
        counts.update(
            tokenizer.encode(
                raw.decode("utf-8", errors="ignore"), add_special_tokens=False
            ).ids
        )
        total_bytes += len(raw)
        corpus.append(
            {
                "path": str(path),
                "bytes_used": len(raw),
                "sha256_used": hashlib.sha256(raw).hexdigest(),
            }
        )
    if not counts:
        raise ValueError("empty corpus")
    selected = select_ids(counts, valid, special, args.size, args.base_count)
    assert len(selected) == args.size and len(set(selected)) == args.size
    assert special <= set(selected)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(selected, args.output)
    loaded = torch.load(args.output, weights_only=True)
    assert loaded == selected
    manifest = {
        "model": str(args.model.resolve()),
        "model_vocab_size": vocab_size,
        "tokenizer_vocab_size": len(valid),
        "tokenizer_sha256": hashlib.sha256(tokenizer_path.read_bytes()).hexdigest(),
        "size": len(selected),
        "base_count": args.base_count,
        "special_ids": sorted(special),
        "all_special_ids_included": True,
        "ids_sha256": hashlib.sha256(
            json.dumps(selected, separators=(",", ":")).encode()
        ).hexdigest(),
        "map_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "corpus_bytes": total_bytes,
        "corpus_tokens": sum(counts.values()),
        "corpus_coverage": sum(counts[i] for i in selected) / sum(counts.values()),
        "corpus": corpus,
        "evaluation_cases_used": False,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {key: value for key, value in manifest.items() if key != "corpus"}, indent=2
        )
    )


if __name__ == "__main__":
    main()
