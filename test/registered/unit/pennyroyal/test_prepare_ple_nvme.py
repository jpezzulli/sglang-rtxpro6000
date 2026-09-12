import hashlib
import importlib.util
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).parents[4] / "scripts" / "pennyroyal" / "prepare_ple_nvme.py"
SPEC = importlib.util.spec_from_file_location("prepare_ple_nvme", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

CHECK_SCRIPT = Path(__file__).parents[4] / "scripts" / "pennyroyal" / "check_ple_nvme.py"
CHECK_SPEC = importlib.util.spec_from_file_location("check_ple_nvme", CHECK_SCRIPT)
CHECK = importlib.util.module_from_spec(CHECK_SPEC)
assert CHECK_SPEC.loader is not None
sys.modules[CHECK_SPEC.name] = CHECK
CHECK_SPEC.loader.exec_module(CHECK)


def write_safetensors(path: Path, tensors: list[tuple[str, str, list[int], bytes]]):
    header = {}
    payload = bytearray()
    for name, dtype, shape, data in tensors:
        start = len(payload)
        payload.extend(data)
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [start, len(payload)],
        }
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * ((-len(encoded)) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)


def read_safetensors(path: Path):
    data = path.read_bytes()
    header_size = struct.unpack("<Q", data[:8])[0]
    header = json.loads(data[8 : 8 + header_size])
    payload = data[8 + header_size :]
    values = {}
    for name, metadata in header.items():
        start, end = metadata["data_offsets"]
        values[name] = payload[start:end]
    return header, values


class PreparePLENVMETest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def checkpoint(self, layer_ids=(2,), split_parts=1):
        source = self.root / f"source-{len(list(self.root.iterdir()))}"
        source.mkdir()
        (source / "config.json").write_text(
            json.dumps(
                {
                    "text_config": {
                        "ple_layer_ids": list(layer_ids),
                        "split_ngram_parts": split_parts,
                    }
                }
            )
        )
        (source / "tokenizer.json").write_text('{"unchanged":true}')
        (source / "tokenizer_config.json").write_text('{"unchanged":true}')
        return source

    def write_index(self, source: Path, weight_map: dict[str, str], total_size: int):
        (source / MODULE.INDEX_NAME).write_text(
            json.dumps(
                {"metadata": {"total_size": total_size}, "weight_map": weight_map}
            )
        )

    def integrity_checkpoint(self):
        source = self.checkpoint()
        ple = (
            "model.language_model.layers.1.ple.ple_embedding."
            "ngram_embedding.shard_0.weight"
        )
        retained = "model.language_model.layers.1.ple.ple_embedding.weight_scale"
        ordinary = "model.language_model.layers.1.input_layernorm.weight"
        write_safetensors(
            source / "mixed.safetensors",
            [
                (ple, "F8_E4M3", [2, 1], b"\x01\x02"),
                (retained, "BF16", [1], b"\x03\x04"),
            ],
        )
        write_safetensors(
            source / "ordinary.safetensors",
            [(ordinary, "F32", [1], b"\x05\x06\x07\x08")],
        )
        self.write_index(
            source,
            {
                ple: "mixed.safetensors",
                retained: "mixed.safetensors",
                ordinary: "ordinary.safetensors",
            },
            8,
        )
        output = self.root / "prepared"
        MODULE._prepare(source, output)
        return source, output, ordinary, retained

    def check(self, source: Path, output: Path):
        return CHECK.check(source, output)

    def test_preflight_accepts_exact_preparer_output(self):
        source, output, _, _ = self.integrity_checkpoint()

        identity = self.check(source, output)

        self.assertEqual(len(identity), 64)

    def test_preflight_rejects_ordinary_weight_map_omission(self):
        source, output, ordinary, _ = self.integrity_checkpoint()
        index_path = output / MODULE.INDEX_NAME
        index = json.loads(index_path.read_text())
        index["weight_map"].pop(ordinary)
        index_path.write_text(json.dumps(index))

        with self.assertRaisesRegex(ValueError, "weight_map"):
            self.check(source, output)

    def test_preflight_rejects_ordinary_shard_relink(self):
        source, output, _, _ = self.integrity_checkpoint()
        shard = output / "ordinary.safetensors"
        decoy = self.root / "ordinary-copy.safetensors"
        decoy.write_bytes((source / "ordinary.safetensors").read_bytes())
        shard.unlink()
        shard.symlink_to(decoy)

        with self.assertRaisesRegex(ValueError, "resolve to source"):
            self.check(source, output)

    def test_preflight_rejects_retained_mixed_tensor_tamper(self):
        source, output, _, _ = self.integrity_checkpoint()
        mixed = output / "mixed.safetensors"
        data = bytearray(mixed.read_bytes())
        data[-1] ^= 0xFF
        mixed.write_bytes(data)

        with self.assertRaisesRegex(ValueError, "retained tensor"):
            self.check(source, output)

    def test_numeric_shard_order_and_runtime_layer_enumeration(self):
        source = self.checkpoint(layer_ids=(5, 2), split_parts=11)
        tensors = []
        weight_map = {}
        expected = bytearray()
        # Deliberately put shard 10 before shard 2 in the source header/index.
        for shard in (0, 1, 10, 2, 3, 4, 5, 6, 7, 8, 9):
            name = (
                "model.language_model.layers.1.ple.ple_embedding."
                f"ngram_embedding.shard_{shard}.weight"
            )
            tensors.append((name, "BF16", [1, 1], bytes((shard, shard))))
            weight_map[name] = "ple.safetensors"
        for shard in range(11):
            expected.extend((shard, shard))
        write_safetensors(source / "ple.safetensors", tensors)
        # The second configured PLE layer gets its own complete table.
        second = []
        for shard in range(11):
            name = (
                "model.language_model.layers.4.ple.ple_embedding."
                f"ngram_embedding.shard_{shard}.weight"
            )
            second.append((name, "BF16", [1, 1], bytes((100 + shard, 100 + shard))))
            weight_map[name] = "ple-second.safetensors"
        write_safetensors(source / "ple-second.safetensors", second)
        self.write_index(source, weight_map, 44)

        output = self.root / "prepared"
        MODULE._prepare(source, output)

        manifest = json.loads((output / MODULE.MANIFEST_NAME).read_text())
        self.assertEqual([table["layer"] for table in manifest["tables"]], [0, 1])
        self.assertEqual(manifest["tables"][0]["dtype"], "bfloat16")
        self.assertEqual((output / "ple/layer-0.bin").read_bytes(), bytes(expected))
        self.assertEqual(
            manifest["tables"][0]["sha256"], hashlib.sha256(expected).hexdigest()
        )
        self.assertEqual(
            (output / "ple/layer-1.bin").read_bytes(),
            b"".join(bytes((100 + shard, 100 + shard)) for shard in range(11)),
        )
        self.assertFalse((output / "ple.safetensors").exists())
        self.assertEqual(
            json.loads((output / MODULE.INDEX_NAME).read_text())["metadata"][
                "total_size"
            ],
            0,
        )

    def test_mixed_file_preserves_scale_and_non_ple_payload(self):
        source = self.checkpoint()
        shard = (
            "model.language_model.layers.1.ple.ple_embedding."
            "ngram_embedding.shard_0.weight"
        )
        scale = (
            "model.language_model.layers.1.ple.ple_embedding."
            "ngram_embedding.weight_scale"
        )
        ordinary = "model.language_model.layers.1.input_layernorm.weight"
        write_safetensors(
            source / "mixed.safetensors",
            [
                (shard, "F8_E4M3", [2, 1], b"\x07\x08"),
                (scale, "BF16", [1], b"\x09\x0a"),
                (ordinary, "F32", [1], b"\x0b\x0c\x0d\x0e"),
            ],
        )
        self.write_index(
            source,
            {name: "mixed.safetensors" for name in (shard, scale, ordinary)},
            8,
        )
        before = {path.name: path.read_bytes() for path in source.iterdir()}

        output = self.root / "prepared"
        MODULE._prepare(source, output)

        header, values = read_safetensors(output / "mixed.safetensors")
        self.assertNotIn(shard, header)
        self.assertEqual(values, {scale: b"\x09\x0a", ordinary: b"\x0b\x0c\x0d\x0e"})
        index = json.loads((output / MODULE.INDEX_NAME).read_text())
        self.assertEqual(set(index["weight_map"]), {scale, ordinary})
        manifest = json.loads((output / MODULE.MANIFEST_NAME).read_text())
        self.assertEqual(manifest["tables"][0]["dtype"], "float8_e4m3fn")
        self.assertEqual((output / "ple/layer-0.bin").read_bytes(), b"\x07\x08")
        self.assertTrue((output / "tokenizer.json").is_symlink())
        self.assertEqual(
            before, {path.name: path.read_bytes() for path in source.iterdir()}
        )

    def test_missing_unsupported_and_wrong_layer_fail_without_output(self):
        cases = (
            ("missing", 2, "BF16", 2),
            ("dtype", 1, "F16", 2),
            ("layer", 1, "BF16", 3),
        )
        for label, split_parts, dtype, model_layer in cases:
            with self.subTest(label=label):
                source = self.checkpoint(split_parts=split_parts)
                name = (
                    f"model.language_model.layers.{model_layer - 1}.ple.ple_embedding."
                    "ngram_embedding.shard_0.weight"
                )
                write_safetensors(
                    source / "ple.safetensors", [(name, dtype, [1, 1], b"\0\0")]
                )
                self.write_index(source, {name: "ple.safetensors"}, 2)
                output = self.root / f"prepared-{label}"
                with self.assertRaises(ValueError):
                    MODULE._prepare(source, output)
                self.assertFalse(output.exists())

    def test_existing_output_and_corrupt_header_fail_closed(self):
        source = self.checkpoint()
        name = (
            "model.language_model.layers.1.ple.ple_embedding."
            "ngram_embedding.shard_0.weight"
        )
        (source / "ple.safetensors").write_bytes(struct.pack("<Q", 999) + b"{}")
        self.write_index(source, {name: "ple.safetensors"}, 2)
        output = self.root / "prepared"
        with self.assertRaises(ValueError):
            MODULE._prepare(source, output)
        self.assertFalse(output.exists())
        output.mkdir()
        with self.assertRaises(FileExistsError):
            MODULE._prepare(source, output)

    def test_output_parent_alias_cannot_write_inside_source(self):
        source = self.checkpoint()
        name = (
            "model.language_model.layers.1.ple.ple_embedding."
            "ngram_embedding.shard_0.weight"
        )
        write_safetensors(source / "ple.safetensors", [(name, "BF16", [1, 1], b"\0\0")])
        self.write_index(source, {name: "ple.safetensors"}, 2)
        alias = self.root / "alias"
        alias.symlink_to(source, target_is_directory=True)
        before = set(source.iterdir())
        with self.assertRaisesRegex(ValueError, "inside the source"):
            MODULE._prepare(source, alias / "prepared")
        self.assertEqual(set(source.iterdir()), before)


if __name__ == "__main__":
    unittest.main()
