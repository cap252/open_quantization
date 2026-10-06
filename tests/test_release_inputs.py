"""Public input regressions, run against the installed release wheel in CI."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import yaml

from opennpu_quant._io import sha256
from opennpu_quant.fetch import fetch_assets
from opennpu_quant.paths import Paths


class ReleaseInputTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)

    def write_local(self, value):
        path = self.root / "local.yaml"
        path.write_text(yaml.safe_dump(value), encoding="utf-8")
        return path

    def invoke(self, *args):
        return subprocess.run(
            [sys.executable, "-m", "opennpu_quant", *map(str, args)],
            cwd=self.root,
            capture_output=True,
            text=True,
        )

    def assert_input_error(self, result, *words):
        self.assertEqual(2, result.returncode, result.stdout + result.stderr)
        self.assertNotIn("Traceback", result.stderr)
        for word in words:
            self.assertIn(word, result.stderr)

    def test_fetch_yaml_error_names_file_and_location_before_writing(self):
        manifest = self.root / "broken.yaml"
        manifest.write_text("models: [broken\n", encoding="utf-8")
        result = self.invoke(
            "fetch", "models", "--manifest", manifest, "--models", "resnet18"
        )
        self.assert_input_error(result, str(manifest), "YAML", "line", "column")
        self.assertFalse((self.root / "workspace").exists())

    def test_valid_manifest_still_fetches_and_verifies_local_asset(self):
        archive = self.root / "asset.zip"
        with zipfile.ZipFile(archive, "w") as output:
            output.writestr("payload.txt", "verified asset")
        manifest = self.root / "assets.yaml"
        manifest.write_text(
            yaml.safe_dump(
                {
                    "models": {
                        "toy": {"url": archive.as_uri(), "sha256": sha256(archive)}
                    }
                }
            ),
            encoding="utf-8",
        )
        paths = Paths.resolve(home=self.root / "workspace")
        result = fetch_assets("models", manifest, ["toy"], paths)
        self.assertEqual([str(paths.models / "toy")], result)
        self.assertEqual(
            "verified asset", (paths.models / "toy/payload.txt").read_text()
        )

    def test_local_path_values_require_nonempty_strings(self):
        for key in ("home", "models", "data", "cache", "runs", "weights"):
            for value in (None, False, 1, [], {}, "", "  ", "bad\x00path"):
                with self.subTest(key=key, value=value):
                    path = self.write_local({key: value})
                    with self.assertRaisesRegex(ValueError, key):
                        Paths.resolve(local=path)

    def test_dataset_roots_require_a_mapping_of_names_to_paths(self):
        for value in (None, [], "images", 1, {1: "/images"}, {"": "/images"}):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "datasets"):
                    Paths.resolve(local=self.write_local({"datasets": value}))
        for value in (None, False, 1, [], {}, "", "  ", "bad\x00path"):
            with self.subTest(root=value):
                with self.assertRaisesRegex(ValueError, "datasets.imagenet"):
                    Paths.resolve(
                        local=self.write_local({"datasets": {"imagenet": value}})
                    )

    def test_cli_rejects_invalid_local_settings_before_creating_workspace(self):
        for value, key in (
            ({"models": None}, "models"),
            ({"datasets": []}, "datasets"),
            ({"datasets": {"imagenet": False}}, "datasets.imagenet"),
        ):
            with self.subTest(value=value):
                path = self.write_local(value)
                result = self.invoke("run", "--local-config", path, "--dry-run")
                self.assert_input_error(result, str(path), key)
                self.assertFalse((self.root / "workspace").exists())

    def test_valid_paths_preserve_precedence_and_custom_dataset_roots(self):
        local = self.write_local(
            {
                "home": str(self.root / "local"),
                "models": str(self.root / "local models"),
                "datasets": {"custom_data": str(self.root / "image folder")},
            }
        )
        with patch.dict(os.environ, {"OPENNPU_QUANT_MODELS": str(self.root / "env")}):
            paths = Paths.resolve(local=local)
            self.assertEqual(self.root / "env", paths.models)
            paths = Paths.resolve(local=local, models_dir=self.root / "explicit")
            self.assertEqual(self.root / "explicit", paths.models)
            self.assertEqual(self.root / "local", paths.home)
            self.assertEqual(
                {"custom_data": str(self.root / "image folder")}, paths.datasets
            )

    def test_default_missing_and_explicit_empty_local_files_remain_valid(self):
        result = self.invoke("run", "--dry-run")
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertEqual(
            str(self.root / "workspace"), json.loads(result.stdout)["paths"]["home"]
        )
        for text in ("", "{}\n", "datasets: {}\n"):
            with self.subTest(text=text):
                local = self.root / "empty.yaml"
                local.write_text(text, encoding="utf-8")
                paths = Paths.resolve(home=self.root, local=local)
                self.assertEqual({}, paths.datasets)


if __name__ == "__main__":
    unittest.main()
