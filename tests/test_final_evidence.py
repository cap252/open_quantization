import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from opennpu_quant._io import read_json
from opennpu_quant.comparison import write_comparison
from opennpu_quant.results import tables
from tests.test_comparison import fixture as comparison_fixture, save_fixture


class FinalEvidenceTests(unittest.TestCase):
    def test_real_decode_and_evaluation_settings_reject_old_fp32_pair(self):
        import onnx
        from onnx import helper as h
        import yaml
        from opennpu_quant import runner
        from tests.test_result_reliability import fixture

        for setting in ("decode", "evaluation"):
            with self.subTest(setting=setting), tempfile.TemporaryDirectory() as temp:
                paths, config = fixture(Path(temp), host="postprocess")
                post = paths.models / "toy/postprocess.onnx"
                model = onnx.load(post)
                model.graph.node.append(
                    h.make_node("Identity", ["scores"], ["alternate_scores"])
                )
                model.graph.output.append(
                    h.make_tensor_value_info("alternate_scores", 1, [1, 6])
                )
                onnx.save(model, post)
                runner.run(config, paths)
                out = paths.runs / config["name"]
                base = next(
                    r
                    for r in read_json(out / "records.json")["records"]
                    if r["condition"] == "fp32"
                )
                before = read_json(out / "toy/fp32/result.json")
                changed = copy.deepcopy(config)
                changed["name"] = setting
                if setting == "decode":
                    recipe_path = paths.models / "toy/recipe.yaml"
                    recipe = yaml.safe_load(recipe_path.read_text())
                    recipe["decode"]["output"] = "alternate_scores"
                    recipe_path.write_text(yaml.safe_dump(recipe))
                else:
                    changed["evaluation"]["limit"] = 1
                runner.run(changed, paths)
                after_dir = paths.runs / changed["name"]
                q = next(
                    r
                    for r in read_json(after_dir / "records.json")["records"]
                    if r["condition"] == "pot_rtn"
                )
                after = read_json(after_dir / "toy/pot_rtn/result.json")
                self.assertEqual(before["graph_identities"], after["graph_identities"])
                self.assertEqual(
                    before["preprocessing_identity"], after["preprocessing_identity"]
                )
                self.assertNotEqual(
                    before["evaluation_identity"], after["evaluation_identity"]
                )
                q["run_id"] = base["run_id"]
                row = tables([base, q])[0][1]
                self.assertEqual("mismatch", row["pairing_status"])
                self.assertIsNone(row["recovery_pct"])
                self.assertIsNotNone(row["int8_pct"])

    @unittest.skipUnless(importlib.util.find_spec("openpyxl"), "optional report extra")
    def test_xlsx_unmatched_keeps_accuracy_and_excludes_summary(self):
        from openpyxl import load_workbook

        catalog, records = comparison_fixture()
        for row in records:
            if row["record_id"].startswith("00_asym_pc_pot_"):
                row["provenance"]["preprocessing_identity"] = "changed"
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = save_fixture(root, catalog, records)
            write_comparison(source, root / "out", xlsx=True)
            expected = json.loads((root / "out/comparison.json").read_text())["sheets"]
            workbook = load_workbook(
                root / "out/CORE10_COMPARISON.xlsx", read_only=True, data_only=True
            )
            try:

                def rows(sheet):
                    values = list(workbook[sheet].values)
                    return [dict(zip(values[0], row)) for row in values[1:]]

                exported = rows("All_History")
                self.assertEqual(len(expected["All_History"]), len(exported))
                unmatched = [
                    (row, original)
                    for row, original in zip(exported, expected["All_History"])
                    if original["record_id"].startswith("00_asym_pc_pot_")
                ]
                self.assertEqual(6, len(unmatched))
                for row, original in unmatched:
                    self.assertEqual("mismatch", original["pairing_status"])
                    self.assertFalse(original["eligible"])
                    self.assertIsInstance(row["int8_accuracy"], (int, float))
                    self.assertAlmostEqual(
                        original["accuracy_percent"], row["int8_accuracy"]
                    )
                    self.assertIsNone(row["recovery_percent"])
                    self.assertIn("preprocessing", original["reason"])
                summary = [
                    r
                    for r in rows("Summary")
                    if r["selection"] == "validation_posthoc"
                    and r["group"] == "asym_pc"
                    and r["scale_family"] == "pot"
                ]
                self.assertEqual(3, len(summary))
                for row in summary:
                    self.assertEqual("incomplete", row["state"])
                    self.assertIsNone(row["mean_recovery_percent"])
                    self.assertIsNone(row["target_met"])
                other = next(
                    r
                    for r in rows("Summary")
                    if r["selection"] == "validation_posthoc"
                    and r["group"] == "asym_pc"
                    and r["scale_family"] == "float"
                )
                self.assertEqual("complete", other["state"])
                self.assertIsInstance(other["mean_recovery_percent"], (int, float))
            finally:
                workbook.close()
