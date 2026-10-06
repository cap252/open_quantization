import csv
import io
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from opennpu_quant._io import atomic_json, object_hash, sha256
from opennpu_quant.results import (
    export_records,
    tables,
    normalize_scheme,
    read_records,
)


def record(condition="fp32", value=0.8, **changes):
    result = dict(
        run_id="r",
        model="m",
        model_name="테스트 모델",
        condition=condition,
        state="evaluated",
        origin="historical_research",
        primary_metric="top1",
        dataset="imagenet",
        split="validation",
        samples=100,
        expected_samples=100,
        complete=True,
        evaluation_identity="same-inputs",
        model_source={"repo": "https://example.test/model"},
        metrics={
            "top1": {"value": value, "unit": "fraction"},
            "top5": {"value": 0.95, "unit": "fraction"},
        },
        quantization=None
        if condition == "fp32"
        else normalize_scheme(
            dict(
                scope="all",
                activation_symmetric=False,
                weight_granularity="per_channel",
                method="percentile",
                percentile=99.9,
                activation_scale="pot_ceil",
                weight_scale="pot_ceil",
            )
        ),
        evidence={
            "result_hash_verified": True,
            "model_identity": "source",
            "preprocessing_identity": "pre",
            "source_code_identity": "code",
            "runtime": {"optimization": "ORT_DISABLE_ALL"},
        },
    )
    result.update(changes)
    return result


class ResultTests(unittest.TestCase):
    def test_units_pairing_missing_and_zero(self):
        base, q = record(), record("pot_rtn", 0.78)
        rows, detail = tables([base, q])
        self.assertAlmostEqual(97.5, rows[1]["recovery_pct"])
        self.assertAlmostEqual(-2, rows[1]["delta_pp"])
        self.assertEqual(78, rows[1]["int8_pct"])
        self.assertIsNone(rows[0]["int8_pct"])
        self.assertEqual(4, len(detail))
        q["metrics"]["top1"] = {"value": 78, "unit": "percent"}
        self.assertAlmostEqual(97.5, tables([base, q])[0][1]["recovery_pct"])
        q["evaluation_identity"] = "different-inputs"
        self.assertIsNone(tables([base, q])[0][1]["recovery_pct"])
        q = record(
            "pending",
            None,
            state="pending",
            metrics={},
            complete=False,
            evaluation_identity=None,
        )
        row = tables([base, q])[0][1]
        self.assertEqual(80, row["fp32_pct"])
        self.assertIsNone(row["int8_pct"])
        self.assertIsNone(row["recovery_pct"])
        self.assertIsNone(
            tables([record(value=0), record("q", 0)])[0][1]["recovery_pct"]
        )
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            tables([base, base])
        with self.assertRaises(ValueError):
            tables([record(value=float("nan"))])
        q = record("q", 0.78)
        q["metrics"]["top1"]["unit"] = "unknown"
        self.assertIsNone(tables([base, q])[0][1]["recovery_pct"])

    def test_historical_protocol_mismatch_and_missing_evidence_are_unmatched(self):
        for field in (
            "model_identity",
            "preprocessing_identity",
            "source_code_identity",
            "runtime",
        ):
            for missing in (True, False):
                with self.subTest(field=field, missing=missing):
                    base, q = record(), record("q", 0.78)
                    if missing:
                        q["evidence"].pop(field)
                    else:
                        q["evidence"][field] = (
                            {"optimization": "other"} if field == "runtime" else "other"
                        )
                    row = tables([base, q])[0][1]
                    self.assertIsNone(row["recovery_pct"])
                    self.assertEqual(
                        "insufficient_evidence" if missing else "mismatch",
                        row["pairing_status"],
                    )

    def test_excel_raw_schema_unicode_and_details_roundtrip(self):
        base, q = record(), record("q", 0.78)
        q["model_name"] = "=1+1\t위험\n줄"
        q["metric_details"] = {"per_class_iou": [0.2, None, 0.4]}
        with tempfile.TemporaryDirectory() as temp:
            export_records([base, q], temp)
            p = Path(temp)
            self.assertTrue(
                (p / "results.csv").read_bytes().startswith(b"\xef\xbb\xbf")
            )
            content = (p / "results.tsv").read_text()
            self.assertEqual(3, len(content.splitlines()))
            cells = list(csv.DictReader(io.StringIO(content), delimiter="\t"))
            self.assertTrue(cells[1]["model"].startswith("'="))
            self.assertEqual(78, float(cells[1]["int8_accuracy"]))
            self.assertEqual(97.5, float(cells[1]["recovery_percent"]))
            self.assertEqual(
                q["metric_details"],
                read_records(p / "records.json")["records"][1]["metric_details"],
            )
            self.assertEqual([base, q], read_records(p / "records.json")["records"])
            self.assertEqual(
                4,
                len(
                    list(
                        csv.DictReader(
                            io.StringIO(
                                (p / "metrics.csv").read_text(encoding="utf-8-sig")
                            )
                        )
                    )
                ),
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "opennpu_quant",
                    "results",
                    str(p / "records.json"),
                    "--stdout",
                    "tsv",
                    "--view",
                    "full",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            self.assertEqual(content, completed.stdout)
            self.assertEqual("", completed.stderr)
            code = "from opennpu_quant.cli import main; import sys; main(['results',sys.argv[1]]); assert not set(['numpy','onnx','onnxruntime','torch','tensorflow','cv2']) & set(sys.modules)"
            subprocess.run(
                [sys.executable, "-c", code, str(p / "records.json")],
                stdout=subprocess.DEVNULL,
                check=True,
            )

    def test_report_preserves_partial_denominators_and_rejects_tampering(self):
        from opennpu_quant.report import report

        cfg = dict(
            name="test",
            models=["m"],
            conditions={"fp32": {}, "pot_rtn": {"scheme": "pot"}},
            schemes={
                "pot": dict(
                    scope="all",
                    activation_symmetric=False,
                    weight_granularity="per_channel",
                    method="percentile",
                    percentile=99.9,
                )
            },
            adaround={},
            evaluation={"split": "validation"},
        )
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp)
            atomic_json(p / "effective_config.json", cfg)

            def write(c, v):
                f = p / "m" / c / "result.json"
                r = dict(
                    model="m",
                    title="테스트",
                    metric="top1",
                    value=v,
                    metrics={"top1": v, "top5": 0.9},
                    samples=5,
                    expected_samples=100,
                    complete=False,
                    reference=False,
                    model_identity="source",
                    preprocessing_identity="pre",
                    evaluation_identity="same",
                    config_identity=object_hash(cfg),
                    scheme=cfg["schemes"]["pot"] if c != "fp32" else None,
                    model_metadata={
                        "source": {"repo": "https://example.test"},
                        "dataset": "imagenet",
                        "metric": "top1",
                    },
                )
                atomic_json(f, r)
                atomic_json(
                    f.parent / "complete.json",
                    {"result_sha256": sha256(f), "files": {}},
                )

            write("fp32", 0.8)
            write("pot_rtn", 0.78)
            value = report(p)
            self.assertEqual(2, len(value["rows"]))
            self.assertIsNone(value["mean_recovery"]["pot_rtn"])
            self.assertAlmostEqual(97.5, value["rows"][1]["recovery_pct"])
            (p / "m/pot_rtn/result.json").write_text("{}")
            value = report(p)
            self.assertEqual("unverified", value["rows"][1]["state"])
            self.assertIsNone(value["rows"][1]["value_pct"])
            self.assertIn("—", (p / "summary.md").read_text())
            self.assertTrue((p / "records.json").is_file())
            code = "from opennpu_quant.cli import main; import sys; main(['report',sys.argv[1],'--stdout','tsv']); assert not set(['numpy','onnx','onnxruntime','torch','tensorflow','cv2']) & set(sys.modules)"
            subprocess.run(
                [sys.executable, "-c", code, str(p)],
                stdout=subprocess.DEVNULL,
                check=True,
            )
