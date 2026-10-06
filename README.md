# opennpu-quant

Quantize ONNX models to INT8 QDQ with optional power-of-two (PoT) scales and fixed-encoding AdaRound. Includes ten CNN recipes, streaming evaluation and bounded calibration/activation caches.

**Linux x86_64 · Python 3.11 · CPU or NVIDIA CUDA** · [한국어](README.ko.md)

[Support](#support) · [Install](#install) · [Datasets](#datasets) · [Configuration](#configuration) · [Python API](#python-api) · [Results](#results) · [Troubleshooting](#troubleshooting) · [Custom models](#custom-models) · [License](#license)

<a id="support"></a>
## Features and supported models

Supports signed INT8 per-tensor activations (symmetric or asymmetric), symmetric INT8 per-tensor/per-channel weights and INT32 bias. Range selection supports MinMax, Percentile and Entropy. Activation scales can be float, PoT nearest or PoT ceil. Weight scales also support PoT MSE. `float` means unconstrained **INT8 scales**, not FP32.

| Recipe name | Task / dataset | External input size | Main metric |
|---|---|---|---|
| `resnet18` | Classification / ImageNet | 224×224 | Top1 |
| `resnet50` | Classification / ImageNet | 224×224 | Top1 |
| `mobilenet_v2` | Classification / ImageNet | 224×224 | Top1 |
| `mobilenet_v3_large` | Classification / ImageNet | 224×224 | Top1 |
| `efficientnet_b0` | Classification / ImageNet | 224×224 | Top1 |
| `inception_v3` | Classification / ImageNet | 299×299 | Top1 |
| `yolov5s` | Detection / COCO | 640×640 | AP |
| `ssd_mobilenet_v2_320` | Detection / COCO | 320×320 | AP |
| `retinaface_mnet025_640` | Face detection / WIDER FACE | 640×640 | AP_medium |
| `deeplabv3_mnv2_voc` | Segmentation / VOC2012 | Padded 513×513 | mIoU |

The package preserves model-specific preprocessing and CPU host graphs. SSD takes external uint8 320×320 input and preserves its internal 300×300 preprocessing. RetinaFace uses the retained 640×640 protocol, which differs from original-resolution WIDER evaluation. DeepLab export checks the first three VOC validation images: prepare VOC before exporting it.

<a id="install"></a>
## Install and run one model

From a checkout, choose a CPU or CUDA profile:

```bash
./setup.sh --device cpu
source .venv/bin/activate
opennpu-quant info
opennpu-quant demo --adaround
```

Setup needs `bash`, `curl`, `tar` and `git`. It uses `uv` from your PATH when available. Otherwise, it uses `.tools/uv`, downloading it if needed. You do not need to install uv beforehand. Setup installs Python 3.11.13 into the checkout and creates `.venv`, leaving system Python unchanged.

For NVIDIA CUDA, use `./setup.sh --device cuda` instead. The CUDA profile uses Torch 2.5.1+cu124 and ONNX Runtime GPU 1.20.2. It requires a compatible NVIDIA driver and works without a separately installed system CUDA toolkit. Run `./setup.sh --help` for the available profiles and optional data support.

The demo uses a tiny synthetic model and needs no model or dataset download. Model weights, ONNX files and images are not bundled.

With your ImageNet folder available:

```bash
opennpu-quant models
opennpu-quant export resnet18
opennpu-quant data prepare imagenet --root /your/imagenet --source folder
opennpu-quant run --models resnet18 --device cpu --conditions fp32,pot_rtn \
  --limit 32 --calibration-samples 8 --save-models --output workspace/smoke-runs
opennpu-quant report workspace/smoke-runs/core10
```

The quantized graph is `workspace/smoke-runs/core10/resnet18/pot_rtn/model.onnx`. To try AdaRound, add `pot_adaround` to the condition list and `--adaround-steps 20`, using a new output directory. Packaged AdaRound runs 10,000 steps per layer by default and can take hours or longer.

`export MODEL --weights FILE` accepts a local copy of the pinned original checkpoint and checks its checksum. Exporting YOLOv5 or RetinaFace acquires upstream code at a pinned revision. Use `--source-dir` to select an existing checkout. Export always performs parity checks.

Downloads occur during setup, export/checkpoint acquisition, explicit ImageNet `data prepare --download`, or `fetch` with a supplied URL-and-SHA256 manifest. `run` never downloads assets or installs dependencies.

<a id="datasets"></a>
## Prepare datasets and paths

Preparation writes manifests under `<home>/data/DATASET/manifest.json` and references existing images without copying them. Only datasets for selected models are evaluated.

```bash
opennpu-quant data prepare imagenet --root /your/imagenet --source folder
opennpu-quant data prepare coco --root /your/coco
opennpu-quant data prepare widerface --root /your/widerface
opennpu-quant data prepare voc --root /your/VOCdevkit/VOC2012
opennpu-quant data status
```

| Dataset | Required layout |
|---|---|
| ImageNet folder | Use `train/<class>/` and `val/<class>/` (or `validation/`). Both splits must use the same 1,000 canonical ImageNet wnid class names. |
| COCO | `val2017/`, the required 128 calibration images in `train2017/`, `annotations/instances_val2017.json` |
| WIDER FACE | `WIDER_train/images/`, `WIDER_val/images/`, `eval_tools/ground_truth/wider_{face,easy,medium,hard}_val.mat` |
| VOC2012 | `JPEGImages/`, `SegmentationClass/`, `ImageSets/Segmentation/train.txt` and `val.txt` |

Reference calibration counts are ImageNet 1,000 and the other datasets 128 each. Full validation counts are ImageNet 50,000, COCO 5,000, WIDER 3,226 and VOC 1,449. Packaged calibration IDs retain their order. VOC label 255 is ignored.

<details>
<summary>ImageNet download and path configuration</summary>

Use `./setup.sh --device cpu --data` (or the CUDA profile) to install optional Hugging Face data support. `data prepare imagenet --source hf --download --root DIR` explicitly streams the pinned gated ImageNet source using your authorized access. This can be a substantial download. Existing images are preserved. Keep access tokens out of committed configuration files.

Directory precedence is **CLI → environment → local YAML → defaults**:

| CLI option | Environment variable | Default |
|---|---|---|
| `--home` | `OPENNPU_QUANT_HOME` | `./workspace` |
| `--models-dir` | `OPENNPU_QUANT_MODELS` | `<home>/models` |
| `--data-dir` | `OPENNPU_QUANT_DATA` | `<home>/data` |
| `--cache-dir` | `OPENNPU_QUANT_CACHE` | `<home>/cache` |
| `--output` (run) | `OPENNPU_QUANT_RUNS` | `<home>/runs` |
| `--weights-dir` | `OPENNPU_QUANT_WEIGHTS` | `<home>/weights` |

An optional, ignored `opennpu_quant.local.yaml` in the working directory can contain:

```yaml
home: ./workspace
datasets:
  imagenet: /your/imagenet
  coco: /your/coco
  widerface: /your/widerface
  voc: /your/VOCdevkit/VOC2012
```

`datasets` overrides the image roots recorded in manifests. The automatically discovered local configuration file is optional. A file explicitly passed with `--local-config` must exist. Use the same path options for preparation and execution. `info` and `run --dry-run` show resolved paths. A dry run does not check GPU availability or run inference.

</details>

<a id="configuration"></a>
## Configure an experiment

`-c core10` selects the default configuration. Use `-c quick` for smaller sample counts and RTN conditions. Both are complete packaged configurations. Preview a selection before execution:

```bash
opennpu-quant run -c quick --models resnet18 --conditions fp32,pot_rtn --dry-run
cp src/opennpu_quant/resources/experiments/quick.yaml experiment.yaml
opennpu-quant run -c experiment.yaml --models resnet18 --dry-run
```

Edit the copied YAML rather than supplying a partial configuration. It contains `name`, `models`, `runtime`, `calibration`, `schemes`, `conditions`, `percentiles`, `adaround`, `evaluation`, `output` and `report`, with optional `activation_cache`. The output directory is `<output>/<name>`.

The `core10` conditions are `fp32`, `pot_rtn`, `pot_adaround`, `float_rtn` and `float_adaround`. Custom names are allowed when defined in the selected configuration. `fp32` must be an empty mapping (`fp32: {}`). `--conditions` automatically includes FP32 and requires that entry to exist. Without a CLI condition selection, an INT8-only configuration is allowed, but provides no FP32 recovery values.

<details>
<summary>Settings, valid values and numerical policy</summary>

| Setting | Contract |
|---|---|
| `runtime.device` | Use `cpu` or `cuda:N`. `calibration.device` also accepts `run`. |
| `runtime.intra_op_threads`, `inter_op_threads` | Use nonnegative integers. Both default to 1 in the packaged configurations. |
| `evaluation.limit` | Positive integer or YAML `null` for the whole validation split |
| `calibration.samples` | Set a positive integer sample count for every selected model's dataset. The mapping must contain at least one entry. |
| `calibration.histogram_bins` | Use an integer at least 2 and no smaller than the scheme's `quantized_bins`. The default is 2048. |
| `schemes.FAMILY` | `scope: basic/all`, activation symmetry, weight granularity, method, percentile and scale policies |
| `percentiles.MODEL.FAMILY` | Number in (0, 100] when the scheme uses `percentile: per_model` |
| `conditions.NAME` | INT8 conditions select a scheme family and an optional boolean `adaround`. The `fp32` condition must be an empty mapping. |
| `output.save_qdq_models`, `save_predictions` | YAML booleans `true`/`false`, not quoted strings |
| `adaround` | Positive integer steps/batch/window limits, nonnegative integer seed, finite learning/loss settings |

Unknown keys, invalid types and an empty `conditions` mapping are rejected before execution. Scheme methods are `minmax`, `percentile` and `entropy`. Use `percentile: null` for `minmax` and `entropy`. Activation scales are `float`, `pot_nearest` and `pot_ceil`. Weight scales additionally allow `pot_mse`. Weight granularity is `per_tensor` or `per_channel`.

Calibration collects min/max and fixed-bin histograms from replayed feeds. Range selection and PoT snapping then produce INT8 QDQ. PoT nearest uses exponent-space ties-to-even. Asymmetric zero points are recomputed after snapping. Weight scales may be enlarged to prevent INT32 bias overflow. Range selection is not reoptimized for PoT.

AdaRound changes only INT8 weight rounding codes. Scales, zero points, bias and graph connections stay fixed. Defaults include 10,000 steps/layer, learning rate 0.001, warm-up 0.2, regularization 0.01, beta 20→2 and seed 20260928. It optimizes local operator reconstruction, not entire residual blocks. `--recalibrate` is described below and does not add post-AdaRound activation recalibration.

The runtime uses `ORT_DISABLE_ALL`. CUDA execution disables TF32, uses `HEURISTIC` convolution search and disables maximum workspace. Preserved host preprocessing/postprocessing graphs run on CPU. Requested CUDA execution must pass placement checks instead of silently falling back to CPU.

</details>

<a id="python-api"></a>
## Use the Python API

This complete example creates a model and feeds in memory, runs on CPU and writes `workspace/api-demo.onnx`. It requires the installed runtime and no external model, dataset or test helper.

```python
import numpy as np
from onnx import helper as h, numpy_helper as n
from opennpu_quant import OrtConfig, Scheme, calibrate, quantize, save_model

model = h.make_model(
    h.make_graph(
        [h.make_node("Gemm", ["images", "w", "b"], ["logits"], name="projection")],
        "tiny",
        [h.make_tensor_value_info("images", 1, [1, 4])],
        [h.make_tensor_value_info("logits", 1, [1, 6])],
        [
            n.from_array(np.arange(24, dtype=np.float32).reshape(4, 6) / 40 - 0.3, "w"),
            n.from_array(np.arange(6, dtype=np.float32) / 10, "b"),
        ],
    ),
    opset_imports=[h.make_opsetid("", 18)],
    ir_version=10,
)


def feeds():
    for i in range(8):
        yield {"images": np.asarray([[i / 8, -0.5, 0.3, 0.1]], np.float32)}


ort = OrtConfig.cpu()
stats = calibrate(model, feeds, ort=ort)
result = quantize(
    model, stats,
    Scheme(activation_scale="pot_ceil", weight_scale="pot_ceil"), ort=ort,
)
save_model(result.model, "workspace/api-demo.onnx")
```

Pass `feeds`, not `feeds()` or a list. Calibration and AdaRound require a zero-argument factory returning a fresh iterator over identical ordered input dictionaries on every call. The example uses finite FP32 arrays. Public model-transforming functions work on copies.

| API | Purpose |
|---|---|
| `calibrate`, `CalibrationConfig`, `CalibrationStats` | Collect and represent reusable calibration statistics |
| `quantize`, `Scheme`, `QuantizationConfig`, `QuantizationResult` | Choose explicit policies and produce a QDQ model with audit data |
| `apply_adaround`, `AdaroundConfig`, `ActivationCacheConfig` | Optimize rounding with bounded intermediate caching and optional checkpoint resume |
| `evaluate`, `EvaluationSample`, `TopKClassification` | Evaluate a sample iterable. Custom evaluators implement `reset`, `update` and `finalize`. |
| `OrtConfig` | CPU/CUDA runtime settings passed through `ort=` |
| `save_model` | Save an ONNX model and return an `identity`/`files` dictionary |
| `prepare_feed_cache` | Advanced verified input cache with explicit identity and source verification callback |

`Scheme()` and `quantize()` without an explicit policy default to all/asymmetric activation/per-channel weights/Percentile 99.99. `QuantizationConfig()` defaults to basic/symmetric activation/per-tensor weights/MinMax. `calibrate()` defaults to `CalibrationConfig()`. Pass explicit policies to match an experiment. Use `help()` on any public name for its arguments and examples.

Importing the package does not initialize numerical runtimes or create directories. Evaluating exported ONNX does not require the original training framework. Using AdaRound requires Torch.

<a id="results"></a>
## Read and export results

For the run above:

```bash
opennpu-quant report workspace/smoke-runs/core10 --stdout tsv
opennpu-quant report workspace/smoke-runs/core10 --stdout csv
opennpu-quant report workspace/smoke-runs/core10 --format csv,tsv
opennpu-quant results workspace/smoke-runs/core10/records.json --stdout tsv
```

| Output | Contents |
|---|---|
| `summary.md`, `summary.json` | Run summary and matched condition aggregates |
| `records.json` | Portable measurements, original metric units, configuration and provenance |
| `accuracy.csv/tsv`, `results.csv/tsv` | The same 15-column raw table, ready for Excel |
| `metrics.csv/tsv` | One row per scalar metric |
| `<model>/<condition>/result.json` | Stores the measurement. Evidence that the run completed is recorded separately. |
| `<model>/<condition>/model.onnx` | Quantized model when `--save-models` is enabled |
| `<model>/<condition>/predictions.jsonl.gz` | Per-sample predictions when `--save-predictions` is enabled |

Raw tables use these columns in order: `model`, `task`, `dataset`, `metric`, `fp32_accuracy`, `int8_accuracy`, `recovery_percent`, `scheme`, `percentile`, `model_source_url`, `model_source_revision`, `top5_percent`, `AP50_percent`, `AP75_percent`, `metrics_json`. The `report` and `results` commands use this schema for CSV/TSV files and `--stdout`. Both `--view accuracy` and `--view full` select these columns. Run IDs, status and configuration details are available in `records.json` and `summary.json`.

The `scheme` column joins settings with ` | ` in a fixed order. Percentile values such as `99.99` are stored in the numeric `percentile` column, so rows with the same model and other settings share a scheme label. The calibration method remains in the label. FP32, MinMax and Entropy rows leave `percentile` empty. Activation and weight scale policies are named separately so that float, PoT nearest, PoT ceil and PoT MSE remain distinguishable. For example:

```text
scope=all | activation=asymmetric | weight=per_channel | activation_scale=pot_ceil | weight_scale=pot_ceil | calibration=percentile | rounding=AdaRound
```

FP32 rows use `scheme=FP32` and leave `int8_accuracy` empty. Unmeasured candidates still show their planned scheme. Failed or missing measurements have empty accuracy cells. A measured INT8 value can be present without a matched FP32 baseline, in which case recovery is empty. Additional policy settings appear inside `scheme`. Missing task metadata is left empty. `metrics_json` retains the original metric values and units.

Recovery is calculated as `recovery_percent = 100 × INT8 metric / matched FP32 metric`. The JSON details also record `delta_pp = INT8(%) − FP32(%)`. FP32 and INT8 records must use matching model and evaluation conditions. Missing or failed measurements are not treated as zeros. A zero FP32 metric has no recovery ratio. Recovery above 100% is not clipped.

CSV uses a UTF-8 BOM and TSV uses UTF-8. `fp32_accuracy`, `int8_accuracy` and the `_percent` columns contain percentage values as numbers, such as `78.5` for 78.5%. Keep those cells in a numeric format in Excel because percentage formatting multiplies the displayed value by 100. Detailed metrics retain their units. COCO `AP_medium` describes medium-size objects and is different from WIDER's medium difficulty subset. `report --format` only writes selected formats and preserves other existing exports. Metadata commands `report`, `results` and `compare` do not run inference.

<details>
<summary>Compare measurements from a catalog</summary>

`compare` accepts a **catalog/history pair** describing models, quantization settings and measurements. To compare the records and export Excel:

```bash
uv pip install --python .venv/bin/python -e '.[report]'
opennpu-quant compare /path/to/catalog.json --output workspace/comparison --xlsx
```

Run the installation command from the repository root after setup. If `uv` is not on your PATH, use `./.tools/uv`. Setup installs uv at that location when it needs to download it.

Use a new output directory because existing snapshots are rejected. Without `--xlsx`, CSV/TSV/JSON/Markdown output needs no Excel extra. The input format is defined in [comparison.py](src/opennpu_quant/comparison.py), with a complete generated example in [test_comparison.py](tests/test_comparison.py). For an ordinary run's `records.json`, use `results` instead.

The comparison presents Summary, Task_Summary, Model_Best, Core_Raw, Core_Measured, Complete10_Raw, All_History and Coverage. Its core candidate grid is all-scope Asym-PC/Sym-PT, float/PoT-ceil, Percentile 99.9/99.99/99.999, RTN/AdaRound.

Core_Raw, Core_Measured, Complete10_Raw and All_History use the same 15 raw columns in CSV, TSV and Excel. Excel keeps accuracy and recovery cells numeric, freezes the header and enables filters. Detailed comparison records remain in `comparison.json`, with exclusion reasons also listed in Coverage.

`Core_Measured` is generated from `Core_Raw` and contains only rows with a measured FP32 or INT8 accuracy, including zero values. Excel opens this sheet first. Each percentile measurement remains a separate row. Use `model` and `scheme` to group the rows and take the maximum `int8_accuracy` across percentiles.

</details>

<a id="troubleshooting"></a>
## Reuse, caches and troubleshooting

| Situation | What to do |
|---|---|
| Unknown model or condition | Check `models` and the selected experiment. Local models require `recipe.yaml`. |
| Missing dataset manifest | Prepare the dataset with the same `--home`/`--data-dir` used for the run. |
| YAML/type/range error | Check the reported file, line/column or key path. Remove quotes from boolean values and use integers for counts. |
| Missing `--local-config` file | Supply an existing YAML file or omit the option for default path discovery. |
| Existing run no longer matches | Prefer a new `--output` to preserve previous results. `--force` bypasses completed-result reuse and recomputes selected results. Calibration caches and AdaRound checkpoints may still be reused. |
| Need new calibration | `--recalibrate` requires `calibration.source: compute` and an INT8 condition. It recollects from the original FP32 network and rebuilds dependent selected INT8 results. |
| Feeds rejected | Use a replayable factory such as `lambda: iter(feed_list)`, preserving values and order. |
| Runtime version warning | Verified ORT version: 1.20.2. `--strict-versions` rejects other versions. |
| Existing comparison output | Choose a new directory to preserve the prior snapshot. |

<details>
<summary>Cache budgets, identity and resume</summary>

Packaged presets enable calibration input caching with a 2 GiB limit **per cache**, under `<cache-dir>/inputs/<model>/<identity>/`. It stores the prepared network's fixed-shape, finite FP32 batch-one feeds, not validation inputs. Disable it with `--no-input-cache` or `calibration.input_cache.enabled: false`.

AdaRound intermediate activation caching has separate controls: `--host-cache-gib`, `--gpu-window-mib` and `--no-cache`. Budgets limit the data retained in each cache. They do not preallocate memory or set a total disk quota. To reclaim space, delete an input cache identity directory only when no job is using it.

Reuse automatically checks model files, ordered inputs, preprocessing and execution settings. If these change, use a new output directory. Moving files without changing their contents preserves model identity.

AdaRound checkpoints resume after the validated sequence of completed layers. An interrupted layer restarts. `--force` is not a request to clear every cache/checkpoint. The retained segmentation prediction is the label map used by the current evaluator.

</details>

<a id="custom-models"></a>
## Custom models

Use the Python API for arbitrary supported ONNX graphs, with your own feed factory and evaluator. For CLI integration, place `recipe.yaml` and `network.onnx` under `<models-dir>/my_model/`, with optional CPU host graphs. Start from a [packaged recipe](src/opennpu_quant/resources/models/resnet18.yaml), then adapt its input shape/layout, dataset, task, metric, preprocessing, decoder, source metadata and graph boundaries. Add the model and any per-model percentiles to a copied experiment. Local recipes take precedence over packaged names.

Built-in decoders cover logits, YOLOv5, TF SSD, RetinaFace 640 and segmentation. Use a custom Python evaluator for unrelated outputs. Keep network/host boundaries explicit. The `all` scope quantizes supported feature-data operations inside the network. CPU normalization/NMS and shape/index parameters remain outside its scope. Do not change descriptive fields to bypass fixed implementation checks.

Custom `module:function` preprocessing cannot reliably identify helper/global/file dependencies. Disable persistent input caching for it. Fresh execution remains possible. Automatic result reuse is refused, and FP32–INT8 recovery remains unmatched without sufficient preprocessing/protocol evidence.

<a id="license"></a>
## License and references

Package code is licensed under [Apache-2.0](LICENSE). Retain [NOTICE](NOTICE) and [third-party attribution](THIRD_PARTY.md). Original model code, checkpoints and datasets retain their own terms. ONNX conversion does not grant redistribution rights.

- AdaRound: [Up or Down? Adaptive Rounding for Post-Training Quantization](https://proceedings.mlr.press/v119/nagel20a.html), ICML 2020.
- Dataset sources: [ImageNet](https://huggingface.co/datasets/ILSVRC/imagenet-1k), [COCO](https://cocodataset.org/#download), [WIDER FACE](http://shuoyang1213.me/WIDERFACE/), [VOC2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/).
