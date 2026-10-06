# opennpu-quant

ONNX 모델에 INT8 QDQ, 선택적 power-of-two(PoT) scale과 고정 encoding AdaRound를 적용합니다. CNN 10개 recipe, 스트리밍 평가, 용량 제한 calibration 입력·activation 캐시를 제공합니다.

**Linux x86_64 · Python 3.11 · CPU 또는 NVIDIA CUDA** · [English](README.md)

[지원 범위](#support) · [설치](#install) · [데이터](#datasets) · [설정](#configuration) · [Python API](#python-api) · [결과](#results) · [문제 해결](#troubleshooting) · [사용자 모델](#custom-models) · [라이선스](#license)

<a id="support"></a>
## 기능과 지원 모델

Signed INT8 per-tensor activation(대칭 또는 비대칭), symmetric INT8 per-tensor/per-channel weight와 INT32 bias를 지원합니다. 범위 선택은 MinMax·Percentile·Entropy를 제공합니다. Activation scale은 float·PoT nearest·PoT ceil이며 weight는 PoT MSE도 지원합니다. `float`는 제약 없는 **INT8 scale**을 뜻하며 FP32가 아닙니다.

| Recipe 이름 | 작업 / 데이터셋 | 외부 입력 크기 | 대표 지표 |
|---|---|---|---|
| `resnet18` | 분류 / ImageNet | 224×224 | Top1 |
| `resnet50` | 분류 / ImageNet | 224×224 | Top1 |
| `mobilenet_v2` | 분류 / ImageNet | 224×224 | Top1 |
| `mobilenet_v3_large` | 분류 / ImageNet | 224×224 | Top1 |
| `efficientnet_b0` | 분류 / ImageNet | 224×224 | Top1 |
| `inception_v3` | 분류 / ImageNet | 299×299 | Top1 |
| `yolov5s` | 객체 검출 / COCO | 640×640 | AP |
| `ssd_mobilenet_v2_320` | 객체 검출 / COCO | 320×320 | AP |
| `retinaface_mnet025_640` | 얼굴 검출 / WIDER FACE | 640×640 | AP_medium |
| `deeplabv3_mnv2_voc` | 분할 / VOC2012 | 513×513 패딩 | mIoU |

모델별 전처리와 CPU host graph를 유지합니다. SSD는 외부 uint8 320×320 입력을 받고 내부 300×300 전처리를 보존합니다. RetinaFace는 640×640 규약으로 원본 해상도 WIDER 평가와 다릅니다. DeepLab export는 VOC validation 첫 3장으로 검사하므로 VOC를 먼저 준비해야 합니다.

<a id="install"></a>
## 설치와 모델 하나 실행

저장소를 받은 후 CPU 또는 CUDA profile을 선택합니다.

```bash
./setup.sh --device cpu
source .venv/bin/activate
opennpu-quant info
opennpu-quant demo --adaround
```

설치에는 `bash`, `curl`, `tar`, `git`이 필요합니다. Setup은 PATH에 있는 `uv`를 사용합니다. PATH에 없으면 `.tools/uv`를 사용하며, 이 파일도 없으면 자동으로 내려받습니다. uv를 미리 설치할 필요는 없습니다. Setup은 저장소 안에 Python 3.11.13을 설치하고 `.venv`를 만듭니다. 시스템 Python은 변경하지 않습니다.

NVIDIA CUDA를 사용하려면 대신 `./setup.sh --device cuda`를 실행합니다. CUDA profile은 Torch 2.5.1+cu124와 ONNX Runtime GPU 1.20.2를 사용합니다. 호환 NVIDIA driver가 필요하며 별도 시스템 CUDA toolkit 없이 동작합니다. `./setup.sh --help`에서 profile과 선택적 데이터 지원 옵션을 확인할 수 있습니다.

데모는 작은 합성 모델을 사용하므로 모델·데이터셋 다운로드가 필요 없습니다. Weight·ONNX 파일·이미지는 저장소에 포함하지 않습니다.

ImageNet 폴더를 준비한 상태에서 실행합니다.

```bash
opennpu-quant models
opennpu-quant export resnet18
opennpu-quant data prepare imagenet --root /your/imagenet --source folder
opennpu-quant run --models resnet18 --device cpu --conditions fp32,pot_rtn \
  --limit 32 --calibration-samples 8 --save-models --output workspace/smoke-runs
opennpu-quant report workspace/smoke-runs/core10
```

양자화 그래프는 `workspace/smoke-runs/core10/resnet18/pot_rtn/model.onnx`에 저장됩니다. AdaRound를 짧게 확인하려면 조건 목록에 `pot_adaround`, 옵션에 `--adaround-steps 20`을 추가하고 새 출력 폴더를 사용합니다. 기본 AdaRound는 레이어당 10,000 steps로 수 시간 이상 걸릴 수 있습니다.

`export MODEL --weights FILE`로 고정된 원본 checkpoint를 직접 지정할 수 있으며 checksum을 검사합니다. YOLOv5·RetinaFace export는 고정된 upstream 소스를 가져오고 `--source-dir`로 기존 checkout을 지정할 수도 있습니다. Export는 항상 parity 검사를 수행합니다.

다운로드는 setup, export/checkpoint 준비, 명시적인 ImageNet `data prepare --download`, URL·SHA256 manifest를 지정한 `fetch`에서 수행합니다. `run`은 자산을 내려받거나 의존성을 설치하지 않습니다.

<a id="datasets"></a>
## 데이터와 경로 준비

준비 과정은 `<home>/data/DATASET/manifest.json`을 만들고 기존 이미지를 복사하지 않고 참조합니다. 선택한 모델의 데이터셋만 평가합니다.

```bash
opennpu-quant data prepare imagenet --root /your/imagenet --source folder
opennpu-quant data prepare coco --root /your/coco
opennpu-quant data prepare widerface --root /your/widerface
opennpu-quant data prepare voc --root /your/VOCdevkit/VOC2012
opennpu-quant data status
```

| 데이터셋 | 필요한 구조 |
|---|---|
| ImageNet 폴더 | `train/<class>/`와 `val/<class>/` 또는 `validation/`을 사용합니다. 두 split에는 같은 1,000개 ImageNet 표준 wnid 클래스 이름이 있어야 합니다. |
| COCO | `val2017/`, `train2017/`의 필수 calibration 이미지 128장, `annotations/instances_val2017.json` |
| WIDER FACE | `WIDER_train/images/`, `WIDER_val/images/`, `eval_tools/ground_truth/wider_{face,easy,medium,hard}_val.mat` |
| VOC2012 | `JPEGImages/`, `SegmentationClass/`, `ImageSets/Segmentation/train.txt`와 `val.txt` |

기준 calibration 수는 ImageNet 1,000장, 나머지 데이터셋은 각각 128장입니다. 전체 validation 수는 ImageNet 50,000장, COCO 5,000장, WIDER 3,226장, VOC 1,449장입니다. 패키지의 calibration ID 순서를 유지합니다. VOC label 255는 무시합니다.

<details>
<summary>ImageNet 다운로드와 경로 설정</summary>

`./setup.sh --device cpu --data` 또는 CUDA profile로 선택적 Hugging Face 데이터 의존성을 설치합니다. `data prepare imagenet --source hf --download --root DIR`는 사용자에게 허가된 접근 권한으로 고정된 gated ImageNet 소스를 명시적으로 스트리밍합니다. 다운로드 규모가 클 수 있습니다. 기존 이미지는 덮어쓰지 않습니다. 토큰은 커밋할 설정에 넣지 않습니다.

경로 우선순위는 **CLI → 환경변수 → 로컬 YAML → 기본값**입니다.

| CLI 옵션 | 환경변수 | 기본값 |
|---|---|---|
| `--home` | `OPENNPU_QUANT_HOME` | `./workspace` |
| `--models-dir` | `OPENNPU_QUANT_MODELS` | `<home>/models` |
| `--data-dir` | `OPENNPU_QUANT_DATA` | `<home>/data` |
| `--cache-dir` | `OPENNPU_QUANT_CACHE` | `<home>/cache` |
| `--output` (run) | `OPENNPU_QUANT_RUNS` | `<home>/runs` |
| `--weights-dir` | `OPENNPU_QUANT_WEIGHTS` | `<home>/weights` |

작업 폴더의 선택적 `opennpu_quant.local.yaml`은 Git에서 제외되며 다음처럼 작성할 수 있습니다.

```yaml
home: ./workspace
datasets:
  imagenet: /your/imagenet
  coco: /your/coco
  widerface: /your/widerface
  voc: /your/VOCdevkit/VOC2012
```

`datasets`는 manifest에 기록된 이미지 root를 덮어씁니다. 자동 탐색하는 로컬 파일은 없어도 되지만 `--local-config`로 명시한 파일은 반드시 존재해야 합니다. 데이터 준비와 실행에 같은 경로 옵션을 사용하세요. `info`와 `run --dry-run`으로 적용 경로를 확인할 수 있습니다. dry-run은 GPU 존재 여부를 검사하거나 추론을 실행하지 않습니다.

</details>

<a id="configuration"></a>
## 실험 설정

기본값은 `-c core10`이며 `-c quick`은 표본 수를 줄이고 RTN 조건을 사용합니다. 둘 다 완전한 패키지 설정입니다. 실행 전에 선택한 구성을 확인합니다.

```bash
opennpu-quant run -c quick --models resnet18 --conditions fp32,pot_rtn --dry-run
cp src/opennpu_quant/resources/experiments/quick.yaml experiment.yaml
opennpu-quant run -c experiment.yaml --models resnet18 --dry-run
```

일부 키만 작성한 설정 대신 복사한 YAML을 수정합니다. `name`, `models`, `runtime`, `calibration`, `schemes`, `conditions`, `percentiles`, `adaround`, `evaluation`, `output`, `report`와 선택적 `activation_cache`를 포함합니다. 결과 폴더는 `<output>/<name>`입니다.

`core10`의 조건은 `fp32`, `pot_rtn`, `pot_adaround`, `float_rtn`, `float_adaround`입니다. 선택한 설정에 정의한 사용자 조건 이름도 허용합니다. `fp32`는 반드시 빈 mapping인 `fp32: {}`여야 합니다. `--conditions`는 FP32를 자동으로 포함하므로 설정에 해당 항목이 필요합니다. CLI로 조건을 별도 선택하지 않으면 INT8 단독 설정도 허용하지만 FP32 복원율은 제공하지 않습니다.

<details>
<summary>설정값과 수치 규약</summary>

| 설정 | 계약 |
|---|---|
| `runtime.device` | `cpu` 또는 `cuda:N`을 사용합니다. `calibration.device`는 `run`도 허용합니다. |
| `runtime.intra_op_threads`, `inter_op_threads` | 0 이상의 정수를 사용합니다. 패키지 설정의 기본값은 각각 1입니다. |
| `evaluation.limit` | 양의 정수 또는 전체 validation을 뜻하는 YAML `null` |
| `calibration.samples` | 선택한 각 모델의 데이터셋에 양의 정수 표본 수를 지정합니다. Mapping에는 적어도 한 항목이 있어야 합니다. |
| `calibration.histogram_bins` | 2 이상이며 scheme의 `quantized_bins`보다 작지 않은 정수를 사용합니다. 기본값은 2048입니다. |
| `schemes.FAMILY` | `scope: basic/all`, activation 대칭성, weight granularity, method, percentile, scale 정책 |
| `percentiles.MODEL.FAMILY` | scheme이 `percentile: per_model`일 때 (0, 100] 범위의 수치 |
| `conditions.NAME` | INT8 조건은 scheme family와 선택적 boolean `adaround`를 지정합니다. `fp32` 조건은 반드시 빈 mapping이어야 합니다. |
| `output.save_qdq_models`, `save_predictions` | 문자열이 아닌 YAML boolean `true`/`false` |
| `adaround` | 양의 정수 steps/batch/window 한도, 0 이상의 정수 seed, 유한한 학습·손실 설정 |

알 수 없는 키, 잘못된 자료형, 빈 `conditions` mapping은 실행 전에 거부합니다. Scheme method는 `minmax`, `percentile`, `entropy`이며 minmax/entropy에는 `percentile: null`을 사용합니다. Activation scale은 `float`, `pot_nearest`, `pot_ceil`, weight는 추가로 `pot_mse`를 허용합니다. Weight granularity는 `per_tensor` 또는 `per_channel`입니다.

Calibration은 반복 가능한 feeds에서 min/max와 고정 bin histogram을 모읍니다. 이후 범위 선택과 PoT snapping을 적용해 INT8 QDQ를 만듭니다. PoT nearest는 지수 공간에서 ties-to-even을 사용합니다. 비대칭 zero point는 snapping 후 다시 계산하고 INT32 bias overflow를 방지하려 weight scale이 커질 수 있습니다. PoT에 맞춰 범위를 다시 최적화하지 않습니다.

AdaRound는 INT8 weight rounding codes만 바꾸고 scale·zero point·bias·그래프 연결은 고정합니다. 기본값은 레이어당 10,000 steps, learning rate 0.001, warm-up 0.2, regularization 0.01, beta 20→2, seed 20260928입니다. 전체 residual block이 아닌 지역 연산 출력 복원을 최적화합니다. 아래의 `--recalibrate`도 AdaRound 이후 activation 재보정 기능은 아닙니다.

런타임은 `ORT_DISABLE_ALL`을 사용합니다. CUDA는 TF32를 끄고 convolution search는 `HEURISTIC`, max workspace는 비활성화합니다. 보존한 host 전후처리 그래프는 CPU에서 실행합니다. 요청한 CUDA 실행은 배치 검사를 통과해야 하며 조용히 CPU로 대체하지 않습니다.

</details>

<a id="python-api"></a>
## Python API 사용

다음은 모델과 feeds를 메모리에서 직접 만들고 CPU에서 실행해 `workspace/api-demo.onnx`를 저장하는 완전한 예제입니다. 설치된 런타임 외에 외부 모델·데이터셋·테스트 helper가 필요하지 않습니다.

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

`feeds()`나 리스트 대신 `feeds`를 전달합니다. Calibration과 AdaRound는 인자가 없고 호출할 때마다 같은 순서·값의 입력 dict를 새 iterator로 반환하는 factory를 요구합니다. 예제는 유한한 FP32 배열을 사용합니다. 공개 모델 변환 함수는 복사본을 다룹니다.

| API | 용도 |
|---|---|
| `calibrate`, `CalibrationConfig`, `CalibrationStats` | 재사용 가능한 보정 통계 수집과 표현 |
| `quantize`, `Scheme`, `QuantizationConfig`, `QuantizationResult` | 명시적 정책 선택과 audit 정보를 포함한 QDQ 모델 생성 |
| `apply_adaround`, `AdaroundConfig`, `ActivationCacheConfig` | 용량 제한 중간 캐시와 선택적 checkpoint 재개를 사용하는 rounding 최적화 |
| `evaluate`, `EvaluationSample`, `TopKClassification` | Sample iterable을 평가합니다. 사용자 evaluator는 `reset`, `update`, `finalize`를 구현해야 합니다. |
| `OrtConfig` | `ort=`로 전달하는 CPU/CUDA 실행 설정 |
| `save_model` | ONNX 저장 후 `identity`/`files`를 담은 dict 반환 |
| `prepare_feed_cache` | 명시적 identity와 원본 검증 callback을 받는 고급 입력 캐시 |

`Scheme()`와 정책을 명시하지 않은 `quantize()`의 기본값은 all/비대칭 activation/per-channel weight/Percentile 99.99입니다. `QuantizationConfig()`는 basic/대칭 activation/per-tensor weight/MinMax입니다. `calibrate()`의 기본값은 `CalibrationConfig()`이며 특정 실험과 맞추려면 정책을 명시하세요. 각 공개 이름의 `help()`에서 인자와 예제를 볼 수 있습니다.

패키지 import만으로 수치 런타임을 초기화하거나 폴더를 만들지 않습니다. Export된 ONNX 평가는 원본 학습 프레임워크를 요구하지 않으며, AdaRound 사용 시에는 Torch가 필요합니다.

<a id="results"></a>
## 결과 확인과 내보내기

앞의 실행 결과를 확인합니다.

```bash
opennpu-quant report workspace/smoke-runs/core10 --stdout tsv
opennpu-quant report workspace/smoke-runs/core10 --stdout csv
opennpu-quant report workspace/smoke-runs/core10 --format csv,tsv
opennpu-quant results workspace/smoke-runs/core10/records.json --stdout tsv
```

| 출력 | 내용 |
|---|---|
| `summary.md`, `summary.json` | 실행 요약과 대응 검증을 거친 조건별 집계 |
| `records.json` | 공유 가능한 측정값, 원래 metric 단위, 설정과 provenance |
| `accuracy.csv/tsv`, `results.csv/tsv` | Excel에서 사용할 동일한 15열 raw 표 |
| `metrics.csv/tsv` | scalar metric마다 한 행 |
| `<model>/<condition>/result.json` | 측정값을 저장합니다. 실행 완료를 확인하는 증거는 별도로 기록합니다. |
| `<model>/<condition>/model.onnx` | `--save-models`를 켰을 때 양자화 모델 |
| `<model>/<condition>/predictions.jsonl.gz` | `--save-predictions`를 켰을 때 표본별 예측 |

Raw 표의 열 순서는 `model`, `task`, `dataset`, `metric`, `fp32_accuracy`, `int8_accuracy`, `recovery_percent`, `scheme`, `percentile`, `model_source_url`, `model_source_revision`, `top5_percent`, `AP50_percent`, `AP75_percent`, `metrics_json`입니다. `report`와 `results`의 CSV/TSV 파일 및 `--stdout` 출력에 같은 형식을 적용합니다. `--view accuracy`와 `--view full` 모두 이 열 구성을 사용합니다. 실행 ID, 상태, 상세 설정은 `records.json`과 `summary.json`에서 확인할 수 있습니다.

`scheme`은 설정을 일정한 순서로 나열하며 구분자는 ` | `로 통일합니다. `99.99` 같은 percentile 값은 별도 숫자 열인 `percentile`에 기록하므로, 모델과 나머지 설정이 같으면 같은 scheme으로 묶을 수 있습니다. 보정 방식인 `calibration=percentile`은 scheme에 유지합니다. FP32, MinMax, Entropy의 `percentile` 셀은 비워 둡니다. Activation과 weight의 scale 정책을 각각 표시하므로 float, PoT nearest, PoT ceil, PoT MSE를 구별할 수 있습니다. 예시는 다음과 같습니다.

```text
scope=all | activation=asymmetric | weight=per_channel | activation_scale=pot_ceil | weight_scale=pot_ceil | calibration=percentile | rounding=AdaRound
```

FP32 행은 `scheme`에 `FP32`를 표시하고 `int8_accuracy`를 비웁니다. 미측정 후보도 예정된 scheme을 표시합니다. 실패하거나 없는 측정값은 정확도 셀을 비워 둡니다. INT8 측정값이 있어도 대응 FP32가 없으면 복원율은 비워 둡니다. 추가로 구별할 설정은 `scheme` 안에 함께 표시합니다. Task 정보가 없는 기록은 해당 셀을 비워 둡니다. `metrics_json`에는 원래 지표 값과 단위를 보존합니다.

복원율은 `recovery_percent = 100 × INT8 지표 / 대응 FP32 지표`로 계산합니다. JSON 상세 기록에는 `delta_pp = INT8(%) − FP32(%)`도 제공합니다. FP32와 INT8 기록은 모델과 평가 조건이 일치해야 합니다. 누락·실패를 0으로 취급하지 않습니다. FP32 지표가 0이면 복원율을 계산하지 않습니다. 100%를 넘는 복원율을 자르지 않습니다.

CSV는 UTF-8 BOM, TSV는 UTF-8을 사용합니다. `fp32_accuracy`, `int8_accuracy`, `_percent` 열은 78.5%를 `78.5`로 기록하는 숫자 열입니다. Excel에서는 숫자 서식을 유지합니다. 백분율 서식을 적용하면 표시값이 다시 100배가 됩니다. 상세 metric은 원래 단위를 유지합니다. COCO의 `AP_medium`은 중간 크기 객체이며 WIDER의 medium 난이도와 다릅니다. `report --format`은 선택한 형식만 쓰고 다른 기존 출력은 보존합니다. Metadata 명령인 `report`, `results`, `compare`는 추론하지 않습니다.

<details>
<summary>Catalog에 담긴 측정값 비교</summary>

`compare`는 모델, 양자화 설정, 측정값을 담은 **catalog/history 쌍**을 입력으로 받습니다. 측정값을 비교하고 Excel로 내보내려면 다음처럼 실행합니다.

```bash
uv pip install --python .venv/bin/python -e '.[report]'
opennpu-quant compare /path/to/catalog.json --output workspace/comparison --xlsx
```

설치 명령은 setup을 마친 저장소 루트에서 실행합니다. PATH에 `uv`가 없으면 `./.tools/uv`를 사용하세요. setup이 uv를 내려받은 경우 이 위치에 설치합니다.

기존 snapshot은 거부하므로 새 출력 폴더를 사용합니다. `--xlsx`가 없으면 CSV/TSV/JSON/Markdown 출력에는 Excel extra가 필요하지 않습니다. 입력 형식은 [comparison.py](src/opennpu_quant/comparison.py), 완전한 입력 생성 예시는 [test_comparison.py](tests/test_comparison.py)에서 확인할 수 있습니다. 일반 run의 `records.json`은 `results` 명령으로 내보냅니다.

Summary, Task_Summary, Model_Best, Core_Raw, Core_Measured, Complete10_Raw, All_History, Coverage를 제공합니다. 핵심 후보는 all scope의 Asym-PC/Sym-PT, float/PoT-ceil, Percentile 99.9/99.99/99.999, RTN/AdaRound 조합입니다.

Core_Raw, Core_Measured, Complete10_Raw, All_History도 CSV·TSV·Excel에서 같은 15열 형식을 사용합니다. Excel의 정확도·복원율 셀은 숫자로 저장하며 헤더 고정과 필터를 제공합니다. 상세 비교 기록은 `comparison.json`에 보존하고, 제외 사유는 Coverage에서도 확인할 수 있습니다.

`Core_Measured`는 `Core_Raw`에서 FP32 또는 INT8 정확도가 실제로 측정된 행만 추려 자동 생성합니다. 정확도가 0인 행도 포함하며 Excel을 열면 이 시트가 먼저 표시됩니다. 각 percentile의 측정은 개별 행으로 유지합니다. `model`과 `scheme`을 기준으로 묶어 `int8_accuracy` 최댓값을 구하면 percentile을 제외한 설정별 대표값을 얻을 수 있습니다.

</details>

<a id="troubleshooting"></a>
## 재사용·캐시와 문제 해결

| 상황 | 확인할 내용 |
|---|---|
| 알 수 없는 모델·조건 | `models`와 선택한 실험 설정을 확인합니다. 로컬 모델에는 `recipe.yaml`이 필요합니다. |
| 데이터 manifest 없음 | 실행과 같은 `--home`/`--data-dir`로 데이터를 준비합니다. |
| YAML·자료형·범위 오류 | 안내된 파일·행/열·키 경로를 확인합니다. Boolean의 따옴표를 제거하고 count에는 정수를 사용합니다. |
| `--local-config` 파일 없음 | 존재하는 YAML을 지정하거나 옵션을 빼고 기본 파일 탐색을 사용합니다. |
| 기존 run 불일치 | 과거 결과 보존을 위해 새 `--output`을 우선 사용합니다. `--force`는 완료 결과 재사용을 우회해 선택 결과를 재계산하며 보정 캐시·AdaRound checkpoint는 재사용할 수 있습니다. |
| 새 보정 통계 필요 | `--recalibrate`는 `calibration.source: compute`와 INT8 조건을 요구합니다. 원본 FP32 network에서 통계를 다시 모으고 의존하는 선택 INT8 결과를 재생성합니다. |
| feeds 거부 | `lambda: iter(feed_list)` 같은 반복 가능한 factory를 사용하고 값·순서를 유지합니다. |
| 런타임 버전 경고 | 검증한 ORT 버전은 1.20.2입니다. `--strict-versions`는 다른 버전을 거부합니다. |
| 비교 출력 폴더 존재 | 새 폴더를 선택하고 과거 snapshot을 덮어쓰지 않습니다. |

<details>
<summary>캐시 예산, identity와 재개</summary>

패키지 설정은 `<cache-dir>/inputs/<model>/<identity>/`에 **캐시 하나당** 2 GiB 한도로 calibration 입력을 저장합니다. 전처리한 network의 고정 shape·유한 FP32·batch-one feeds를 저장하며 validation 입력은 대상이 아닙니다. `--no-input-cache` 또는 `calibration.input_cache.enabled: false`로 비활성화합니다.

AdaRound 중간 activation 캐시는 별도이며 `--host-cache-gib`, `--gpu-window-mib`, `--no-cache`로 조정합니다. 예산은 보존할 데이터 한도이며 메모리 사전 할당이나 전체 디스크 할당량이 아닙니다. 공간을 확보할 때 사용 중이지 않은 입력 캐시 identity 폴더만 삭제하고 실행 중인 작업의 파일은 건드리지 않습니다.

재사용 시 모델 파일, 입력 순서, 전처리와 실행 설정을 자동으로 검사합니다. 이 조건이 바뀌면 새 출력 폴더를 사용합니다. 내용이 같은 파일의 경로 이동만으로 모델 identity는 바뀌지 않습니다.

AdaRound checkpoint는 검증된 완료 레이어 구간을 재개하고 중단된 레이어는 다시 시작합니다. `--force`는 모든 캐시·checkpoint 삭제 요청이 아닙니다. 저장한 segmentation prediction은 현재 evaluator가 사용한 label map입니다.

</details>

<a id="custom-models"></a>
## 사용자 모델

지원되는 임의 ONNX에는 자신의 feed factory·evaluator와 Python API를 사용합니다. CLI에 연결하려면 `<models-dir>/my_model/`에 `recipe.yaml`, `network.onnx`와 선택적 CPU host graph를 둡니다. [기본 recipe](src/opennpu_quant/resources/models/resnet18.yaml)를 바탕으로 입력 shape/layout, 데이터셋, task, metric, 전처리, decoder, 출처 정보와 그래프 경계를 수정합니다. 복사한 실험 설정에 모델과 필요한 모델별 percentile도 추가합니다. 로컬 recipe가 기본 제공 모델보다 우선합니다.

기본 decoder는 logits, YOLOv5, TF SSD, RetinaFace 640, segmentation입니다. 다른 출력에는 사용자 Python evaluator를 사용합니다. Network/host 경계를 명시하고 `all`은 network 내부의 지원 feature-data 연산을 양자화한다는 점에 유의하세요. CPU 정규화/NMS나 shape/index parameter는 대상이 아닙니다. 고정 구현 검사를 우회하려고 설명 필드를 바꾸지 않습니다.

사용자 `module:function` 전처리는 helper·전역 변수·파일 의존성을 신뢰성 있게 식별할 수 없습니다. Persistent 입력 캐시를 끄고 사용합니다. Fresh 실행은 가능하지만 자동 결과 재사용은 거부되며 충분한 전처리·protocol 증거가 없으면 FP32–INT8 복원율은 unmatched로 남습니다.

<a id="license"></a>
## 라이선스와 참고 자료

패키지 코드는 [Apache-2.0](LICENSE)이며 [NOTICE](NOTICE)와 [외부 출처 고지](THIRD_PARTY.md)를 유지합니다. 원본 모델 코드·checkpoint·데이터셋은 각 배포 조건을 따릅니다. ONNX 변환 자체가 재배포 권리를 부여하지는 않습니다.

- AdaRound: [Up or Down? Adaptive Rounding for Post-Training Quantization](https://proceedings.mlr.press/v119/nagel20a.html), ICML 2020.
- 데이터 출처: [ImageNet](https://huggingface.co/datasets/ILSVRC/imagenet-1k), [COCO](https://cocodataset.org/#download), [WIDER FACE](http://shuoyang1213.me/WIDERFACE/), [VOC2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/).
