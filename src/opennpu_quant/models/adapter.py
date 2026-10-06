import numpy as np
from ..evaluation.metrics import TopKClassification, SegmentationEvaluator
from ..evaluation.coco import CocoMetric
from ..evaluation.dataset_metrics import wider_face_ap
from .decode import (
    yolo,
    retinaface_priors,
    decode_retinaface,
    wider_face_record,
    segmentation_labels,
)


class ModelEvaluator:
    def __init__(self, bundle, dataset):
        self.bundle, self.dataset = bundle, dataset
        self.recipe = bundle.spec.recipe
        self.kind = self.recipe["decode"]["kind"]
        if self.kind == "logits":
            self.metric = TopKClassification(
                output_name=self.recipe["decode"]["output"]
            )
        elif self.kind == "segmentation_labels":
            self.metric = SegmentationEvaluator(output_name="labels")
        elif self.kind in ("yolo", "tf_od_api"):
            self.metric = CocoMetric(dataset.annotations["coco"])
            from .._io import read_json

            self.categories = sorted(
                x["id"] for x in read_json(dataset.annotations["coco"])["categories"]
            )
        elif self.kind == "retinaface":
            self.metric = None
            cfg = self.recipe["decode"]
            w, h = self.recipe["preprocess"]["size"]
            self.priors = retinaface_priors(
                h, w, cfg["priors"]["min_sizes"], cfg["priors"]["steps"]
            )
        else:
            raise ValueError("Unknown decoder: " + self.kind)
        self.reset()

    def reset(self):
        if self.metric:
            self.metric.reset()
        self.detections = {}
        self.truth = []
        self.prediction = None

    def update(self, outputs, sample):
        out = self.bundle.finish(outputs, sample.metadata)
        cfg = self.recipe["decode"]
        meta = sample.metadata
        if any(not np.isfinite(v).all() for v in out.values()):
            raise ValueError("Nonfinite host output")
        if self.kind == "logits":
            self.metric.update(out, sample)
            self.prediction = self.metric.prediction
            return
        if self.kind == "segmentation_labels":
            labels = segmentation_labels(
                out[cfg["output"]], meta["valid_size"], meta["original_size"]
            )
            self.metric.update({"labels": labels}, sample)
            self.prediction = {"labels": labels.tolist()}
            return
        if self.kind == "retinaface":
            policy = dict(
                prior_box=dict(variance=cfg["variance"]),
                score_threshold=cfg["score_threshold"],
                nms_iou=cfg["nms_iou"],
                max_detections=cfg["max_detections"],
            )
            boxes, scores = decode_retinaface(
                out["loc"][0],
                out["conf"][0],
                self.priors,
                policy,
                self.recipe["preprocess"]["size"],
                meta["original_size"],
            )
            if not np.isfinite(boxes).all() or not np.isfinite(scores).all():
                raise ValueError("Nonfinite decoded face")
            value = wider_face_record(boxes, scores)
            self.detections[int(sample.sample_id)] = value
            self.truth.append(meta["dataset_row"]["truth"])
            self.prediction = value
            return
        if self.kind == "yolo":
            policy = dict(
                confidence_threshold=cfg["confidence_threshold"],
                multi_label=cfg["multi_label"],
                max_nms_candidates=cfg["max_nms_candidates"],
                nms_iou_threshold=cfg["nms"]["iou"],
                max_detections=cfg["max_detections"],
            )
            rows = yolo(out[cfg["output"]], meta, self.categories, policy)
        else:
            count = int(out[cfg["count"]][0])
            width, height = meta["original_size"]
            if not 0 <= count <= out[cfg["boxes"]].shape[1]:
                raise ValueError("Invalid SSD detection count")
            boxes = out[cfg["boxes"]][0, :count][:, [1, 0, 3, 2]].copy() * np.array(
                [width, height, width, height]
            )
            scores = out[cfg["scores"]][0, :count]
            raw = out[cfg["classes"]][0, :count]
            if not np.array_equal(raw, raw.astype(int)):
                raise ValueError("Noninteger SSD class")
            labels = raw.astype(int)
            if not set(labels.tolist()) <= set(self.categories):
                raise ValueError("Invalid sparse COCO category ID")
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, width)
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, height)
            boxes[:, 2:] -= boxes[:, :2]
            rows = [
                dict(category_id=int(label), bbox=box.tolist(), score=float(score))
                for box, score, label in zip(boxes, scores, labels)
                if box[2] > 0 and box[3] > 0
            ]
        self.metric.update(rows, sample)
        self.prediction = rows

    @property
    def metric_details(self):
        return getattr(
            self.metric, "metric_details", getattr(self.metric, "details", {})
        )

    @property
    def metric_units(self):
        return getattr(self.metric, "metric_units", {})

    def finalize(self):
        if self.kind == "retinaface":
            return wider_face_ap(self.detections, self.truth)
        return self.metric.finalize()
