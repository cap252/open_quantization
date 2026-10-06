def evaluate_predictions(ground_truth, predictions, image_ids):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    if predictions:
        detected = ground_truth.loadRes(predictions)
    else:
        detected = COCO()
        detected.dataset = dict(
            images=ground_truth.dataset["images"],
            categories=ground_truth.dataset["categories"],
            annotations=[],
        )
        detected.createIndex()
    evaluator = COCOeval(ground_truth, detected, "bbox")
    evaluator.params.imgIds = list(image_ids)
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    names = [
        "AP",
        "AP50",
        "AP75",
        "AP_small",
        "AP_medium",
        "AP_large",
        "AR1",
        "AR10",
        "AR100",
        "AR_small",
        "AR_medium",
        "AR_large",
    ]
    metrics = {name: float(v) for name, v in zip(names, evaluator.stats)}
    per_class = []
    for i, category in enumerate(evaluator.params.catIds):
        precision = evaluator.eval["precision"][:, :, i, 0, -1]
        values = precision[precision > -1]
        per_class.append(
            dict(
                category_id=int(category),
                name=ground_truth.cats[category]["name"],
                AP=float(values.mean()) if values.size else None,
            )
        )
    return dict(
        metrics=metrics,
        per_class=per_class,
        image_count=len(image_ids),
        prediction_count=len(predictions),
        iou_thresholds=evaluator.params.iouThrs.tolist(),
        max_detections=evaluator.params.maxDets,
        evaluator="pycocotools.cocoeval.COCOeval bbox; standard 101 recall thresholds and crowd/area handling",
    )


class CocoMetric:
    def __init__(self, annotation, *, iou_type="bbox"):
        if iou_type not in ("bbox", "keypoints"):
            raise ValueError("COCO bbox or keypoints required")
        self.annotation = str(annotation)
        self.iou_type = iou_type
        self.reset()

    def reset(self):
        self.predictions = []
        self.image_ids = []
        self.details = {}

    def update(self, predictions, sample):
        image_id = int(sample.sample_id)
        self.image_ids.append(image_id)
        self.predictions.extend((dict(image_id=image_id, **row) for row in predictions))

    def finalize(self):
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

        ground_truth = COCO(self.annotation)
        if self.iou_type == "bbox":
            result = evaluate_predictions(
                ground_truth, self.predictions, self.image_ids
            )
            self.details = result
            return result["metrics"]
        if self.predictions:
            detected = ground_truth.loadRes(self.predictions)
        else:
            detected = COCO()
            detected.dataset = dict(
                images=ground_truth.dataset["images"],
                categories=ground_truth.dataset["categories"],
                annotations=[],
            )
            detected.createIndex()
        evaluator = COCOeval(ground_truth, detected, "keypoints")
        evaluator.params.imgIds = list(self.image_ids)
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
        names = [
            "AP",
            "AP50",
            "AP75",
            "AP_medium",
            "AP_large",
            "AR",
            "AR50",
            "AR75",
            "AR_medium",
            "AR_large",
        ]
        return dict(zip(names, map(float, evaluator.stats)))
