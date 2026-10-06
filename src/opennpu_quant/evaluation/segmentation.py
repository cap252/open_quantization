import numpy as np


class _ConfusionMatrix:
    def __init__(self, classes, ignore_label=255):
        if type(classes) is not int or classes < 1:
            raise ValueError("classes must be a positive integer")
        self.classes, self.ignore_label = classes, ignore_label
        self.matrix = np.zeros((classes, classes), np.int64)

    def add(self, predicted, truth):
        predicted, truth = np.asarray(predicted), np.asarray(truth)
        if predicted.shape != truth.shape:
            raise ValueError(
                "Segmentation coordinates/size must be restored by the adapter"
            )
        valid = truth != self.ignore_label
        target, prediction = truth[valid], predicted[valid]
        for values in (target, prediction):
            if (
                not np.isfinite(values).all()
                or np.any(values < 0)
                or np.any(values >= self.classes)
                or np.any(values != np.floor(values))
            ):
                raise ValueError("Invalid segmentation labels")
        self.matrix += np.bincount(
            self.classes * target.astype(np.int64) + prediction.astype(np.int64),
            minlength=self.classes**2,
        ).reshape(self.classes, self.classes)
        return int(valid.sum()), int((prediction == target).sum())

    def metrics(self):
        intersection = np.diag(self.matrix).astype(np.float64)
        union = self.matrix.sum(0) + self.matrix.sum(1) - intersection
        present = union > 0
        if not present.any():
            raise ValueError("No evaluable pixels")
        iou = np.divide(
            intersection, union, out=np.zeros_like(intersection), where=present
        )
        return dict(
            mIoU=float(iou[present].mean()),
            pixel_accuracy=float(intersection.sum() / self.matrix.sum()),
            classes_evaluated=int(present.sum()),
            per_class_iou=[
                float(value) if occurs else None for value, occurs in zip(iou, present)
            ],
        )
