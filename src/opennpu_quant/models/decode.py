import itertools
import math
import numpy as np


def retinaface_priors(height, width, min_sizes, steps):
    """Prior boxes [cx, cy, w, h] normalized by the input size, in the order of the network outputs."""
    priors = []
    for level_sizes, step in zip(min_sizes, steps):
        rows, columns = math.ceil(height / step), math.ceil(width / step)
        for row, column in itertools.product(range(rows), range(columns)):
            for min_size in level_sizes:
                priors.append(
                    [
                        (column + 0.5) * step / width,
                        (row + 0.5) * step / height,
                        min_size / width,
                        min_size / height,
                    ]
                )
    return np.asarray(priors, np.float32)


def greedy_nms(boxes, scores, iou_threshold):
    """Indices kept by score-ordered NMS; a box is dropped when IoU > threshold (torchvision rule)."""
    order = np.argsort(-scores, kind="stable")
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    kept = []
    while order.size:
        best = order[0]
        kept.append(best)
        rest = order[1:]
        left_top = np.maximum(boxes[best, :2], boxes[rest, :2])
        right_bottom = np.minimum(boxes[best, 2:], boxes[rest, 2:])
        overlap = np.clip(right_bottom - left_top, 0, None).prod(axis=1)
        iou = overlap / (areas[best] + areas[rest] - overlap)
        order = rest[iou <= iou_threshold]
    return np.asarray(kept, np.int64)


def decode_retinaface(loc, conf, priors, postprocessing, input_size, original_size):
    """Decode (loc, conf) of one image into face boxes [x1, y1, x2, y2] in original pixels.

    The adapter supplies thresholds and limits from the recipe's decode settings.
    Direct resizing requires independent x/y restoration before clipping.
    """
    variance = postprocessing["prior_box"]["variance"]
    centers = priors[:, :2] + loc[:, :2] * variance[0] * priors[:, 2:]
    sizes = priors[:, 2:] * np.exp(loc[:, 2:] * variance[1])
    boxes = np.concatenate([centers - sizes / 2, centers + sizes / 2], axis=1)
    input_width, input_height = input_size
    boxes = boxes * np.array(
        [input_width, input_height, input_width, input_height], np.float32
    )
    scores = conf[:, 1]
    candidates = scores > postprocessing["score_threshold"]
    boxes, scores = boxes[candidates], scores[candidates]
    kept = greedy_nms(boxes, scores, postprocessing["nms_iou"])[
        : postprocessing["max_detections"]
    ]
    boxes, scores = boxes[kept].astype(np.float64), scores[kept]
    width, height = original_size
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] * (width / input_width)).clip(0, width)
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] * (height / input_height)).clip(0, height)
    return boxes, scores


def wider_face_record(boxes, scores):
    """Detections as the official result files store them: integer x, y, w, h and the score."""
    rows = []
    for (x1, y1, x2, y2), score in zip(boxes.tolist(), scores.tolist()):
        rows.append(
            [int(x1), int(y1), int(x2) - int(x1), int(y2) - int(y1), float(score)]
        )
    return rows


def segmentation_labels(logits, valid_size, original_size):
    """NCHW logits of the padded input -> label map of the original image (crop, bilinear resize, argmax)."""
    import cv2

    valid_width, valid_height = valid_size
    cropped = np.ascontiguousarray(
        logits[0, :, :valid_height, :valid_width].transpose(1, 2, 0)
    )
    resized = cv2.resize(cropped, tuple(original_size), interpolation=cv2.INTER_LINEAR)
    return resized.argmax(axis=2).astype(np.uint8)


import cv2


def class_nms(boxes, scores, labels, threshold, max_det):
    """Class-aware greedy NMS; no class offsets that assume bounded coordinates."""
    kept = []
    for label in np.unique(labels):
        idx = np.flatnonzero(labels == label)
        b = boxes[idx].copy()
        b[:, 2:] -= b[:, :2]
        selected = cv2.dnn.NMSBoxes(b.tolist(), scores[idx].tolist(), 0.0, threshold)
        kept.extend(idx[np.asarray(selected, dtype=np.int64).reshape(-1)].tolist())
    kept = np.asarray(kept, dtype=np.int64)
    return kept[np.argsort(-scores[kept], kind="stable")[:max_det]]


def yolo(predictions, metadata, category_ids, policy):
    p = predictions[0]
    if p.ndim != 2 or p.shape[1] != 85 or not np.isfinite(p).all():
        raise ValueError("Invalid decoded YOLO predictions")
    scores = p[:, 5:] * p[:, 4:5]
    if policy["multi_label"]:
        candidates, labels = np.nonzero(scores > policy["confidence_threshold"])
        confidence = scores[candidates, labels]
    else:
        labels = scores.argmax(axis=1)
        confidence = scores[np.arange(len(scores)), labels]
        candidates = np.flatnonzero(confidence > policy["confidence_threshold"])
        labels, confidence = labels[candidates], confidence[candidates]
    if not len(candidates):
        return []
    order = np.argsort(-confidence, kind="stable")[: policy["max_nms_candidates"]]
    candidates, labels, confidence = candidates[order], labels[order], confidence[order]
    xywh = p[candidates, :4]
    boxes = np.concatenate(
        (xywh[:, :2] - xywh[:, 2:] / 2, xywh[:, :2] + xywh[:, 2:] / 2), axis=1
    )
    keep = class_nms(
        boxes, confidence, labels, policy["nms_iou_threshold"], policy["max_detections"]
    )
    boxes, labels, confidence = boxes[keep], labels[keep], confidence[keep]
    boxes[:, [0, 2]] = (
        (boxes[:, [0, 2]] - metadata["pad"][0]) / metadata["scale"][0]
    ).clip(0, metadata["original_size"][0])
    boxes[:, [1, 3]] = (
        (boxes[:, [1, 3]] - metadata["pad"][1]) / metadata["scale"][1]
    ).clip(0, metadata["original_size"][1])
    boxes[:, 2:] -= boxes[:, :2]
    return [
        dict(category_id=category_ids[int(label)], bbox=b.tolist(), score=float(score))
        for b, label, score in zip(boxes, labels, confidence)
        if b[2] > 0 and b[3] > 0
    ]
