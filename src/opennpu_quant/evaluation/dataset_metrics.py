import numpy as np

WIDER_SETTINGS = ("easy", "medium", "hard")
WIDER_SCORE_THRESHOLDS = 1000
WIDER_IOU_THRESHOLD = 0.5


def _overlaps_plus_one(detections, ground_truth):
    """IoU matrix with the +1 pixel convention of the official evaluation code (xyxy inputs)."""
    detection_area = (detections[:, 2] - detections[:, 0] + 1) * (
        detections[:, 3] - detections[:, 1] + 1
    )
    truth_area = (ground_truth[:, 2] - ground_truth[:, 0] + 1) * (
        ground_truth[:, 3] - ground_truth[:, 1] + 1
    )
    width = (
        np.minimum(detections[:, None, 2], ground_truth[None, :, 2])
        - np.maximum(detections[:, None, 0], ground_truth[None, :, 0])
        + 1
    ).clip(0, None)
    height = (
        np.minimum(detections[:, None, 3], ground_truth[None, :, 3])
        - np.maximum(detections[:, None, 1], ground_truth[None, :, 1])
        + 1
    ).clip(0, None)
    intersection = width * height
    return intersection / (detection_area[:, None] + truth_area[None, :] - intersection)


def _voc_ap(recall, precision):
    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([0.0], precision, [0.0]))
    for index in range(precision.size - 1, 0, -1):
        precision[index - 1] = max(precision[index - 1], precision[index])
    changed = np.where(recall[1:] != recall[:-1])[0]
    return float(
        np.sum((recall[changed + 1] - recall[changed]) * precision[changed + 1])
    )


def wider_face_ap(detections_by_image, ground_truth):
    """Official WIDER FACE AP for the easy, medium and hard settings.

    detections_by_image: {image_id: [[x, y, w, h, score], ...]}
    ground_truth:        [{image_id, boxes: [[x, y, w, h], ...], keep: {easy: [...], ...}}, ...]
                         with `keep` holding the 0-based indices of the faces counted per setting
    Every ground-truth image must have an entry (possibly empty); scores are min-max normalized
    over the whole set exactly as the reference implementation does.
    """
    missing = [
        item["image_id"]
        for item in ground_truth
        if item["image_id"] not in detections_by_image
    ]
    if missing:
        raise ValueError(
            "WIDER FACE images without a detection record: " + str(missing[:5])
        )
    all_scores = [row[4] for rows in detections_by_image.values() for row in rows]
    if not all_scores:
        raise ValueError("No WIDER FACE detections at all")
    lowest, highest = min(all_scores), max(all_scores)
    span = highest - lowest
    if span <= 0:
        raise ValueError("WIDER FACE scores cannot be normalized")
    thresholds = 1 - (np.arange(WIDER_SCORE_THRESHOLDS) + 1) / WIDER_SCORE_THRESHOLDS

    curves = {
        setting: np.zeros((WIDER_SCORE_THRESHOLDS, 2)) for setting in WIDER_SETTINGS
    }
    faces = dict.fromkeys(WIDER_SETTINGS, 0)
    for item in ground_truth:
        rows = np.asarray(detections_by_image[item["image_id"]], np.float64).reshape(
            -1, 5
        )
        rows = rows[np.argsort(-rows[:, 4], kind="stable")]
        truth = np.asarray(item["boxes"], np.float64).reshape(-1, 4)
        if not len(truth) or not len(rows):
            for setting in WIDER_SETTINGS:
                faces[setting] += len(item["keep"][setting])
            continue
        scores = (rows[:, 4] - lowest) / span
        detection_boxes = np.column_stack(
            [rows[:, 0], rows[:, 1], rows[:, 0] + rows[:, 2], rows[:, 1] + rows[:, 3]]
        )
        truth_boxes = np.column_stack(
            [
                truth[:, 0],
                truth[:, 1],
                truth[:, 0] + truth[:, 2],
                truth[:, 1] + truth[:, 3],
            ]
        )
        overlaps = _overlaps_plus_one(detection_boxes, truth_boxes)
        best_truth = overlaps.argmax(axis=1)
        best_overlap = overlaps.max(axis=1)
        # Number of detections with a normalized score >= each threshold (scores are descending).
        counted = np.searchsorted(-scores, -thresholds, side="right")
        for setting in WIDER_SETTINGS:
            keep = item["keep"][setting]
            faces[setting] += len(keep)
            counts_face = np.zeros(len(truth), bool)
            counts_face[list(keep)] = (
                True  # an image without counted faces still adds false positives
            )
            matched = np.zeros(len(truth), bool)
            proposals = np.ones(len(rows), bool)
            recalled = np.zeros(len(rows))
            recalled_so_far = 0
            for index in range(len(rows)):
                if best_overlap[index] >= WIDER_IOU_THRESHOLD:
                    target = best_truth[index]
                    if not counts_face[target]:
                        proposals[index] = (
                            False  # matches an ignored face: not a proposal
                        )
                    elif not matched[target]:
                        matched[target] = True
                        recalled_so_far += 1
                recalled[index] = recalled_so_far
            cumulative_proposals = np.cumsum(proposals)
            active = counted > 0
            last = counted[active] - 1
            curves[setting][active, 0] += cumulative_proposals[last]
            curves[setting][active, 1] += recalled[last]

    result = {}
    for setting in WIDER_SETTINGS:
        proposals, recalled = curves[setting][:, 0], curves[setting][:, 1]
        # No proposal at a threshold means precision 0 there (the reference would divide by zero).
        precision = np.divide(
            recalled, proposals, out=np.zeros_like(recalled), where=proposals > 0
        )
        recall = recalled / faces[setting]
        result["AP_" + setting] = _voc_ap(recall, precision)
    return result
