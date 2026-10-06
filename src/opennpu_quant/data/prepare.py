from pathlib import Path
from importlib.resources import files
import random, re
from .._io import atomic_json, read_json, sha256

COUNTS = {"imagenet": 50000, "coco": 5000, "widerface": 3226, "voc": 1449}


def identifiers(name):
    return [
        s.split()
        for s in files("opennpu_quant.resources.splits")
        .joinpath(name + ".txt")
        .read_text()
        .splitlines()
        if s and not s.startswith("#")
    ]


def prepare(name, root, destination, *, source="folder", download=False):
    root = Path(root).expanduser().resolve()
    destination = Path(destination)
    if name not in COUNTS:
        raise ValueError("Unknown dataset")
    if download:
        if name != "imagenet" or source != "hf":
            raise ValueError(
                "Automatic download is currently available for ImageNet HF; use documented official archives for this dataset"
            )
        download_imagenet(root)
    annotations = {}
    # Dataset byte equivalence requires independent verification, not matching names.
    reference = False
    if name == "imagenet":
        if source == "hf" or (root / "validation_images").is_dir():
            validation = []
            for path in sorted((root / "validation_images").glob("*.jpg")):
                match = re.fullmatch(r"img_(\d+)_label_(\d+)\.jpg", path.name)
                if match is None:
                    raise ValueError("Invalid HF ImageNet filename")
                validation.append(
                    dict(
                        id=int(match[1]),
                        path=str(path.relative_to(root)),
                        target=int(match[2]),
                    )
                )
            train_dir = (
                "train_images"
                if (root / "train_images").is_dir()
                else "calibration_images"
            )
            calibration = [
                dict(
                    id=int(i),
                    path=f"{train_dir}/img_{int(i):08d}_label_{int(label)}.jpg",
                    target=int(label),
                )
                for i, label in identifiers(name)
            ]
            if [v["id"] for v in validation] != list(range(COUNTS[name])) or any(
                not 0 <= v["target"] < 1000 for v in validation
            ):
                raise ValueError(
                    "HF validation IDs/labels are incomplete or out of range"
                )
        else:
            train = root / "train"
            val = root / ("val" if (root / "val").exists() else "validation")
            classes = sorted(p.name for p in train.iterdir() if p.is_dir())
            if (
                len(classes) != 1000
                or any(not re.fullmatch(r"n[0-9]{8}", c) for c in classes)
                or classes != sorted(p.name for p in val.iterdir() if p.is_dir())
            ):
                raise ValueError(
                    "ImageFolder requires matching 1000 class directories in train and val"
                )

            def listing(folder):
                return [
                    dict(
                        id=f"{cls}/{p.name}",
                        path=str(p.relative_to(root)),
                        target=label,
                    )
                    for label, cls in enumerate(classes)
                    for p in sorted((folder / cls).iterdir())
                    if p.suffix.lower() in (".jpg", ".jpeg", ".png")
                ]

            validation = listing(val)
            training = listing(train)
            calibration = random.Random(20260917).sample(training, 1000)
    elif name == "coco":
        annotation = "annotations/instances_val2017.json"
        doc = read_json(root / annotation)
        annotations["coco"] = annotation
        validation = [
            dict(id=row["id"], path="val2017/" + row["file_name"])
            for row in sorted(doc["images"], key=lambda x: x["id"])
        ]
        calibration = [
            dict(id=int(row[0]), path=f"train2017/{int(row[0]):012d}.jpg")
            for row in identifiers(name)
        ]
    elif name == "voc":
        names = (root / "ImageSets/Segmentation/val.txt").read_text().split()
        validation = [
            dict(id=i, path=f"JPEGImages/{i}.jpg", target=f"SegmentationClass/{i}.png")
            for i in names
        ]
        calibration = [
            dict(id=i[0], path=f"JPEGImages/{i[0]}.jpg") for i in identifiers(name)
        ]
        annotations["split"] = "ImageSets/Segmentation/val.txt"
    else:
        # Conversion output lives next to this manifest; keep source dataset untouched.
        truth = wider_ground_truth(root)
        target = destination.parent / "wider_ground_truth.json"
        atomic_json(target, truth)
        validation = [
            dict(id=i, path=row["path"], truth=row) for i, row in enumerate(truth)
        ]
        # Event directory names contain spaces; restore tokens before making IDs.
        calibration = [
            dict(id=" ".join(i), path="WIDER_train/images/" + " ".join(i))
            for i in identifiers(name)
        ]
    if len(validation) != COUNTS[name]:
        raise ValueError(
            f"Expected {COUNTS[name]} validation images, found {len(validation)}"
        )
    seen = {}
    for split, rows in [("calibration", calibration), ("validation", validation)]:
        if len({str(r["id"]) for r in rows}) != len(rows):
            raise ValueError("Duplicate image IDs")
        for row in rows:
            path = (root / row["path"]).resolve()
            if not path.is_relative_to(root) or not path.is_file():
                raise ValueError("Missing/unsafe image: " + str(path))
            # Detect leakage by content, including renamed copies.
            digest = sha256(path)
            if digest in seen and seen[digest] != split:
                raise ValueError("Calibration/validation content overlap")
            seen[digest] = split
            if (
                isinstance(row.get("target"), str)
                and not (root / row["target"]).is_file()
            ):
                raise ValueError("Missing label map")
    result = dict(
        schema_version=1,
        dataset=name,
        root=str(root),
        source=source,
        reference_bytes_verified=reference,
        expected_samples=COUNTS[name],
        annotations=annotations,
        calibration=calibration,
        validation=validation,
    )
    atomic_json(destination, result)
    return result


def wider_ground_truth(root):
    from scipy.io import loadmat

    truth = root / "eval_tools/ground_truth"
    face = loadmat(truth / "wider_face_val.mat")
    kept = {
        s: loadmat(truth / f"wider_{s}_val.mat")["gt_list"]
        for s in ("easy", "medium", "hard")
    }
    rows = []
    for ei, event in enumerate(face["event_list"]):
        event_name = str(event[0][0])
        for fi, item in enumerate(face["file_list"][ei][0]):
            file = str(item[0][0])
            boxes = face["face_bbx_list"][ei][0][fi][0].astype(float).tolist()
            keep = {
                s: (kept[s][ei][0][fi][0].reshape(-1).astype(int) - 1).tolist()
                for s in kept
            }
            if any(i < 0 or i >= len(boxes) for v in keep.values() for i in v):
                raise ValueError("Invalid WIDER ground truth indices")
            rows.append(
                dict(
                    image_id=len(rows),
                    event=event_name,
                    file=file,
                    path=f"WIDER_val/images/{event_name}/{file}.jpg",
                    boxes=boxes,
                    keep=keep,
                )
            )
    totals = {s: sum(len(row["keep"][s]) for row in rows) for s in kept}
    if len(rows) != 3226 or totals != dict(easy=7211, medium=13319, hard=31958):
        raise ValueError("Unexpected WIDER ground truth")
    return rows


def download_imagenet(root):
    try:
        from datasets import load_dataset
    except ImportError:
        raise RuntimeError("Install the data extra for HF preparation") from None
    selected = {int(i): int(label) for i, label in identifiers("imagenet")}
    for split in ("train", "validation"):
        destination = root / (split + "_images")
        destination.mkdir(parents=True, exist_ok=True)
        dataset = load_dataset(
            "ILSVRC/imagenet-1k",
            revision="49e2ee26f3810fb5a7536bbf732a7b07389a47b5",
            split=split,
            streaming=True,
        )
        for index, row in enumerate(dataset):
            if split == "train" and index > max(selected):
                break
            if split == "train" and index not in selected:
                continue
            if split == "train" and int(row["label"]) != selected[index]:
                raise ValueError("HF labels differ from fixed split")
            path = destination / f"img_{index:08d}_label_{int(row['label'])}.jpg"
            if path.exists():
                continue
            temporary = path.with_suffix(".pending")
            row["image"].convert("RGB").save(temporary, format="JPEG", quality=95)
            temporary.replace(path)
