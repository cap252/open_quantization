import numpy as np
from PIL import Image


def preprocess(image, settings, layout="NCHW"):
    import cv2

    image = image.convert("RGB")
    width, height = image.size
    meta = dict(original_size=[width, height])
    kind = settings["kind"]
    if kind == "center_crop":
        short, crop = settings["resize"], settings["crop"]
        resized = (
            (short, int(short * height / width))
            if width <= height
            else (int(short * width / height), short)
        )
        method = {
            "bilinear": Image.Resampling.BILINEAR,
            "bicubic": Image.Resampling.BICUBIC,
        }[settings["interpolation"]]
        image = image.resize(resized, method)
        left, top = (
            int(round((resized[0] - crop) / 2)),
            int(round((resized[1] - crop) / 2)),
        )
        image = image.crop((left, top, left + crop, top + crop))
        x = np.asarray(image, np.float32).transpose(2, 0, 1) / np.float32(255)
        x = (x - np.asarray(settings["mean"], np.float32)[:, None, None]) / np.asarray(
            settings["std"], np.float32
        )[:, None, None]
        return np.ascontiguousarray(x[None]), meta
    if kind == "letterbox":
        size = settings["size"]
        ratio = min(size / width, size / height)
        resized = (round(width * ratio), round(height * ratio))
        pixels = cv2.resize(np.asarray(image), resized, interpolation=cv2.INTER_LINEAR)
        dw, dh = (size - resized[0]) / 2, (size - resized[1]) / 2
        left, right, top, bottom = (
            round(dw - 0.1),
            round(dw + 0.1),
            round(dh - 0.1),
            round(dh + 0.1),
        )
        pad = settings["pad_value"]
        pixels = cv2.copyMakeBorder(
            pixels, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(pad, pad, pad)
        )
        meta.update(scale=[ratio, ratio], pad=[left, top])
        return np.ascontiguousarray(
            pixels.transpose(2, 0, 1)[None], dtype=np.float32
        ) / np.float32(255), meta
    if kind in ("resize_uint8", "resize_bgr_mean"):
        pixels = cv2.resize(
            np.asarray(image), tuple(settings["size"]), interpolation=cv2.INTER_LINEAR
        )
        if kind == "resize_uint8":
            return pixels[None], meta
        pixels = pixels.astype(np.float32)[..., ::-1] - np.array(
            settings["mean_bgr"], np.float32
        )
        return np.ascontiguousarray(
            pixels.transpose(2, 0, 1)[None], dtype=np.float32
        ), meta
    if kind == "longer_side_pad":
        size = settings["size"]
        ratio = size / max(width, height)
        valid = (max(1, round(width * ratio)), max(1, round(height * ratio)))
        pixels = np.asarray(
            image.resize(valid, Image.Resampling.BILINEAR), dtype=np.float32
        )
        normalized = (pixels - np.asarray(settings["mean"], np.float32)) / np.asarray(
            settings["std"], np.float32
        )
        canvas = np.zeros((size, size, 3), np.float32)
        canvas[: valid[1], : valid[0]] = normalized
        if layout == "NCHW":
            canvas = canvas.transpose(2, 0, 1)
        meta["valid_size"] = valid
        return np.ascontiguousarray(canvas[None]), meta
    if ":" in kind:
        from importlib import import_module

        module, name = kind.rsplit(":", 1)
        return getattr(import_module(module), name)(image, settings, layout)
    raise ValueError("Unknown preprocessing kind: " + kind)
