import numpy as np
from PIL import Image


def synthetic_image(index, width, height, seed=20260916):
    if index == 0:
        values = np.full((height, width, 3), 127, dtype=np.uint8)
    elif index == 1:
        x = np.linspace(0, 255, width, dtype=np.float32)[None, :]
        y = np.linspace(0, 255, height, dtype=np.float32)[:, None]
        values = np.stack(
            [
                np.broadcast_to(x, (height, width)),
                np.broadcast_to(y, (height, width)),
                (x + y) / 2,
            ],
            axis=-1,
        ).astype(np.uint8)
    else:
        values = np.random.default_rng(seed + index).integers(
            0, 256, (height, width, 3), dtype=np.uint8
        )
    return Image.fromarray(values, "RGB")
