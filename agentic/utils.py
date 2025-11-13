import numpy as np
from pathlib import Path
import io
import base64
from typing import Any

from PIL import Image

def crop_central_component(img, threshold=8, margin=6, search_radius=200):
    """
    Crops to the connected component of non-black pixels nearest the image center.
    - threshold: <= threshold in all RGB channels is treated as black background
    - margin: add a few pixels of border back after cropping
    - search_radius: how far from center we'll look for the first non-black pixel
    """
    arr = np.asarray(img)  # works for np.array or PIL.Image
    if arr.ndim == 2:                       # grayscale -> fake RGB
        arr = np.repeat(arr[..., None], 3, axis=2)
    mask = (arr > threshold).any(axis=2)    # non-black

    H, W = mask.shape
    cy, cx = H // 2, W // 2

    ys, xs = np.where(mask)
    if ys.size == 0:
        return arr  # nothing to crop

    d2 = (ys - cy)**2 + (xs - cx)**2
    # pick a seed near center (inside search_radius if possible)
    order = np.argsort(d2)
    seed_idx = order[0]
    for i in order:
        if d2[i] <= search_radius**2:
            seed_idx = i
            break
    sy, sx = int(ys[seed_idx]), int(xs[seed_idx])

    # BFS flood fill to get the connected component containing (sy, sx)
    from collections import deque
    q = deque([(sy, sx)])
    visited = np.zeros_like(mask, dtype=bool)
    visited[sy, sx] = True

    y0 = y1 = sy
    x0 = x1 = sx
    while q:
        y, x = q.popleft()
        y0 = min(y0, y); y1 = max(y1, y)
        x0 = min(x0, x); x1 = max(x1, x)
        for ny, nx in ((y-1,x),(y+1,x),(y,x-1),(y,x+1)):
            if 0 <= ny < H and 0 <= nx < W and mask[ny, nx] and not visited[ny, nx]:
                visited[ny, nx] = True
                q.append((ny, nx))

    # add a small margin and clamp
    y0 = max(y0 - margin, 0); y1 = min(y1 + margin, H-1)
    x0 = max(x0 - margin, 0); x1 = min(x1 + margin, W-1)
    return arr[y0:y1+1, x0:x1+1]




def image_to_base64(image_data: Path | str | np.ndarray):
    if isinstance(image_data, (str, Path)):
        image_path = Path(image_data)
        image = Image.open(image_path)
        format = image_path.suffix
    elif isinstance(image_data, np.ndarray):
        image = Image.fromarray(image_data.astype('uint8'), 'RGB')
        format = "png"
    else:
        raise RuntimeError(f"Image data type not supported: {type(image_data)}")

    image_bytes = io.BytesIO()
    image.save(image_bytes, format=format)
    base64_image = base64.b64encode(image_bytes.getvalue()).decode("utf-8")
    return base64_image


def extract_xml_key(data: str, key: str) -> str | None:
    if f"<{key}>" in data:
        data = data.split(f"<{key}>")[1]
        if f"</{key}>" in data:
            return data.split(f"</{key}>")[0]
        

def extract_xml_kv(data: str, keys: list[str]) -> dict[str, Any]:
    extracted = {}
    for k in keys:
        if (v := extract_xml_key(data, k)) is not None:
            extracted[k] = v
    return extracted

