"""Spine segmentation with SAM 3.1 (Meta API). One text prompt ('book spine') returns one tight box per spine:
40/40 on a dense shelf photo in ~4 s, where the vision model's boxes drift onto the neighbouring spine.

The answer is plain text, one line per frame ('<0f>' for a still image), then comma-separated records:
    0<|box;x1=661;y1=48;x2=734;y2=388;w=1080;h=810|><|mask;x=0;y=0;data=341,74,!!!!!rDb…|>
Coordinates are pixels of the image we sent; x2/y2 are inclusive (the mask is (y2-y1+1) x (x2-x1+1)).
The mask payload is a compressed encoding Meta does not document; boxes are all the pipeline needs, so masks are
kept raw (parse(masks=True)) and never decoded.
"""
import base64
import re
import time

import httpx

from pipeline import metrics, settings

RECORD = re.compile(r"(\d+)<\|box;x1=([\d.]+);y1=([\d.]+);x2=([\d.]+);y2=([\d.]+);w=(\d+);h=(\d+)\|>"
                    r"(?:<\|mask;x=(\d+);y=(\d+);data=([^|]*)\|>)?")


def available():
    return settings.SPINE_SOURCE == "sam" and bool(settings.META_API_KEY)


def parse(text, masks=False):
    """Boxes from the response text. A record that does not match, or has an empty box, is skipped."""
    out = []
    for m in RECORD.finditer(text or ""):
        x1, y1, x2, y2 = (round(float(v)) for v in m.group(2, 3, 4, 5))
        w, h = int(m.group(6)), int(m.group(7))
        if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
            continue
        d = {"id": int(m.group(1)), "box": [x1, y1, x2, y2], "w": w, "h": h}
        if masks and m.group(10):
            d["mask"] = {"x": int(m.group(8)), "y": int(m.group(9)), "data": m.group(10)}
        out.append(d)
    return out


def segment(jpeg, prompt=None):
    """One image -> spine boxes. Raises on any API failure: the caller falls back to the vision model's spines."""
    if not settings.META_API_KEY:
        raise RuntimeError("META_API_KEY not set")
    body = {"model": settings.SAM_MODEL, "metadata": {"mask_encoding": "one_bit"},
            "input": [{"type": "message", "role": "user", "content": [
                {"type": "input_text", "text": prompt or settings.SAM_PROMPT},
                {"type": "input_image", "image_url": "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()}]}]}
    t0 = time.perf_counter()
    for attempt in range(2):  # one retry on rate limit / server error, then give up and let the caller fall back
        r = httpx.post(f"{settings.META_BASE_URL}/responses", json=body, timeout=60,
                       headers={"Authorization": f"Bearer {settings.META_API_KEY}"})
        if r.status_code not in (429, 500, 502, 503, 504):
            break
        time.sleep(2)
    r.raise_for_status()
    try:
        text = r.json()["output"][0]["content"][0]["text"]
    except (KeyError, IndexError, TypeError, ValueError) as e:
        raise RuntimeError(f"SAM response without output text: {r.text[:200]}") from e
    metrics.count("sam_images")
    metrics.count("sam_seconds", time.perf_counter() - t0)
    return parse(text)


def to_spines(boxes):
    """SAM boxes -> the spine shape vision reads use, so detections/cluster/geometry.spine work unchanged.
    Text fields stay empty: a spine's identity comes from OCR on its own crop and from matched reader claims."""
    out = []
    for b in boxes:
        x1, y1, x2, y2 = b["box"]
        w, h = b["w"], b["h"]
        out.append({"box_2d": [round(y1 / h * 1000, 1), round(x1 / w * 1000, 1),  # x2/y2 inclusive: +1
                               round((y2 + 1) / h * 1000, 1), round((x2 + 1) / w * 1000, 1)],
                    "orientation": "flat" if (x2 - x1) > 1.6 * (y2 - y1) else "upright",
                    "legibility": "none", "title": "", "author": "", "publisher": "", "marks": [],
                    "source": "sam", "sam_id": b["id"]})
    return out
