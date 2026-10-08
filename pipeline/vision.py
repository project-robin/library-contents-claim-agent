"""Gemini vision calls. The model transcribes and classifies what is visible; it never prices or measures."""
import difflib
import re
import threading
import time
from typing import Literal

import httpx
from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from pydantic import BaseModel

from pipeline import metrics, settings

_clients, _client_lock = {}, threading.Lock()
_spent = set()  # (key, model) whose daily quota is gone: calls to that model move on to the next key


def key_for(model):
    """The first key (settings.gemini_keys order) whose daily quota for this model is not known to be spent."""
    keys = settings.gemini_keys()
    return next((k for k in keys if (k, model) not in _spent), keys[0] if keys else "")


def client(key=None):
    """One shared client per key. Locked: threads racing here built several, and the losers' __del__ closed the HTTP
    client mid-request ("Cannot send a request, as the client has been closed")."""
    key = key or key_for(None)
    with _client_lock:
        if key not in _clients:
            _clients[key] = genai.Client(api_key=key,
                                         http_options=types.HttpOptions(timeout=settings.VISION_TIMEOUT_S * 1000))
    return _clients[key]


class Spine(BaseModel):
    box_2d: list[int]  # [ymin, xmin, ymax, xmax], 0-1000
    orientation: Literal["upright", "flat"]
    legibility: Literal["full", "partial", "none"]
    visible_text: str  # verbatim, "…" where covered; generated before title so the title is checked against it
    title: str
    author: str
    publisher: str
    marks: list[str]  # visible: signed, leather, gilt, first edition, limited, old binding


class Item(BaseModel):
    box_2d: list[int]
    category: Literal["shelving", "furniture", "lighting", "art", "rug", "electronics",
                      "appliance", "decor", "plant", "textile", "other"]
    description: str
    material: str
    brand_model: str  # only if a logo/model text is legible
    confidence: float


class FrameRead(BaseModel):
    spines: list[Spine]
    items: list[Item]
    issue: Literal["none", "blur", "glare", "too_far", "occluded", "dark"]


FRAME_PROMPT = """You are the vision sensor of an insurance inventory system. Report only what is visible in this one frame.

SPINES: one entry per book spine visible, including books lying flat or stacked (orientation "flat").
- box_2d = [ymin, xmin, ymax, xmax] normalised 0-1000, tight around the spine.
- visible_text: every word you can actually see printed on this spine, in order, exactly as printed. Where text is
  hidden (price sticker, label, glare, fold, frame edge) write "…" and do not fill it in.
- Transcribe title/author/publisher EXACTLY as printed, using only words that appear in visible_text. Never complete,
  correct or guess missing words: a title half-covered by a sticker is NOT another book by the same author.
  If a field is not readable leave it "". legibility: full = title fully readable, partial = some words, none = nothing.
- marks: only visible physical evidence (e.g. "signed", "leather", "gilt", "first edition", "limited", "old binding").
ITEMS: every non-book object (shelving units, furniture, lamps, framed art/portraits, rugs, electronics, appliances such as
coffee machines, decor, plants). box_2d as above. brand_model only if a logo or model text is legible, else "".
issue: the main capture problem for reading spines in this frame, or "none"."""


_primary_down_until = 0.0  # circuit breaker for VISION_MODEL, shared by every thread in the process


def _generate(model, parts, schema, prompt, thinking=None):
    """Retries 429/5xx with backoff. VISION_MODEL falls back to VISION_FALLBACK_MODEL after one failure, and a
    timeout, overload or spent daily quota trips a breaker so later calls skip straight to the fallback: on a replay
    every read paid the 120 s timeout first (10 min to packet), then 3.8 Flash's 20/day free quota killed the packet.
    The checker never falls back: its value is being a different model."""
    global _primary_down_until
    first = True
    for attempt in range(10):
        primary = model == settings.VISION_MODEL and first and time.time() >= _primary_down_until
        m = settings.VISION_FALLBACK_MODEL if model == settings.VISION_MODEL and not primary else model
        key = key_for(m)
        # fallback runs at its default thinking: 'low' made flash-lite over-detect (27 boxes for 22 spines)
        cfg = types.GenerateContentConfig(
            response_mime_type="application/json", response_schema=schema, temperature=0,
            thinking_config=types.ThinkingConfig(thinking_level=thinking) if thinking and m == model else None)
        try:
            resp = client(key).models.generate_content(model=m, contents=[*parts, prompt], config=cfg)
            metrics.add_tokens(m, resp.usage_metadata)
            if attempt:
                metrics.count("vision_retries", attempt)
            return resp.parsed
        except (genai_errors.APIError, httpx.TimeoutException) as e:  # a timeout surfaces as a 504 or as httpx's
            code = getattr(e, "code", 504)
            if code not in (429, 500, 502, 503, 504) or attempt == 9:
                raise
            # per-minute quotas (free tier: 15 RPM) say how long to wait; a per-day quota says hours
            wait = re.search(r"retryDelay'?\"?:\s*'?\"?(\d+)s", str(e))
            daily = bool(wait) and int(wait[1]) > 120
            if daily:
                _spent.add((key, m))
                if key_for(m) != key:  # another account's key still has quota for this model: no sleep, no fallback
                    metrics.count("gemini_key_rotations")
                    continue
            first = False
            if m == settings.VISION_MODEL and model == settings.VISION_MODEL:
                _primary_down_until = time.time() + (int(wait[1]) if daily else settings.VISION_BREAKER_S)
                metrics.count("vision_fallbacks")
                continue  # straight to the fallback model, no sleep
            if daily:
                raise
            time.sleep(int(wait[1]) + 1 if wait else 2 * (attempt + 1))


def _img(jpeg):
    return types.Part.from_bytes(data=jpeg, mime_type="image/jpeg")


def read_frame(jpeg: bytes) -> dict:
    out = _generate(settings.VISION_MODEL, [_img(jpeg)], FrameRead, FRAME_PROMPT, settings.VISION_THINKING)
    return clean_read(out.model_dump()) if out else {"spines": [], "items": [], "issue": "none"}


def _box_ok(b):
    return len(b) == 4 and all(0 <= v <= 1000 for v in b) and b[0] < b[2] and b[1] < b[3]


_STOP = {"the", "a", "an", "of", "and", "to", "in", "on", "for", "by", "with", "at", "from", "is", "my"}


def _words(s):
    return re.findall(r"[a-z0-9]+", s.lower().replace("'", "").replace("\u2019", ""))


def title_supported(title, visible):
    """Every content word of the title must be printed on the spine. With a price sticker over the title the reader
    returned other real books by the same author ('TO… ROW' -> 'Die Trying', 'EVA…' -> 'Evil Under the Sun'), all
    'full' legibility and matched by the catalogue at P>0.98. Those are invisible to every later check."""
    seen = set(_words(visible))
    for w in _words(title):
        if w in _STOP or w in seen:
            continue
        if len(w) >= 4 and any(difflib.SequenceMatcher(None, w, v).ratio() >= 0.8 for v in seen):
            continue  # OCR-level spelling differences
        return False
    return True


def _check_title(s):
    """Blank a title the spine does not show; keep it under title_unsupported for the audit."""
    if "visible_text" not in s or not s["title"] or title_supported(s["title"], s["visible_text"]):
        return s  # reads saved before visible_text existed are left as they were
    return {**s, "title_unsupported": s["title"], "title": "",
            "legibility": "partial" if s["legibility"] == "full" else s["legibility"]}


def clean_read(r: dict) -> dict:
    """Drop detections whose box is not [ymin, xmin, ymax, xmax] in 0-1000 (the schema cannot pin a list length, and
    one 5-value box once crashed the whole packet), and blank titles the spine text does not support."""
    out = {**r, "spines": [_check_title(s) for s in r["spines"] if _box_ok(s["box_2d"])],
           "items": [i for i in r["items"] if _box_ok(i["box_2d"])]}
    out["dropped_bad_boxes"] = r.get("dropped_bad_boxes", 0) + len(r["spines"]) + len(r["items"]) \
        - len(out["spines"]) - len(out["items"])
    return out


class CropText(BaseModel):
    index: int
    visible_text: str
    title: str
    author: str
    publisher: str
    legibility: Literal["full", "partial", "none"]


class CropBatch(BaseModel):
    crops: list[CropText]


def read_crops(jpegs: list[bytes]) -> list[dict]:
    """Second, closer read of spine crops. Returns one dict per crop, in order."""
    parts = []
    for i, j in enumerate(jpegs):
        parts += [f"Crop {i}:", _img(j)]
    prompt = ("Each crop is one book spine (it may be rotated). For each crop index: visible_text = every word actually "
              "printed and visible, in order, with \"…\" where text is covered or cut off; then title, author and "
              "publisher exactly as printed, using only words in visible_text. Do not guess or complete missing words "
              "(a covered title is not another book by the same author); leave unreadable fields empty.")
    try:
        out = _generate(settings.VISION_MODEL, parts, CropBatch, prompt, settings.VISION_THINKING)
    except Exception:  # noqa: BLE001 - spent quota: these crops stay unread (review), the packet still builds
        metrics.count("crop_read_failures")
        out = None
    by_i = {c.index: _check_title(c.model_dump()) for c in (out.crops if out else [])}
    return [by_i.get(i, {"title": "", "author": "", "publisher": "", "legibility": "none"}) for i in range(len(jpegs))]


class Verdict(BaseModel):
    verdict: Literal["match", "mismatch", "unsure"]
    reason: str


class IndexedVerdict(Verdict):
    index: int


class VerdictBatch(BaseModel):
    verdicts: list[IndexedVerdict]


def check_identities(claims: list[tuple[bytes, str, str]]) -> list[dict]:
    """check_identity for several spines in one call: free Flash-Lite allows 15 calls a minute, and one call per book
    spent about 2 minutes of a sweep waiting on that limit. claims: [(crop, title, author)]; one verdict each, in order.
    Raises on failure, like check_identity."""
    parts = []
    for i, (crop, title, author) in enumerate(claims):
        parts += [f'Crop {i}: an inventory system claims this spine is "{title}" by {author or "unknown author"}.',
                  _img(crop)]
    prompt = ("Each crop is one book spine. For each crop index, judge only from that spine's visible text: match = the "
              "visible text supports the claim; mismatch = it contradicts it; unsure = not enough visible text. Text on "
              "a neighbouring spine at the crop's edge is not this spine. Give a one-line reason.")
    out = _generate(settings.CHECKER_MODEL, parts, VerdictBatch, prompt)
    by_i = {v.index: {"verdict": v.verdict, "reason": v.reason} for v in (out.verdicts if out else [])}
    return [by_i.get(i, {"verdict": "unsure", "reason": "no verdict returned"}) for i in range(len(claims))]


def check_identity(crop: bytes, title: str, author: str) -> dict:
    """Independent audit: does this spine really show the claimed book?"""
    prompt = (f'An inventory system claims this spine is "{title}" by {author or "unknown author"}. '
              "Judge only from the visible spine text. match = the visible text supports this claim; mismatch = it "
              "contradicts it; unsure = not enough visible text. Give a one-line reason.")
    out = _generate(settings.CHECKER_MODEL, [_img(crop)], Verdict, prompt)
    return out.model_dump() if out else {"verdict": "unsure", "reason": "no response"}
