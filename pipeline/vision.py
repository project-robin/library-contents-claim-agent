"""Gemini vision calls. The model transcribes and classifies what is visible; it never prices or measures."""
import time
from typing import Literal

from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from pydantic import BaseModel

from pipeline import metrics, settings

_client = None


def client():
    global _client
    if _client is None:
        _client = genai.Client(api_key=settings.GEMINI_API_KEY)
    return _client


class Spine(BaseModel):
    box_2d: list[int]  # [ymin, xmin, ymax, xmax], 0-1000
    orientation: Literal["upright", "flat"]
    legibility: Literal["full", "partial", "none"]
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
- Transcribe title/author/publisher EXACTLY as printed. Never complete, correct or guess missing words.
  If a field is not readable leave it "". legibility: full = title fully readable, partial = some words, none = nothing.
- marks: only visible physical evidence (e.g. "signed", "leather", "gilt", "first edition", "limited", "old binding").
ITEMS: every non-book object (shelving units, furniture, lamps, framed art/portraits, rugs, electronics, appliances such as
coffee machines, decor, plants). box_2d as above. brand_model only if a logo or model text is legible, else "".
issue: the main capture problem for reading spines in this frame, or "none"."""


def _generate(model, parts, schema, prompt, thinking=None):
    """Retries 429/5xx with backoff; after one 'high demand' failure switches to the fallback model."""
    for attempt in range(5):
        # the checker never falls back: its value is being a different model
        m = settings.VISION_FALLBACK_MODEL if attempt >= 1 and model == settings.VISION_MODEL else model
        # fallback runs at its default thinking: 'low' made flash-lite over-detect (27 boxes for 22 spines)
        cfg = types.GenerateContentConfig(
            response_mime_type="application/json", response_schema=schema, temperature=0,
            thinking_config=types.ThinkingConfig(thinking_level=thinking) if thinking and m == model else None)
        try:
            resp = client().models.generate_content(model=m, contents=[*parts, prompt], config=cfg)
            metrics.add_tokens(m, resp.usage_metadata)
            if attempt:
                metrics.count("vision_retries", attempt)
            return resp.parsed
        except genai_errors.APIError as e:
            if e.code not in (429, 500, 502, 503, 504) or attempt == 4:
                raise
            time.sleep(2 * (attempt + 1))


def _img(jpeg):
    return types.Part.from_bytes(data=jpeg, mime_type="image/jpeg")


def read_frame(jpeg: bytes) -> dict:
    out = _generate(settings.VISION_MODEL, [_img(jpeg)], FrameRead, FRAME_PROMPT, settings.VISION_THINKING)
    return out.model_dump() if out else {"spines": [], "items": [], "issue": "none"}


class CropText(BaseModel):
    index: int
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
    prompt = ("Each crop is one book spine (it may be rotated). For each crop index, transcribe title, author and "
              "publisher exactly as printed. Do not guess or complete missing words; leave unreadable fields empty.")
    out = _generate(settings.VISION_MODEL, parts, CropBatch, prompt, settings.VISION_THINKING)
    by_i = {c.index: c.model_dump() for c in (out.crops if out else [])}
    return [by_i.get(i, {"title": "", "author": "", "publisher": "", "legibility": "none"}) for i in range(len(jpegs))]


class Verdict(BaseModel):
    verdict: Literal["match", "mismatch", "unsure"]
    reason: str


def check_identity(crop: bytes, title: str, author: str) -> dict:
    """Independent audit: does this spine really show the claimed book?"""
    prompt = (f'An inventory system claims this spine is "{title}" by {author or "unknown author"}. '
              "Judge only from the visible spine text. match = the visible text supports this claim; mismatch = it "
              "contradicts it; unsure = not enough visible text. Give a one-line reason.")
    out = _generate(settings.CHECKER_MODEL, [_img(crop)], Verdict, prompt)
    return out.model_dump() if out else {"verdict": "unsure", "reason": "no response"}
