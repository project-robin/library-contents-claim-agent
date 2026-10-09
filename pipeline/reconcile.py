"""Reconcile: one vision call per shelf unit decides which candidate lines are the same physical book.

The 3D merge works from positions, titles and box sizes. On a slanted stack, overlapping boxes share one centre and
a spine's top half lands in its own line, so a bed of 12 books still came out as 14-19 lines. A person sorts that
out by looking: the model gets the unit's best full views and every candidate's crop with its OCR and claimed
title, and answers which entries are one book and which are not books.

Guardrails (the brief allows <= 3% confidently wrong):
- It only groups or drops existing lines. It cannot add a book or change a title.
- Every entry must appear exactly once in the answer, or the answer is ignored.
- Two lines with different titles at confidence >= CONF_IDENTIFIED are never merged, and such a line is never
  dropped.
- Every merge and drop goes to the review queue with the model's reason.
"""
from typing import Literal

import cv2
from pydantic import BaseModel

from pipeline import metrics, settings, vision

MAX_ENTRIES = 40  # ponytail: a longer unit is left as it is; chunk it with overlapping views if that ever matters


class Book(BaseModel):
    entries: list[int]
    reason: str


class Drop(BaseModel):
    entry: int
    kind: Literal["part_of_another_spine", "not_a_book", "unsure"]
    reason: str


class Answer(BaseModel):
    books: list[Book]
    not_books: list[Drop]


PROMPT = """You are checking an inventory of books on one shelf or stack, built from a phone video.
The first images are full views of the shelf or stack. Then come candidate entries, each a crop of one detected
spine, with the title our system attached (may be empty or wrong), the text OCR read on the crop, and its size.
Several entries can be the same physical book: seen again from another angle, or only part of its spine (top or
bottom half). Some entries may not be a book at all (a shelf edge, a box, a toy). A book whose spine is covered,
blank or unreadable is still a book: list it in books.

Using the full views, decide the real books:
- books: one item per physical book, listing every entry id that shows it.
- not_books: entries that are not a separate book. kind = part_of_another_spine only if that book is listed in
  books; not_a_book for objects that are not books; unsure if you cannot tell (unsure entries are kept as books).
Every entry id must appear exactly once in total. Never invent a book that has no entry. Do not judge titles."""


def _jpeg(path, side):
    img = cv2.imread(str(path))
    if img is None:
        return None
    s = side / max(img.shape[:2])
    if s < 1:
        img = cv2.resize(img, (int(img.shape[1] * s), int(img.shape[0] * s)))
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()


def _strong(b):
    return bool(b["title"]) and b["id_confidence"] >= settings.CONF_IDENTIFIED


def ask(sweep_dir, unit, frames):
    """unit: the unit's lines (books.json shape); frames: up to 2 frame names showing most of it. Returns Answer."""
    parts = ["Full views of the shelf or stack:"]
    for f in frames:
        j = _jpeg(sweep_dir / "frames" / f, 1280)
        if j:
            parts.append(vision._img(j))
    for i, b in enumerate(unit):
        size = f"{b['spine_height_cm'] or '?'} x {b['spine_thickness_cm'] or '?'} cm"
        parts.append(f"Entry {i}: title attached: {b['title'] or '(none)'} | OCR: {b['ocr_text'][:80] or '(none)'} | "
                     f"{b['orientation']}, spine {size}, seen in {b['views']} frames")
        j = _jpeg(sweep_dir / b["crop"], 384)
        if j:
            parts.append(vision._img(j))
    return vision._generate(settings.VISION_MODEL, parts, Answer, PROMPT, settings.VISION_THINKING)


def apply(unit, ans):
    """Checks the answer against the guardrails. Returns (kept_lines, actions) or None to ignore the answer.
    actions: [(line_id, kind, into_id or '', reason)]."""
    ids = [i for bk in ans.books for i in bk.entries] + [d.entry for d in ans.not_books]
    if sorted(ids) != list(range(len(unit))):
        return None
    drop, actions = set(), []
    for bk in ans.books:
        group = sorted({unit[i]["id"]: unit[i] for i in bk.entries}.values(),
                       key=lambda b: (_strong(b), bool(b["title"]), b["id_confidence"], b["views"]), reverse=True)
        keepers = [group[0]]
        for b in group[1:]:
            if _strong(b) and all(b["title"] != k["title"] for k in keepers):
                keepers.append(b)  # two confident titles: two books, whatever the model grouped
                continue
            k = keepers[0]
            k["views"] += b["views"]
            k["_merged"] = k.get("_merged", []) + [b["id"], *b.get("_merged", [])]
            drop.add(b["id"])
            actions.append((b["id"], "merged", k["id"], bk.reason))
    for d in ans.not_books:
        b = unit[d.entry]
        if d.kind == "unsure" or _strong(b) or b["id"] in drop:
            continue
        drop.add(b["id"])
        actions.append((b["id"], d.kind, "", d.reason))
    return [b for b in unit if b["id"] not in drop], actions


def run(sweep_dir, books, frames_of):
    """frames_of: {line_id: set of frame names its detections came from}. Returns (books, actions)."""
    if not settings.RECONCILE:
        return books, []
    out, actions = [], []
    units = {}
    for b in books:
        units.setdefault(b["shelf"].split("-")[0], []).append(b)
    for name, lines in units.items():
        unit = [b for b in lines if not b["excluded"]]
        if len(unit) < 2 or len(unit) > MAX_ENTRIES:
            out += lines
            continue
        cover = {}
        for b in unit:
            for f in frames_of.get(b["id"], ()):
                cover[f] = cover.get(f, 0) + 1
        frames = sorted(cover, key=lambda f: -cover[f])[:2]
        try:
            res = apply(unit, ask(sweep_dir, unit, frames))
        except Exception:  # noqa: BLE001 - spent quota or outage: the 3D lines stand
            metrics.count("reconcile_failures")
            res = None
        if res is None:
            metrics.count("reconcile_ignored")
            out += lines
            continue
        kept, acts = res
        keep = {b["id"] for b in kept}
        out += [b for b in lines if b["excluded"] or b["id"] in keep]
        actions += acts
    order = {b["id"]: i for i, b in enumerate(books)}
    return sorted(out, key=lambda b: order[b["id"]]), actions
