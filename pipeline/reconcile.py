"""Reconcile: one vision call per shelf unit decides which candidate lines are the same physical book.

The 3D merge works from positions, titles and box sizes. On a slanted stack, overlapping boxes share one centre and
a spine's top half lands in its own line, so a bed of 12 books still came out as 14-19 lines. A person sorts that
out by looking: the model gets the unit's best full views and every candidate's crop with its OCR and claimed
title, and answers which entries are one book and which are not books.

Guardrails (the brief allows <= 3% confidently wrong):
- It only groups or drops existing lines. It cannot add a book or change a title.
- An entry the answer leaves out or repeats is kept as it is (first place counts); an answer covering under half
  the entries is ignored.
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


def _clean(groups, drops, n):
    """The model's answer, made safe: ids out of range are ignored, an id counts only at its first place, an id in
    a group is never also dropped, and an id it left out stays as its own line. On 36 item lines it once repeated
    or skipped an id, and the old all-or-nothing rule threw the whole answer away. Returns (groups, drops) or None
    when the answer covers under half the lines (it did not do the task)."""
    seen, out = set(), []
    for g in groups:
        ids = [i for i in g.entries if 0 <= i < n and i not in seen]
        seen.update(ids)
        out.append((ids, g.reason))
    kept_drops = [d for d in drops if 0 <= d.entry < n and d.entry not in seen and not seen.add(d.entry)]
    if len(seen) < n / 2:
        return None
    if len(seen) < n:
        metrics.count("reconcile_partial_answers")
    out += [([i], "not in the answer: kept as it is") for i in range(n) if i not in seen]
    return out, kept_drops


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
    res = _clean(ans.books, ans.not_books, len(unit))
    if res is None:
        return None
    groups, not_books = res
    drop, actions = set(), []
    for entries, reason in groups:
        group = sorted({unit[i]["id"]: unit[i] for i in entries}.values(),
                       key=lambda b: (_strong(b), bool(b["title"]), b["id_confidence"], b["views"]), reverse=True)
        if not group:  # an empty book in the answer
            continue
        keepers = [group[0]]
        for b in group[1:]:
            if _strong(b) and all(b["title"] != k["title"] for k in keepers):
                keepers.append(b)  # two confident titles: two books, whatever the model grouped
                continue
            k = keepers[0]
            k["views"] += b["views"]
            k["_merged"] = k.get("_merged", []) + [b["id"], *b.get("_merged", [])]
            drop.add(b["id"])
            actions.append((b["id"], "merged", k["id"], reason))
    for d in not_books:
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


# ---------- non-book items ----------
# A live sweep gave 36 item lines for one bedroom: one bed as 5 lines in 4 categories (bed, headboard, mattress
# cover, bedsheet), the wardrobe 3 times, wall sockets and switches, and the printed marker card twice as 'art'.
# 3D extents are too rough to merge by size (the bed measured 110 x 19 x 72 cm in one line, 149 x 67 x 66 in another).

class Thing(BaseModel):
    entries: list[int]
    reason: str


class NotItem(BaseModel):
    entry: int
    kind: Literal["part_of_another_item", "scale_marker", "building_fixture", "unsure"]
    reason: str


class ItemAnswer(BaseModel):
    objects: list[Thing]
    not_items: list[NotItem]


ITEM_PROMPT = """You are checking the non-book contents of one room for an insurance claim, built from a phone video.
The first images are wide views of the room. Then come candidate entries, each a crop of one detection with its
category, description and seen-in count. Several entries can be one physical object, seen again from another angle
or only in part: a headboard, mattress or bed frame seen alone is part of its bed; a wardrobe door is part of its
wardrobe. Bedding, pillows, curtains, rugs and things standing on furniture are separate objects.

Decide the real objects:
- objects: one item per physical object, listing every entry id that shows it.
- not_items: entries that are not a separate contents item. kind = part_of_another_item only if that object is in
  objects; scale_marker for the printed black-and-white square marker card used for measuring, or the clipboard or
  board holding it; building_fixture for things built into the building (wall sockets, light switches, switch
  boards, recessed ceiling lights, doors, windows, built-in shelves); unsure if you cannot tell (kept as an item).
Every entry id must appear exactly once in total. Never invent an object that has no entry. Do not judge prices."""


def ask_items(sweep_dir, items, frames):
    parts = ["Wide views of the room:"]
    for f in frames:
        j = _jpeg(sweep_dir / "frames" / f, 1024)
        if j:
            parts.append(vision._img(j))
    for i, it in enumerate(items):
        parts.append(f"Entry {i}: {it['category']} | {it['description'][:100]} | seen in {it['views']} frames")
        j = _jpeg(sweep_dir / it["crop"], 320)
        if j:
            parts.append(vision._img(j))
    return vision._generate(settings.VISION_MODEL, parts, ItemAnswer, ITEM_PROMPT, settings.VISION_THINKING)


# The kept line of a group is the larger kind of object, and bedding, rugs, decor and art are never folded into a
# piece of furniture: on one of three runs the model put bed, bedsheet, pillows and bolster in one object, and the
# bedsheet (most views) became the line while the bed vanished.
RANK = {"furniture": 0, "shelving": 0, "appliance": 1, "electronics": 1, "lighting": 2, "art": 3, "decor": 4,
        "rug": 5, "textile": 5}
SEPARATE = {"art", "decor", "rug", "textile"}


def _user_said(it):
    return it.get("is_print") is not None or bool(it.get("notes"))


def apply_items(items, ans):
    """Guardrails: see _clean for a malformed answer; a line the user spoke about is never dropped and
    is the one kept when merged. Returns (kept, actions) or None; actions: [(id, kind, into_id or '', reason)]."""
    res = _clean(ans.objects, ans.not_items, len(items))
    if res is None:
        return None
    groups, not_items = res
    drop, actions = set(), []
    for entries, reason in groups:
        group = sorted({items[i]["id"]: items[i] for i in entries}.values(),
                       key=lambda it: (-_user_said(it), RANK.get(it["category"], 6), -it["views"], -it["confidence"]))
        if not group:  # an empty object in the answer
            continue
        k = group[0]
        for it in group[1:]:
            if _user_said(it) and _user_said(k):
                continue  # two lines the user answered about: keep both
            if it["category"] in SEPARATE and k["category"] not in SEPARATE:
                continue  # a bedsheet, rug or picture is its own item, not part of the furniture it lies on
            k["views"] += it["views"]
            k["notes"] = k.get("notes", []) + it.get("notes", [])
            drop.add(it["id"])
            actions.append((it["id"], "merged", k["id"], reason))
    for d in not_items:
        it = items[d.entry]
        if d.kind == "unsure" or _user_said(it) or it["id"] in drop:
            continue
        drop.add(it["id"])
        actions.append((it["id"], d.kind, "", d.reason))
    return [it for it in items if it["id"] not in drop], actions


def run_items(sweep_dir, items, frames):
    """frames: up to 2 wide frames of the room. Returns (items, actions); on failure the lines stand."""
    if not settings.RECONCILE or not 2 <= len(items) <= 60:
        return items, []
    try:
        res = apply_items(items, ask_items(sweep_dir, items, frames))
    except Exception:  # noqa: BLE001 - spent quota or outage
        metrics.count("item_reconcile_failures")
        res = None
    if res is None:
        metrics.count("item_reconcile_ignored")
        return items, []
    return res
