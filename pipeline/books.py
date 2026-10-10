"""Books stage: per-frame spine detections -> one line per physical book, with shelf, dims and identity.

Dedup is spatial: the same spine seen in many frames lands at the same 3D point. Identity comes from a catalogue
match of the transcribed spine text; a spine with no readable text stays unidentified (title blank, dims kept).
"""
import difflib
import json
import re
from collections import Counter

import cv2
import httpx
import numpy as np

from concurrent.futures import ThreadPoolExecutor

from pipeline import drex, metrics, net, settings, vision
from pipeline.geometry import box_px

LEG_W = {"full": 2, "partial": 1, "none": 0}
LEGIBLE = ["none", "partial", "partial", "full"]  # by distinctive OCR word count: 0, 1-2, 3+
SAME_WORK_Q = ("Is catalogue candidate {c} the same book (same work) as the text read from the spine? Spine text may be "
               "partial, abbreviated or show only the author's surname. A different volume number, a different work by "
               "the same author, or a guess that the spine text does not support means no.")


def norm(s):
    s = re.sub(r"[^a-z0-9 ]", " ", (s or "").lower())
    s = re.sub(r"\s+", " ", s).strip()
    return s[4:] if s.startswith("the ") else s


def sim(a, b):
    a, b = norm(a), norm(b)
    if not (a and b):
        return 0.0
    r = difflib.SequenceMatcher(None, a, b).ratio()
    # "Volume 2" is not "Volume 3": series and encyclopedia sets differ only by number
    return min(r, 0.5) if re.findall(r"\d+", a) != re.findall(r"\d+", b) else r


def titled(s):
    """A read worth matching on: 4+ letters. A bare number, article or clipped word ('4', 'THE', 'ISH') is not."""
    n = norm(s)
    return n if len(re.sub(r"[^a-z]", "", n)) >= 4 else ""


def related(a, b):
    """Two reads of one spine: near-identical, or one starts or ends the other ('RAAVAN' / 'RAAVAN: ENEMY OF
    ARYAVARTA', 'OF SECRETS' / 'THE SHIP OF SECRETS'). A title inside a longer one ('IKIGAI' in 'JAPONISME: IKIGAI,
    FOREST BATHING...') is a different book."""
    a, b = titled(a), titled(b)
    if not (a and b):
        return False
    short, long_ = sorted((a, b), key=len)
    return sim(a, b) >= 0.85 or long_.startswith(short) or long_.endswith(short)


def load_notes(sweep_dir):
    p = sweep_dir / "notes.jsonl"
    return [json.loads(x) for x in p.read_text().splitlines() if x] if p.exists() else []


def frame_times(sweep_dir):
    return {r["file"]: r["t"] for r in map(json.loads, (sweep_dir / "frames.jsonl").read_text().splitlines()) if r}


def nearest_det(reads, times, t, key):
    """Voice notes are anchored by time: nearest read keyframe within 2 s, then the detection closest to centre."""
    cands = [(abs(times.get(f, 1e9) - t), f) for f, r in reads.items() if r.get(key)]
    if not cands:
        return None
    dt, f = min(cands)
    if dt > 2.0:
        return None
    dets = reads[f][key]
    k = min(range(len(dets)), key=lambda k: np.hypot((dets[k]["box_2d"][1] + dets[k]["box_2d"][3]) / 2 - 500,
                                                     (dets[k]["box_2d"][0] + dets[k]["box_2d"][2]) / 2 - 500))
    return f"{f}:{k}"


def crop(sweep_dir, frame, box_2d, out_path, pad=0.08):
    img = cv2.imread(str(sweep_dir / "frames" / frame))
    h, w = img.shape[:2]
    x0, y0, x1, y1 = box_px(box_2d, w, h)
    px, py = (x1 - x0) * pad + 4, (y1 - y0) * pad + 4
    c = img[int(max(0, y0 - py)):int(min(h, y1 + py)), int(max(0, x0 - px)):int(min(w, x1 + px))]
    cv2.imwrite(str(out_path), c, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return out_path.read_bytes()


# ---------- dedup ----------

def detections(reads, times, geo):
    """overlap: the box's highest IoU with another box in its frame. On a stack filmed at an angle, axis-aligned
    boxes of diagonal spines take in their neighbours (most boxes overlap > 0.2 there, under 12% on a straight-on
    shelf photo), so a crop from such a box shows the neighbour's text too."""
    dets = []
    for f, r in reads.items():
        for k, s in enumerate(r["spines"]):
            d = {"id": f"{f}:{k}", "frame": f, "t": times.get(f), **s,
                 "area": (s["box_2d"][2] - s["box_2d"][0]) * (s["box_2d"][3] - s["box_2d"][1]),
                 "overlap": max([iou(s["box_2d"], o["box_2d"]) for j, o in enumerate(r["spines"]) if j != k] or [0.0])}
            m = geo.spine(f, s["box_2d"], s["orientation"]) if geo.ok else None
            if m:
                d.update(m)
            dets.append(d)
    return dets


def iou(a, b):
    """Overlap of two box_2d [ymin, xmin, ymax, xmax] boxes, 0-1."""
    h = min(a[2], b[2]) - max(a[0], b[0])
    w = min(a[3], b[3]) - max(a[1], b[1])
    inter = max(h, 0) * max(w, 0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def claims(spines, reads):
    """SAM boxes carry no text. The vision model's spine on the same frame that overlaps a SAM box most (IoU >= 0.3)
    lends it title/author as a claim, which the crop's OCR then confirms or vetoes. Greedy by IoU, one-to-one: a
    drifting vision box can claim only one spine. Returns reads-shaped {frame: {"spines": [...]}} and the number of
    vision spines left unmatched (dropped)."""
    out, unmatched = {}, 0
    for f, boxes in spines.items():
        gem = reads.get(f, {}).get("spines", [])
        pairs = sorted(((iou(s["box_2d"], g["box_2d"]), i, j) for i, s in enumerate(boxes) for j, g in enumerate(gem)),
                       reverse=True)
        dets, used_i, used_j = [dict(s) for s in boxes], set(), set()
        for v, i, j in pairs:
            if v < 0.3:
                break
            if i in used_i or j in used_j:
                continue
            used_i.add(i), used_j.add(j)
            dets[i].update({k: gem[j][k] for k in ("title", "author", "publisher", "marks", "legibility")},
                           claim_iou=round(v, 2))
        unmatched += len(gem) - len(used_j)
        out[f] = {"spines": dets}
    return out, unmatched


def layout_2d(dets):
    """No geometry (a still photo, or reconstruction failed): spines cannot be matched across frames, so each
    detection is its own book. Per frame (unit A, B, ...): rows top to bottom, a box joining a row when it shares
    >= 50% of the smaller height with the row's span; then left to right within the row."""
    clusters = []
    for fi, f in enumerate(dict.fromkeys(d["frame"] for d in dets)):
        rows = []
        for d in sorted((d for d in dets if d["frame"] == f), key=lambda d: d["box_2d"][0]):
            y0, y1 = d["box_2d"][0], d["box_2d"][2]
            row = next((r for r in rows if min(y1, r["y1"]) - max(y0, r["y0"]) >= 0.5 * min(y1 - y0, r["y1"] - r["y0"])),
                       None)
            if row:
                row.update(y0=min(y0, row["y0"]), y1=max(y1, row["y1"]))
                row["dets"].append(d)
            else:
                rows.append({"y0": y0, "y1": y1, "dets": [d]})
        letter = chr(65 + fi) if fi < 26 else f"U{fi}"
        for ri, row in enumerate(rows, 1):
            for pi, d in enumerate(sorted(row["dets"], key=lambda d: d["box_2d"][1]), 1):
                clusters.append({"members": [d], "center": np.zeros(3), "orientation": d["orientation"],
                                 "bottom_z": 0.0, "title": d["title"], "unit": letter, "row": ri,
                                 "shelf": f"{letter}-{ri}", "position": pi})
    return clusters


def cluster(dets):
    """Greedy: best-read detections seed clusters; others join the nearest compatible one."""
    clusters = []
    for d in sorted(dets, key=lambda d: (-LEG_W[d["legibility"]], -d["area"])):
        if "center" not in d:
            continue
        best, best_dist = None, 1e9
        for c in clusters:
            if c["orientation"] != d["orientation"] or abs(c["bottom_z"] - d["bottom_z"]) > 0.08:
                continue
            dist = float(np.linalg.norm(c["center"][:2] - d["center"][:2]))
            s = sim(c["title"], d["title"])
            both = bool(titled(c["title"]) and titled(d["title"]))
            radius = float(np.clip(0.6 * min(c["thickness_cm"], d["thickness_cm"]) / 100, 0.008, 0.03))
            ok = (dist < radius and not (both and s < 0.5)) or (both and s >= 0.85 and dist < 0.15)
            if ok and dist < best_dist:
                best, best_dist = c, dist
        if best is None:
            clusters.append({"members": [d], "center": d["center"].copy(), "orientation": d["orientation"],
                             "bottom_z": d["bottom_z"], "title": d["title"], "thickness_cm": d["thickness_cm"]})
        else:
            best["members"].append(d)
            n = len(best["members"])
            best["center"] = best["center"] + (d["center"] - best["center"]) / n
            best["bottom_z"] += (d["bottom_z"] - best["bottom_z"]) / n
            if not titled(best["title"]):
                best["title"] = d["title"]
    return clusters


def merge(clusters):
    """Second pass over cluster(). In a video, one book read differently across frames, or seeded by an unread box
    before a read one joined, or with its orientation flipping (a tilted spine's box is near square), lands in two
    clusters; so does a read of the author line as the title ('NARENDRA MODI' beside 'EXAM WARRIORS'). Join such pairs
    within 15 cm. Position alone is not enough: on a stack filmed at an angle, neighbouring books' boxes overlap and
    their centres coincide."""
    out = []
    for c in sorted(clusters, key=lambda c: -len(c["members"])):
        best = None
        for o in out:
            dxy = float(np.linalg.norm(o["center"][:2] - c["center"][:2]))
            dz = abs(o["bottom_z"] - c["bottom_z"])
            same = related(o["title"], c["title"]) or any(related(c["title"], m["author"]) for m in o["members"])
            if same and dxy < 0.15 and dz <= 0.08:
                best = o
                break
        if best is None:
            out.append(c)
            continue
        n0, n1 = len(best["members"]), len(c["members"])
        best["members"] += c["members"]
        best["center"] = (best["center"] * n0 + c["center"] * n1) / (n0 + n1)
        best["bottom_z"] = (best["bottom_z"] * n0 + c["bottom_z"] * n1) / (n0 + n1)
        if not titled(best["title"]):
            best["title"] = c["title"]
    for c in out:
        c["orientation"] = Counter(m["orientation"] for m in c["members"]).most_common(1)[0][0]
    return out


def same_book(books):
    """After identification, on 3D clusters. Pieces the merge pass could not join resolve to one catalogue book: SAM
    boxes the top or bottom of a tall upright spine on its own (another bottom height), and garbled reads ('THE OATH C
    THE VAYUPUTH') are not related() to the clean title. Same title within 15 cm, and within half the taller spine's
    height (8 cm at least), is one book; an unread piece inside an identified spine (within 40% of its thickness across, and
    within its height if upright or its thickness if flat) is part of it, and so is an unread piece within 15 cm whose
    OCR names exactly one identified book ('FURRY FAMILY GATES', 'Aciman'). An unread box under 10 cm long is a piece
    of some spine, not a book (spines are 14-32 cm). Views add up. Returns (kept, folded, fragments).
    Ceiling: two copies of one book side by side become one."""
    def near(a, b, title):
        ca, cb = np.array(a["_center"]), np.array(b["_center"])
        dxy, dz = float(np.linalg.norm(ca[:2] - cb[:2])), abs(float(ca[2] - cb[2]))
        tall = max(a["spine_height_cm"] or 0, b["spine_height_cm"] or 0) / 100
        if title:
            return dxy < 0.15 and dz <= max(0.08, tall / 2)
        span = (a["spine_height_cm"] if a["orientation"] == "upright" else a["spine_thickness_cm"]) or 0
        # a neighbour's centre sits a full thickness away; a piece of this spine sits within it
        return dxy < min(0.025, 0.4 * (a["spine_thickness_cm"] or 0) / 100) and dz <= span / 200
    from pipeline.spine_ocr import same_work, supports, tokens  # imports this module

    def volumes_agree(a, b):  # a series' volumes can resolve to one catalogue title ('Volume 3', 'Volume 7')
        na, nb = (re.findall(r"\d+", x["spine_text"]["title"] or "") for x in (a, b))
        return not na or not nb or na == nb

    kept, folded, fragments = [], 0, 0
    for b in sorted(books, key=lambda b: (bool(b["title"]), b["views"]), reverse=True):
        live = [k for k in kept if k["title"] and not k["excluded"] and not b["excluded"]]
        unread = not b["title"] and not b["spine_text"]["title"]
        k = next((k for k in live if b["title"] and same_work(b["title"], k["title"]) and volumes_agree(k, b)
                  and near(k, b, True)), None)
        read = b["spine_text"]["title"]
        if k is None and b["title"] and titled(read):
            # a half-spine read ('THE OAT THE VAYUI') resolved to another work ('The Vayu Purana') next to the book it
            # is part of: the read fits the neighbour's title better than its own (sweep 5ced)
            k = next((k for k in live if near(k, b, True) and sim(read, k["title"]) >= sim(read, b["title"]) + 0.2), None)
        if unread:
            k = next((k for k in live if near(k, b, False)), None)
            named = [k for k in live if near(k, b, True) and supports(f"{k['title']} {k['author']}", tokens(b["ocr_text"]))]
            k = k or (named[0] if len(named) == 1 else None)
        if k is not None:
            k["views"] += b["views"]
            k["_merged"] = k.get("_merged", []) + [b.get("id"), *b.get("_merged", [])]  # their frames show this book too
            folded += 1
        elif unread and not b["excluded"] and (b["spine_height_cm"] or 0) < 10:
            fragments += 1
        else:
            kept.append(b)
    keep = {id(b) for b in kept}
    return [b for b in books if id(b) in keep], folded, fragments


def book_frames(books, clusters):
    """{line id: frames its detections came from}, including lines same_book() folded into it."""
    by_id = {c["id"]: {m["frame"] for m in c["members"]} for c in clusters}
    return {b["id"]: set().union(*(by_id.get(i, set()) for i in [b["id"], *b.get("_merged", [])])) for b in books}


def frame_cap(books, clusters, reads):
    """The most books a unit can hold, when the whole unit fits in one view: Gemini read a photo of 12 books on a bed
    as 12 spines (in 5 frames), while the 3D clusters still held unread pieces. A frame shows a unit when it holds
    members of 75%+ of the unit's identified books; the cap is the most, over such frames, of the spines Gemini
    listed plus the identified books not linked to the frame. Above
    the cap, unread lines (weakest candidate, fewest views first) are dropped; any line with a title stays: on sweep
    203017 dropping low-confidence titles too removed real books (21 -> 14). A long shelf
    never fits one frame and keeps every line. Returns (kept, dropped)."""
    members = book_frames(books, clusters)
    weak = lambda b: b["id_confidence"] < settings.CONF_IDENTIFIED
    unread = lambda b: not b["title"]  # a real book read at low confidence (0.5) is never dropped, only unread lines
    drop = set()
    for unit in sorted({b["shelf"].split("-")[0] for b in books}):
        ub = [b for b in books if b["shelf"].split("-")[0] == unit and not b["excluded"]]
        known = [b for b in ub if not weak(b)]
        if len(known) < 3:
            continue
        seen = lambda f: sum(f in members[b["id"]] for b in known)
        # identified books not linked to the frame are added back: a box of a visible book can land in a neighbour's
        # line (Oath and Ikigai in the bed photo), and a book truly out of view must still fit under the cap
        cap = max((len(reads[f]["spines"]) + len(known) - seen(f) for f in reads if seen(f) >= 0.75 * len(known)),
                  default=0)
        if cap >= len(known) and len(ub) > cap:
            spare = sorted((b for b in ub if unread(b)), key=lambda b: (b["id_confidence"], b["views"]))
            drop |= {b["id"] for b in spare[:len(ub) - cap]}
    return [b for b in books if b["id"] not in drop], len(drop)


def seen_once(clusters, frames):
    """In a sweep of 20+ frames a real spine shows in several; one seen in a single frame with no text is a stray box
    (wall edge, a neighbour's sliver). Returns (kept, dropped)."""
    if frames < 20:
        return clusters, []
    drop = [c for c in clusters if len(c["members"]) == 1 and not titled(c["title"])]
    return [c for c in clusters if c not in drop], drop


def vote(texts):
    """texts: [(title, author, publisher, weight)]. Returns (title, author, publisher, agreement 0-1)."""
    texts = [t for t in texts if norm(t[0]) and t[3] > 0]
    if not texts:
        return "", "", "", 0.0
    groups = []
    for t in texts:
        g = next((g for g in groups if sim(g[0][0], t[0]) >= 0.85), None)
        g.append(t) if g else groups.append([t])
    g = max(groups, key=lambda g: sum(x[3] for x in g))
    agreement = sum(x[3] for x in g) / sum(x[3] for x in texts)
    pick = lambda k: (Counter(x[k] for x in g if norm(x[k])).most_common(1) or [("", 0)])[0][0]
    return pick(0), pick(1), pick(2), agreement


# ---------- shelves ----------

def assign_shelves(clusters):
    """Units = groups of books separated by > 0.4 m; rows = bottom heights separated by > 0.10 m; row 1 = top."""
    n = len(clusters)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    xy = np.array([c["center"][:2] for c in clusters]) if n else np.zeros((0, 2))
    for a in range(n):  # ponytail: O(n^2), fine below a few thousand books
        for b in range(a + 1, n):
            if np.linalg.norm(xy[a] - xy[b]) < 0.4:
                parent[find(a)] = find(b)
    units = {}
    for a in range(n):
        units.setdefault(find(a), []).append(clusters[a])
    order = sorted(units.values(), key=lambda u: min(m["t"] or 0 for c in u for m in c["members"]))
    for ui, unit in enumerate(order):
        letter = chr(65 + ui) if ui < 26 else f"U{ui}"
        pts = np.array([c["center"][:2] for c in unit])
        axis = np.linalg.svd(pts - pts.mean(0), full_matrices=False)[2][0] if len(unit) > 1 else np.array([1.0, 0])
        rows, cur = [], []
        for c in sorted(unit, key=lambda c: -c["bottom_z"]):
            if cur and cur[-1]["bottom_z"] - c["bottom_z"] > 0.10:
                rows.append(cur)
                cur = []
            cur.append(c)
        rows.append(cur)
        for ri, row in enumerate(rows, 1):
            for pi, c in enumerate(sorted(row, key=lambda c: c["center"][:2] @ axis), 1):
                c.update(unit=letter, row=ri, shelf=f"{letter}-{ri}", position=pi)


def shelf_run_m(books):
    """Sum of upright thicknesses + one length per flat stack, per shelf row."""
    total = 0.0
    for b in books:
        if b["orientation"] == "upright" and b["spine_thickness_cm"]:
            total += b["spine_thickness_cm"]
    stacks = {}
    for b in books:
        if b["orientation"] == "flat" and b["spine_height_cm"]:
            key = (b["shelf"], round(b["_center"][0] / 0.15), round(b["_center"][1] / 0.15))
            stacks[key] = max(stacks.get(key, 0), b["spine_height_cm"])
    return round((total + sum(stacks.values())) / 100, 2)


# ---------- catalogue ----------

VOLUME = re.compile(r"[\s,.:;]+((vol(ume)?|band|bd|tome|book|part|teil)\.?\s*)?([ivxlc]+|\d{1,3})\.?$", re.I)


def search_terms(title, author):
    """Spines carry volume markers ('In Nacht und Eis II') and punctuation ('Hofstadter:') that break catalogue search.
    Search the work; DREX later judges whether the volume fits."""
    t = VOLUME.sub("", title.strip()).strip(" ,.:;-") or title
    a = re.sub(r"[^\w\s.'-]", " ", author).strip()
    return t, re.sub(r"\s+", " ", a)


class Unavailable(list):
    """No candidates because the catalogue failed (429 after retries, timeout), not because the book is unknown.
    Never cached: Google Books' daily quota ran out on photo A and 'Wild Wind', 'Hema Malini' and 'Test the Sky with
    Glory' were cached as 'no such book'."""


def _failed():
    metrics.count("catalog_failures")
    return Unavailable()


_gbooks_spent = False  # ponytail: one flag per process; the daily quota resets overnight, a new run tries again


def _gbooks(title, author, free=False):
    """free=True: plain-text query. intitle:/inauthor: returns nothing for some Indian editions ('An Unsuitable Boy',
    'Dream With Your Eyes Open' with a misspelt author) that a plain query finds first."""
    global _gbooks_spent
    if _gbooks_spent:  # daily quota gone: each call would still pay ~7 s of retries for a 429
        return _failed()
    q = f"{title} {author}".strip() if free else f"intitle:{title}" + (f" inauthor:{author}" if author else "")
    params = {"q": q, "maxResults": 5, "printType": "books"}
    key = settings.GOOGLE_BOOKS_API_KEY or settings.GEMINI_API_KEY  # same Google Cloud project once Books API is on
    if key:
        params["key"] = key
    try:
        r = net.request("GET", "https://www.googleapis.com/books/v1/volumes", params=params, timeout=15)
    except httpx.TransportError:  # still timing out after net's retries (rate-limited): no candidates, not a dead packet
        return _failed()
    if r.status_code != 200:
        _gbooks_spent = r.status_code == 429 and "per day" in r.text
        return _failed()
    out = []
    for it in r.json().get("items", []):
        v = it["volumeInfo"]
        isbn = next((x["identifier"] for x in v.get("industryIdentifiers", []) if x["type"] == "ISBN_13"), "")
        out.append({"title": v.get("title", ""), "subtitle": v.get("subtitle", ""), "authors": v.get("authors", []),
                    "publisher": v.get("publisher", ""), "year": (v.get("publishedDate") or "")[:4], "isbn": isbn,
                    "url": v.get("infoLink", ""), "source": "Google Books"})
    return out


def _openlibrary(title, author):
    fields = "title,subtitle,author_name,publisher,first_publish_year,isbn,key"
    params = {"title": title, "limit": 5, "fields": fields} | ({"author": author} if author else {})
    try:
        # Open Library answers in 1-20 s; a slow one is a miss (not cached, a rerun retries), not 80 s of retries
        r = net.request("GET", "https://openlibrary.org/search.json", params=params, timeout=8, tries=2)
        if r.status_code == 200 and not r.json().get("docs"):  # fall back to free text
            r = net.request("GET", "https://openlibrary.org/search.json", timeout=8, tries=2,
                            params={"q": f"{title} {author}".strip(), "limit": 5, "fields": fields})
    except httpx.TransportError:
        return _failed()
    if r.status_code != 200:
        return _failed()
    out = []
    for d in r.json().get("docs", []):
        isbn = next((i for i in d.get("isbn", []) if len(i) == 13), "")
        out.append({"title": d.get("title", ""), "subtitle": d.get("subtitle", ""), "authors": d.get("author_name", []),
                    "publisher": (d.get("publisher") or [""])[0], "year": str(d.get("first_publish_year", "")),
                    "isbn": isbn, "url": f"https://openlibrary.org{d['key']}", "source": "Open Library"})
    return out


def score(read_title, read_author, agreement, cand):
    t = max(sim(read_title, cand["title"]), sim(read_title, f"{cand['title']} {cand['subtitle']}"))
    if read_author:
        a = max([sim(read_author, x) for x in cand["authors"]] +
                [1.0 for x in cand["authors"] if norm(read_author) == norm(x).split(" ")[-1]] + [0.0])
    else:
        a = 0.5
    return round(0.65 * t + 0.2 * a + 0.15 * agreement, 3)


def resolve(title, author, publisher, agreement, cache):
    """Catalogue lookup, then DREX gives a calibrated P(same work) per candidate. id_confidence = the best one."""
    key = f"{norm(title)}|{norm(author)}|{norm(publisher)}|{agreement:.2f}"
    if key not in cache:
        st, sa = search_terms(title, author)
        good = lambda cs: max((score(title, author, agreement, c) for c in cs), default=0) >= 0.8
        calls = [_gbooks(st, sa)]
        if not good(calls[0]):
            calls.append(_gbooks(st, sa, free=True))
        if not good(calls[0] + calls[-1]):
            calls.append(_openlibrary(st, sa))
        cands = [c for call in calls for c in call]
        seen = set()
        cands = [c for c in cands if not (c["url"] in seen or seen.add(c["url"]))]
        cands = sorted(cands, key=lambda c: -score(title, author, agreement, c))[:8]
        probs = {}
        if cands:
            state = {"spine_text_read": {"title": title, "author": author, "publisher": publisher},
                     "share_of_independent_reads_agreeing": round(agreement, 2),
                     "catalogue_candidates": {f"c{i}": {k: c[k] for k in ("title", "subtitle", "authors", "publisher", "year")}
                                              for i, c in enumerate(cands)}}
            probs = drex.ask(state, {f"c{i}": SAME_WORK_Q.format(c=f"c{i}") for i in range(len(cands))})
        entry = {"cands": cands, "probs": probs}
        if any(isinstance(c, Unavailable) for c in calls):
            return pick(title, author, entry)  # a failed source: answer this spine, ask again next time
        cache[key] = entry
    return pick(title, author, cache[key])


def pick(title, author, entry):
    cands, probs = entry["cands"], entry["probs"]
    if not cands:
        return None, 0.0, False
    if author_only(title, author, cands):
        return None, 0.0, True
    best = max(probs.get(f"c{i}", 0) for i in range(len(cands)))
    # editions of one work tie on P(same work): sweep 5ced read 'RAAVAN ENEMY OF ARYAVARTA' and got the Hindi edition
    # 'Raavan - Aryavart Ka Shatru' at 0.9805 over the English one at 0.958. Among near-ties, the closest title wins.
    near = [i for i in range(len(cands)) if probs.get(f"c{i}", 0) >= best - 0.05]
    i = max(near, key=lambda i: (score(title, author, 1.0, cands[i]), probs.get(f"c{i}", 0)))
    return cands[i], round(probs.get(f"c{i}", 0.0), 3), False


def author_only(title, author, cands):
    """Stacked books often show only the author's name. Read as a title it matches odd catalogue works literally named
    'Sidney Sheldon' or essays about the author (DREX P=0.99, checker 'match'): 18 of 123 IDs on a replayed shelf.
    A name with no title is not an identification."""
    t = norm(title)  # full name ("Sidney Sheldon") or surname alone ("SHELDON")
    return not norm(author) and any(t in (norm(a), norm(a).split(" ")[-1]) for c in cands for a in c["authors"] if norm(a))


# ---------- stage ----------

def ocr_support(cand, ocr, boiler):
    """Do the crop's own OCR words (any score) back this catalogue title? True: one is in the title. False: words, none
    in the title. None: no title-relevant words (blank crop, shelf boilerplate or the candidate's author only)."""
    from pipeline import spine_ocr  # imports this module
    toks = [x for x in spine_ocr.tokens(" ".join(w[0] for w in ocr["words"]))
            if x not in boiler and not spine_ocr.supports(" ".join(cand["authors"]), [x])]
    return bool(spine_ocr.supports(cand["title"], toks)) if toks else None


def crop_reader(t, ocr, boiler, cache):
    """Last resort for a spine still unidentified: the vision model reads the crop alone. Returns
    (cand, conf, ocr_support). OCR words that contradict the title reject it; no title-relevant OCR words at all ->
    accepted at most at CONF_MIN and sent to review: flash-lite invents titles under price stickers."""
    if not norm(t["title"]):
        return None, 0.0, False
    cand, conf, _ = resolve(t["title"], t["author"], t["publisher"], 1.0, cache)  # one read agrees with itself
    if not (cand and conf >= settings.CONF_MIN):
        return None, 0.0, False
    support = ocr_support(cand, ocr, boiler)
    if support is None:
        return cand, min(conf, settings.CONF_MIN), False
    return (cand, conf, True) if support else (None, 0.0, False)


def run(sweep_dir, reads, geo, spines=None):
    """spines: {frame: [SAM detections]} from the spines stage, or None to use the vision model's boxes."""
    times = frame_times(sweep_dir)
    src, unmatched = claims(spines, reads) if spines is not None else (reads, 0)
    dets = detections(src, times, geo)
    with_3d = [d for d in dets if "center" in d]
    if geo.ok and with_3d:
        clusters, dropped = seen_once(merge(cluster(dets)), len({d["frame"] for d in with_3d}))
        assign_shelves(clusters)
    else:
        clusters, dropped = layout_2d(dets), []
    det_to_c = {m["id"]: c for c in clusters for m in c["members"]}

    (sweep_dir / "crops").mkdir(exist_ok=True)
    clusters.sort(key=lambda c: (c["unit"], c["row"], c["position"]))
    rereads = []
    for n, c in enumerate(clusters, 1):
        c["id"] = f"b_{n:03d}"
        # the crop OCR reads: the best read; among those, a view clear of neighbouring boxes if one is at least half
        # the size of the largest (a clear view is often a far, blurred one); then the largest
        big = max(m["area"] for m in c["members"])
        best = max(c["members"], key=lambda m: (LEG_W[m["legibility"]], m["overlap"] < 0.2 and m["area"] >= big / 2,
                                                m["area"]))
        c["best"] = best
        c["crop_bytes"] = crop(sweep_dir, best["frame"], best["box_2d"], sweep_dir / "crops" / f"{c['id']}.jpg")
        c["texts"] = [(m["title"], m["author"], m["publisher"], LEG_W[m["legibility"]]) for m in c["members"]]
        # SAM path: no reread of claims; the vision model only reads crops OCR could not (crop_reader, below)
        if spines is None and any(norm(t[0]) for t in c["texts"]) and (len(c["members"]) == 1 or vote(c["texts"])[3] < 1):
            rereads.append(c)
    batches = [rereads[i:i + 12] for i in range(0, len(rereads), 12)]
    with ThreadPoolExecutor(4) as ex:  # one batch at a time took minutes on a 150-book shelf
        for batch, texts in zip(batches, ex.map(lambda b: vision.read_crops([c["crop_bytes"] for c in b]), batches)):
            for c, t in zip(batch, texts):
                c["texts"].append((t["title"], t["author"], t["publisher"], LEG_W[t["legibility"]]))

    from pipeline import spine_ocr  # imports this module
    ocr_p = sweep_dir / "ocr.json"  # keyed by best detection, so reruns from this stage do not re-OCR
    ocr_cache = json.loads(ocr_p.read_text()) if ocr_p.exists() else {}
    todo = [c for c in clusters if c["best"]["id"] not in ocr_cache]

    def run_ocr():
        with metrics.stage("ocr"):
            for c, o in zip(todo, spine_ocr.read([c["crop_bytes"] for c in todo])):
                if o["rotation"] is not None or not settings.OCR_ENABLED:
                    ocr_cache[c["best"]["id"]] = o
                c["ocr"] = o
    # OCR runs while the catalogue resolves the claims (it waited ~25 s in front of them)
    ocr_pool = ThreadPoolExecutor(1)
    ocr_done = ocr_pool.submit(run_ocr)

    for note in load_notes(sweep_dir):
        if note.get("target", "book") != "book" or note["kind"] not in ("note", "exclude"):
            continue
        did = nearest_det(src, times, note["t"], "spines")
        c = det_to_c.get(did)
        if not c:
            continue
        if note["kind"] == "note":
            c.setdefault("notes", []).append(note["text"])
        else:
            for o in clusters:
                if o["shelf"] == c["shelf"]:
                    o["excluded"] = note.get("text") or "user asked to skip this shelf"

    cache_p = sweep_dir / "catalog_cache.json"
    cache = json.loads(cache_p.read_text()) if cache_p.exists() else {}
    votes = [vote(c["texts"]) for c in clusters]
    with ThreadPoolExecutor(8) as ex:  # ponytail: shared cache dict, distinct keys per book
        resolved = list(ex.map(lambda v: resolve(v[0], v[1], v[2], v[3], cache) if v[0] else (None, 0.0, False), votes))
    ocr_done.result()
    ocr_pool.shutdown()
    ocr_p.write_text(json.dumps(ocr_cache))
    for c in clusters:
        c["ocr"] = ocr_cache.get(c["best"]["id"], c.get("ocr", {"words": [], "rotation": None}))
    boiler = spine_ocr.boilerplate([c["ocr"] for c in clusters])

    # OCR witness: veto claims the spine's own text contradicts, then try fragments on spines still unread
    witness = []
    said = [f"{t} {a}" for t, a, _, _ in votes]
    for i, (c, (title, author, _, _), (cand, conf, name_only)) in enumerate(zip(clusters, votes, resolved)):
        neighbours = [r for j, r in enumerate(said) if j != i and titled(votes[j][0]) and not related(votes[j][0], title)]
        witness.append((*spine_ocr.witness(cand, conf, c["ocr"], boiler, author, cache, bool(c.get("excluded")),
                                           neighbours), name_only))
    witness = [(cand, conf, name_only, w) for cand, conf, w, name_only in witness]
    # a fragment ID equal to an adjacent spine's title is a crop that drifted onto the neighbour: drop it
    for i, (cand, conf, name_only, w) in enumerate(witness):
        if w["id_source"] != "ocr_fragments":
            continue
        for j in (i - 1, i + 1):
            o = witness[j] if 0 <= j < len(witness) else None
            if o and o[0] and clusters[j]["shelf"] == clusters[i]["shelf"] and sim(o[0]["title"], cand["title"]) >= 0.9 \
                    and o[3]["id_source"] == "spine_read":
                witness[i] = (None, 0.0, name_only, {**w, "id_source": "", "dropped_as_neighbour": cand["title"]})
                break
    if spines is not None:  # a claim is the vision model's word: without OCR support it is kept but sent to review
        for i, (cand, conf, name_only, w) in enumerate(witness):
            if w["id_source"] == "spine_read":
                support = bool(ocr_support(cand, clusters[i]["ocr"], boiler))
                witness[i] = (cand, conf if support else min(conf, settings.CONF_MIN), name_only,
                              {**w, "ocr_support": support})
    if spines is not None:  # crop reader: every spine still unidentified and not excluded; its OCR words must agree
        todo = [i for i, (c, (cand, conf, *_)) in enumerate(zip(clusters, witness))
                if not (cand and conf >= settings.CONF_MIN) and not c.get("excluded")]
        batches = [todo[k:k + 12] for k in range(0, len(todo), 12)]
        with ThreadPoolExecutor(4) as ex:
            reads_ = [(i, t) for batch, texts in zip(batches, ex.map(
                lambda b: vision.read_crops([clusters[i]["crop_bytes"] for i in b]), batches)) for i, t in zip(batch, texts)]
        with ThreadPoolExecutor(8) as ex:  # their catalogue lookups ran one at a time: ~100 s on a 30-spine sweep
            for (i, t), (cand, conf, support) in zip(reads_, ex.map(
                    lambda it: crop_reader(it[1], clusters[it[0]]["ocr"], boiler, cache), reads_)):
                if cand:
                    witness[i] = (cand, conf, False, {**witness[i][3], "id_source": "crop_reader",
                                                      "crop_read": t["title"], "ocr_support": support})
    resolved = [x[:3] for x in witness]
    books = []
    for c, (title, author, publisher, agreement), (cand, conf, name_only), (*_, w) in zip(clusters, votes, resolved, witness):
        hs = [m["height_cm"] for m in c["members"] if "height_cm" in m]
        ts = [m["thickness_cm"] for m in c["members"] if "thickness_cm" in m]
        marks = sorted({x for m in c["members"] for x in m["marks"]})
        notes = c.get("notes", [])
        publisher_ok = cand and publisher and sim(publisher, cand["publisher"]) >= 0.7
        books.append({
            "id": c["id"], "shelf": c["shelf"], "position": c["position"], "orientation": c["orientation"],
            "frame_ref": f"frames/{c['best']['frame']}#t={c['best']['t']}", "crop": f"crops/{c['id']}.jpg",
            "views": len(c["members"]), "spine_text": {"title": title, "author": author, "publisher": publisher},
            "read_agreement": round(agreement, 2), "author_only": name_only,
            "ocr_text": spine_ocr.ocr_text(c["ocr"]), "ocr_rotation": c["ocr"]["rotation"], **w,
            "legibility": LEGIBLE[min(len(spine_ocr.distinct(c["ocr"], boiler)), 3)],
            "title": cand["title"] if cand and conf >= settings.CONF_MIN else "",
            "author": ", ".join(cand["authors"]) if cand and conf >= settings.CONF_MIN else "",
            "edition": f"{cand['publisher']} {cand['year']}".strip() if publisher_ok and conf >= settings.CONF_MIN else "",
            "isbn": cand["isbn"] if publisher_ok and conf >= settings.CONF_MIN else "",
            "catalog": {"source": cand["source"], "url": cand["url"]} if cand and conf >= settings.CONF_MIN else None,
            "id_confidence": conf if cand else 0.0,
            "spine_height_cm": round(float(np.median(hs)), 1) if hs else None,
            "spine_thickness_cm": round(float(np.median(ts)), 1) if ts else None,
            "marks": marks, "notes": notes,
            "excluded": c.get("excluded", ""),
            "_center": [round(float(v), 3) for v in c["center"]],
        })
    cache_p.write_text(json.dumps(cache))
    books, folded, fragments = same_book(books) if geo.ok and with_3d else (books, 0, 0)
    books, over_cap = frame_cap(books, clusters, reads) if geo.ok and with_3d and spines is not None else (books, 0)
    from pipeline import reconcile  # imports vision
    books, rec = reconcile.run(sweep_dir, books, book_frames(books, clusters)) if geo.ok and with_3d else (books, [])
    kept = [b for b in books if not b["excluded"]]
    stats = {"detections": len(dets), "detections_without_3d": len(dets) - len(with_3d),
             "shelf_run_m": shelf_run_m(kept), "excluded_by_user": len(books) - len(kept),
             "spine_source": "vision" if spines is None else "sam", "vision_spines_unmatched": unmatched,
             "seen_once_dropped": len(dropped), "same_book_folded": folded,
             "partial_boxes_dropped": fragments, "over_frame_cap_dropped": over_cap,
             "reconciled": [{"id": i, "action": k, "into": into, "reason": r} for i, k, into, r in rec]}
    if not geo.ok:
        stats["warning"] = "no metric geometry: books could not be deduplicated or measured"
    return {"books": books, "stats": stats}
