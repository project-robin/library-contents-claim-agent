"""Books stage: per-frame spine detections -> one line per physical book, with shelf, dims and identity.

Dedup is spatial: the same spine seen in many frames lands at the same 3D point. Identity comes from a catalogue
match of the transcribed spine text; a spine with no readable text stays unidentified (title blank, dims kept).
"""
import difflib
import json
import re
from collections import Counter

import cv2
import numpy as np

from concurrent.futures import ThreadPoolExecutor

from pipeline import drex, metrics, net, settings, vision
from pipeline.geometry import box_px

LEG_W = {"full": 2, "partial": 1, "none": 0}
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
    dets = []
    for f, r in reads.items():
        for k, s in enumerate(r["spines"]):
            d = {"id": f"{f}:{k}", "frame": f, "t": times.get(f), **s,
                 "area": (s["box_2d"][2] - s["box_2d"][0]) * (s["box_2d"][3] - s["box_2d"][1])}
            m = geo.spine(f, s["box_2d"], s["orientation"]) if geo.ok else None
            if m:
                d.update(m)
            dets.append(d)
    return dets


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
            both = bool(norm(c["title"]) and norm(d["title"]))
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
            if not norm(best["title"]):
                best["title"] = d["title"]
    return clusters


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


def _gbooks(title, author, free=False):
    """free=True: plain-text query. intitle:/inauthor: returns nothing for some Indian editions ('An Unsuitable Boy',
    'Dream With Your Eyes Open' with a misspelt author) that a plain query finds first."""
    q = f"{title} {author}".strip() if free else f"intitle:{title}" + (f" inauthor:{author}" if author else "")
    params = {"q": q, "maxResults": 5, "printType": "books"}
    key = settings.GOOGLE_BOOKS_API_KEY or settings.GEMINI_API_KEY  # same Google Cloud project once Books API is on
    if key:
        params["key"] = key
    r = net.request("GET", "https://www.googleapis.com/books/v1/volumes", params=params, timeout=15)
    out = []
    for it in r.json().get("items", []) if r.status_code == 200 else []:
        v = it["volumeInfo"]
        isbn = next((x["identifier"] for x in v.get("industryIdentifiers", []) if x["type"] == "ISBN_13"), "")
        out.append({"title": v.get("title", ""), "subtitle": v.get("subtitle", ""), "authors": v.get("authors", []),
                    "publisher": v.get("publisher", ""), "year": (v.get("publishedDate") or "")[:4], "isbn": isbn,
                    "url": v.get("infoLink", ""), "source": "Google Books"})
    return out


def _openlibrary(title, author):
    fields = "title,subtitle,author_name,publisher,first_publish_year,isbn,key"
    params = {"title": title, "limit": 5, "fields": fields} | ({"author": author} if author else {})
    r = net.request("GET", "https://openlibrary.org/search.json", params=params, timeout=20)
    if r.status_code == 200 and not r.json().get("docs"):  # fall back to free text
        r = net.request("GET", "https://openlibrary.org/search.json", timeout=20,
                      params={"q": f"{title} {author}".strip(), "limit": 5, "fields": fields})
    out = []
    for d in r.json().get("docs", []) if r.status_code == 200 else []:
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
        cands = _gbooks(st, sa)
        if not good(cands):
            cands += _gbooks(st, sa, free=True)
        if not good(cands):
            cands += _openlibrary(st, sa)
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
        cache[key] = {"cands": cands, "probs": probs}
    cands, probs = cache[key]["cands"], cache[key]["probs"]
    if not cands:
        return None, 0.0, False
    if author_only(title, author, cands):
        return None, 0.0, True
    i = max(range(len(cands)), key=lambda i: probs.get(f"c{i}", 0))
    return cands[i], round(probs.get(f"c{i}", 0.0), 3), False


def author_only(title, author, cands):
    """Stacked books often show only the author's name. Read as a title it matches odd catalogue works literally named
    'Sidney Sheldon' or essays about the author (DREX P=0.99, checker 'match'): 18 of 123 IDs on a replayed shelf.
    A name with no title is not an identification."""
    t = norm(title)  # full name ("Sidney Sheldon") or surname alone ("SHELDON")
    return not norm(author) and any(t in (norm(a), norm(a).split(" ")[-1]) for c in cands for a in c["authors"] if norm(a))


# ---------- stage ----------

def run(sweep_dir, reads, geo):
    times = frame_times(sweep_dir)
    dets = detections(reads, times, geo)
    with_3d = [d for d in dets if "center" in d]
    clusters = cluster(dets)
    assign_shelves(clusters)
    det_to_c = {m["id"]: c for c in clusters for m in c["members"]}

    (sweep_dir / "crops").mkdir(exist_ok=True)
    clusters.sort(key=lambda c: (c["unit"], c["row"], c["position"]))
    rereads = []
    for n, c in enumerate(clusters, 1):
        c["id"] = f"b_{n:03d}"
        best = max(c["members"], key=lambda m: (LEG_W[m["legibility"]], m["area"]))
        c["best"] = best
        c["crop_bytes"] = crop(sweep_dir, best["frame"], best["box_2d"], sweep_dir / "crops" / f"{c['id']}.jpg")
        c["texts"] = [(m["title"], m["author"], m["publisher"], LEG_W[m["legibility"]]) for m in c["members"]]
        if any(norm(t[0]) for t in c["texts"]) and (len(c["members"]) == 1 or vote(c["texts"])[3] < 1):
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
    with metrics.stage("ocr"):
        for c, o in zip(todo, spine_ocr.read([c["crop_bytes"] for c in todo])):
            if o["rotation"] is not None or not settings.OCR_ENABLED:
                ocr_cache[c["best"]["id"]] = o
            c["ocr"] = o
    ocr_p.write_text(json.dumps(ocr_cache))
    for c in clusters:
        c["ocr"] = ocr_cache.get(c["best"]["id"], c.get("ocr", {"words": [], "rotation": None}))
    boiler = spine_ocr.boilerplate([c["ocr"] for c in clusters])

    for note in load_notes(sweep_dir):
        if note.get("target", "book") != "book" or note["kind"] not in ("note", "exclude"):
            continue
        did = nearest_det(reads, times, note["t"], "spines")
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

    # OCR witness: veto claims the spine's own text contradicts, then try fragments on spines still unread
    witness = []
    for c, (title, author, _, _), (cand, conf, name_only) in zip(clusters, votes, resolved):
        witness.append((*spine_ocr.witness(cand, conf, c["ocr"], boiler, author, cache, bool(c.get("excluded"))),
                        name_only))
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
    kept = [b for b in books if not b["excluded"]]
    stats = {"detections": len(dets), "detections_without_3d": len(dets) - len(with_3d),
             "shelf_run_m": shelf_run_m(kept), "excluded_by_user": len(books) - len(kept)}
    if not geo.ok:
        stats["warning"] = "no metric geometry: books could not be deduplicated or measured"
    return {"books": books, "stats": stats}
