"""Spine OCR witness. A detector + recognizer (PP-OCRv5, gpu/modal_ocr.py) reads each spine crop; it cannot invent a
title, so its words check what the vision model claims and recover titles a price sticker half covers.

Two decisions, both made in code from the OCR words:
- veto: the spine shows distinctive text and none of it is in the claimed title -> the claim is blanked.
  flash-lite read 'TO… ROW' (Gone Tomorrow) as 'Die Trying' and 'EVA…' as 'Evil Under the Sun' at P>0.98.
- fragments: an unread spine with enough distinctive text ('HERCULE POIRO… CHRISTM…') is looked up in the catalogue;
  a candidate must contain most fragments, then DREX gives P(same work). 'EVA' alone is too little and stays blank.
"""
import difflib
import re
import time
from collections import Counter

from pipeline import books, drex, metrics, settings

STOP = {"the", "and", "for", "with", "from", "his", "her", "you", "are", "not", "but", "was", "all", "this", "that"}


def tokens(text):
    return [w for w in books.norm(text.replace("'", "")).split() if len(w) >= 3 and w not in STOP]


def _match(tok, word):
    if tok == word:
        return True
    if len(tok) >= 3 and (tok in word or (len(word) >= 3 and tok.startswith(word))):
        return True  # stickers cut words anywhere: 'ROW' -> tomorrow, 'POIRO' -> poirots, 'SADCY' -> sad
    return len(tok) >= 4 and len(word) >= 4 and difflib.SequenceMatcher(None, tok, word).ratio() >= 0.75


def read(crops):
    """OCR every crop on the Modal CPU service, 16 per call in parallel. Any failure -> empty reads: no veto, no
    fragments, the checker still runs. Never fails the packet."""
    empty = [{"words": [], "rotation": None} for _ in crops]
    if not crops or not settings.OCR_ENABLED:
        return empty
    try:
        from gpu.modal_ocr import remote
        chunks = [crops[i:i + 16] for i in range(0, len(crops), 16)]
        t0 = time.perf_counter()
        out = [r for part in remote().read.map(chunks) for r in part]
        metrics.count("ocr_crops", len(crops))
        metrics.count("ocr_seconds", time.perf_counter() - t0)
        return out
    except Exception:
        metrics.count("ocr_failures")
        return empty


def boilerplate(ocr_list):
    """Words printed on many spines of this shelf (series, imprint, a shared author: 'LEE CHILD', 'REACHER') say
    nothing about which book this is. Learned per sweep, no word list."""
    df = Counter(t for o in ocr_list for t in {t for w in o["words"] for t in tokens(w[0])})
    floor = max(3, 0.15 * len(ocr_list))  # two series on one 40-spine shelf: each series word is on ~15-40%
    core = {t for t, n in df.items() if n >= floor}
    # script fonts garble differently on every spine ('agalla chuistie', 'agallachustie'): near-copies count too
    joined = {a + b for a in core for b in core if a != b}
    return core | {t for t in df if any(difflib.SequenceMatcher(None, t, c).ratio() >= 0.8 for c in core | joined)}


def distinct(ocr, boiler, author="", min_score=None, min_len=3):
    """Confident words on this spine that are neither shelf boilerplate nor the author's name."""
    a = set(tokens(author))
    out = []
    for w in ocr["words"]:
        if w[1] < (settings.OCR_MIN_SCORE if min_score is None else min_score):
            continue
        for t in (t for t in tokens(w[0]) if len(t) >= min_len):
            # the author filter must be near-exact: 'CHRISTM' (Christmas) is 80% like 'christie'; garbled author
            # spellings are already shelf boilerplate
            if t not in boiler and not any(difflib.SequenceMatcher(None, t, x).ratio() >= 0.9 for x in a) and t not in out:
                out.append(t)
    return out


def supports(title, toks):
    """How many OCR tokens appear in the title. OCR often runs words together ('THETUNNELOFTIME', 'iWoz'), so a
    token also counts when it contains, or sits inside, the title with its spaces removed."""
    words, joined = tokens(title), books.norm(title).replace(" ", "")
    return sum(1 for t in toks if any(_match(t, w) for w in words)
               or (len(t) >= 3 and t in joined) or (len(joined) >= 5 and joined in t))


def veto(title, toks):
    return bool(toks) and supports(title, toks) == 0


def enough(toks):
    return len(toks) >= 2 or sum(map(len, toks)) >= 8


def same_work(a, b):
    """Two catalogue titles for one work: editions add '(Poirot)' or a subtitle, which is no rival candidate.
    'Curtain' and 'Curtain Call' stay different works."""
    main = lambda t: books.norm(re.split(r"[:(\[]| - ", t)[0])
    return books.sim(a, b) >= 0.9 or books.sim(main(a), main(b)) >= 0.9


def fragment_lookup(toks, author, cache):
    """Catalogue candidates for an unread spine, from its OCR fragments. Returns (candidate, P, used_tokens)."""
    if not enough(toks):
        return None, 0.0, []
    q = " ".join(toks[:6])
    key = f"ocrfrag|{q}|{books.norm(author)}"
    if key not in cache:
        _, a = books.search_terms(q, author)
        calls = [books._gbooks(q, a, free=True), books._openlibrary(q, a)]
        cands = calls[0] + calls[1]
        seen, keep = set(), []
        for c in cands:
            n = supports(c["title"], toks)  # title only: subtitles carry series text ('A Jack Reacher Thriller')
            if c["url"] in seen or n < min(2, len(toks)) or n < 0.5 * len(toks):
                continue  # half the fragments must be in the title: 'EVA' + noise never picks a book
            seen.add(c["url"])
            keep.append(c)
        keep = keep[:6]
        probs = {}
        if keep:
            state = {"spine_text_ocr_fragments": toks, "author_read": author,
                     "catalogue_candidates": {f"c{i}": {k: c[k] for k in ("title", "subtitle", "authors", "publisher", "year")}
                                              for i, c in enumerate(keep)}}
            probs = drex.ask(state, {f"c{i}": books.SAME_WORK_Q.format(c=f"c{i}") for i in range(len(keep))})
        entry = {"cands": keep, "probs": probs}
        if not any(isinstance(c, books.Unavailable) for c in calls):  # a failed catalogue is not 'no such book'
            cache[key] = entry
    else:
        entry = cache[key]
    keep, probs = entry["cands"], entry["probs"]
    if not keep:
        return None, 0.0, toks
    ranked = sorted(range(len(keep)), key=lambda i: -probs.get(f"c{i}", 0))
    p1 = probs.get(f"c{ranked[0]}", 0.0)
    p2 = max((probs.get(f"c{i}", 0.0) for i in ranked[1:] if not same_work(keep[i]["title"], keep[ranked[0]]["title"])),
             default=0.0)
    if p1 - p2 < 0.3:  # series words fit many books ('HERCULE POIRO'): no clear winner, no identification
        return None, 0.0, toks
    return keep[ranked[0]], round(p1, 3), toks


def witness(cand, conf, ocr, boiler, author, cache, excluded=False):
    """The OCR decision for one spine, shared by the pipeline and the photo scorer. Returns (cand, conf, info)."""
    toks = distinct(ocr, boiler, author)
    # veto evidence must be strong: 'NON' (0.90, upside-down read) and 'WON' (0.83) vetoed No Middle Name;
    # the candidate's own author on the spine ('JHUMPA LAHIRI' on The Lowland) supports it, never contradicts
    strong = distinct(ocr, boiler, author, min_score=settings.OCR_VETO_SCORE, min_len=4)
    w = {"ocr_veto": "", "id_source": "spine_read" if cand and conf >= settings.CONF_MIN else "", "fragments": []}
    if cand and conf >= settings.CONF_MIN and veto(f"{cand['title']} {cand.get('subtitle', '')} {' '.join(cand.get('authors', []))}", strong):
        w.update(ocr_veto=cand["title"], veto_tokens=strong, id_source="")
        cand, conf = None, 0.0
    if not (cand and conf >= settings.CONF_MIN) and not excluded:
        f_cand, f_conf, used = fragment_lookup(toks, author, cache)
        if f_cand and f_conf >= settings.CONF_MIN:
            cand, conf = f_cand, f_conf
            w.update(id_source="ocr_fragments", fragments=used)
    return cand, conf, w


def ocr_text(ocr):
    return " ".join(w[0] for w in ocr["words"] if w[1] >= settings.OCR_MIN_SCORE)
