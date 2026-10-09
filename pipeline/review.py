"""Review: independent checker on identified spines, then the rule list that builds the review queue."""
from concurrent.futures import ThreadPoolExecutor

from pipeline import metrics, settings, vision

# spines per checker call (free tier: 15 calls/min). On the 62 photo identifications, one per call took 288 s and caught
# 3 of 5 hallucinations (+1 unsure); batches of 8 took 37-98 s and caught 2 and 5 of 5 on two runs: within run noise
CHECK_BATCH = 8


def checker(sweep_dir, books):
    """A different model sees only the crop and the claim. Mismatch -> identity cleared (blank beats wrong).
    A failed check (spent daily quota, outage) is 'unsure', so the book goes to review: on a replay the checker's
    500/day free quota ran out mid-sweep and killed a packet whose books were already built."""
    todo = [b for b in books if b["title"] and not b["excluded"]]
    batches = [todo[i:i + CHECK_BATCH] for i in range(0, len(todo), CHECK_BATCH)]

    def batch(bs):
        try:
            return vision.check_identities([((sweep_dir / b["crop"]).read_bytes(), b["title"], b["author"]) for b in bs])
        except Exception as e:  # noqa: BLE001 - any failure of the audit is 'not audited', never a dead packet
            metrics.count("checker_failures", len(bs))
            return [{"verdict": "unsure", "reason": f"checker unavailable ({type(e).__name__})"}] * len(bs)

    with ThreadPoolExecutor(4) as ex:
        for bs, vs in zip(batches, ex.map(batch, batches)):
            for b, v in zip(bs, vs):
                b["checker"] = dict(v)
                if v["verdict"] == "mismatch":
                    b["checker_cleared"] = {"title": b["title"], "author": b["author"]}
                    b.update(title="", author="", edition="", isbn="", catalog=None)
    return books


def queue(geometry, books, items, book_prices, item_prices, book_stats):
    q = []
    add = lambda ref, reason: q.append({"ref_id": ref, "reason": reason})

    if not geometry.get("scale"):
        add("room", "no metric scale: marker not seen and too few measurable spines")
    elif "prior" in geometry.get("scale_method", ""):
        add("room", "scale from book-format prior only (marker not seen): all dimensions approximate")
    for n in geometry.get("notes", []):
        add("room", n)
    if geometry.get("confidence", 0) < 0.6:
        add("room", f"room measurement confidence {geometry.get('confidence', 0)}")
    if "polygon" in geometry.get("shape_method", "") or "unseen" in geometry.get("shape_method", ""):
        add("room", geometry["shape_method"])
    if book_stats.get("partial_boxes_dropped"):
        add("books", f"{book_stats['partial_boxes_dropped']} unread spine boxes under 10 cm long not counted as books "
                     "(pieces of a spine): check the sweep if a small book is missing")
    if book_stats.get("over_frame_cap_dropped"):
        add("books", f"{book_stats['over_frame_cap_dropped']} unread spine lines not counted: the unit "
                     "showed fewer spines in one full view than the 3D count")
    for r in book_stats.get("reconciled", []):
        add(r["into"] or "books", f"line {r['id']} " + (f"merged into {r['into']}" if r["into"] else
                                                         f"not counted ({r['action'].replace('_', ' ')})")
            + f" by the reconcile check: {r['reason']}")
    if "warning" in book_stats:
        add("books", book_stats["warning"])
    if book_stats.get("spine_source") == "vision":
        add("books", "spines from vision model (SAM not used): boxes may drift onto the neighbouring spine")

    heights = sorted(b["spine_height_cm"] for b in books if b["spine_height_cm"] and b["orientation"] == "upright")
    if heights and not 14 <= heights[len(heights) // 2] <= 32:
        add("room", f"scale cross-check failed: median upright spine {heights[len(heights) // 2]} cm is outside "
                    "standard book formats (14-32 cm)")

    for b in books:
        if b["excluded"]:
            continue
        p = book_prices.get(b["id"], {})
        if b.get("checker_cleared"):
            add(b["id"], f"checker rejected '{b['checker_cleared']['title']}': {b['checker']['reason']}")
        elif b.get("checker", {}).get("verdict") == "unsure":
            add(b["id"], f"checker unsure: {b['checker']['reason']}")
        if b.get("ocr_veto"):
            add(b["id"], f"spine OCR reads '{' '.join(b.get('veto_tokens', []))}', which does not support "
                         f"'{b['ocr_veto']}': identity cleared")
        if b.get("ocr_support") is False:  # claim or crop read the spine's own OCR words do not back
            add(b["id"], "title from vision model only, no OCR support")
        if b.get("id_source") == "ocr_fragments":
            add(b["id"], f"identified from partial spine text '{'… '.join(b['fragments'])}…': check the title")
        if not b["title"] and not b.get("checker_cleared"):
            add(b["id"], "spine not readable: logged as unidentified with dimensions only"
                if not b["spine_text"]["title"] else
                f"only the author's name is legible ('{b['spine_text']['title']}'): title unknown, not guessed"
                if b.get("author_only") else
                f"spine text '{b['spine_text']['title']}' did not match a catalogue record confidently")
        elif b["title"] and b["id_confidence"] < settings.CONF_IDENTIFIED:
            add(b["id"], f"low identification confidence {b['id_confidence']}")
        for r in p.get("reasons", []):
            add(b["id"], r)
        if p.get("status") == "identified" and not p.get("replacement_cost"):
            add(b["id"], "no retrievable replacement price: excluded from totals")
        if p.get("status") == "identified" and not p.get("used_value"):
            add(b["id"], "no retrievable used price: excluded from used total")
        if b["spine_height_cm"] is None:
            add(b["id"], "no depth at this spine: dimensions unavailable")
        elif not 8 <= b["spine_height_cm"] <= 50:
            add(b["id"], f"implausible spine height {b['spine_height_cm']} cm")

    for it in items:
        p = item_prices.get(it["id"], {})
        for r in p.get("reasons", []):
            add(it["id"], r)
        if p.get("status") == "range" and p.get("replacement_cost"):
            add(it["id"], "brand/model not visible: priced as a range of comparable items")
        if it["confidence"] < 0.6:
            add(it["id"], f"low detection confidence {it['confidence']}")
    return q
