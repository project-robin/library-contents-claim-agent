"""Identification on SAM spines, offline: claim matching, the still-photo layout and the crop reader's rules."""
import json

import cv2
import numpy as np

from pipeline import books, drex, review, settings, spine_ocr, vision


class NoGeo:
    ok = False


def sam(box, i):
    return {"box_2d": box, "orientation": "upright", "legibility": "none", "title": "", "author": "",
            "publisher": "", "marks": [], "source": "sam", "sam_id": i}


def gem(box, title, author="Lee Child"):
    return {"box_2d": box, "orientation": "upright", "legibility": "full", "title": title, "author": author,
            "publisher": "", "marks": []}


def stub_catalogue(monkeypatch):
    monkeypatch.setattr(books, "_gbooks", lambda t, a, free=False: [
        {"title": t, "subtitle": "", "authors": [a], "publisher": "", "year": "2001", "isbn": "", "url": f"u/{t}",
         "source": "stub"}])
    monkeypatch.setattr(books, "_openlibrary", lambda t, a: [])
    monkeypatch.setattr(drex, "ask", lambda state, qs: {k: 0.95 for k in qs})


def still_sweep(tmp_path):
    sd = tmp_path / "sweep"
    (sd / "frames").mkdir(parents=True)
    img = np.random.default_rng(0).integers(0, 255, (1000, 1000, 3), np.uint8)  # 1000 px: box_2d == pixels
    cv2.imwrite(str(sd / "frames" / "000001.jpg"), img)
    (sd / "frames.jsonl").write_text(json.dumps({"file": "000001.jpg", "t": 0.0, "key": True}))
    return sd


def test_claims_greedy_one_to_one_and_threshold():
    a, b = sam([100, 100, 400, 200], 0), sam([100, 200, 400, 300], 1)
    wide = gem([100, 140, 400, 270], "Wide")  # IoU 0.35 with a, 0.44 with b: claims b only
    far = gem([600, 600, 700, 700], "Far")  # overlaps nothing: dropped
    out, unmatched = books.claims({"f": [a, b]}, {"f": {"spines": [wide, far]}})
    da, db = out["f"]["spines"]
    assert da["title"] == "" and "claim_iou" not in da
    assert db["title"] == "Wide" and db["legibility"] == "full" and db["claim_iou"] == 0.44
    assert unmatched == 1
    low = gem([100, 160, 400, 260], "Low")  # IoU 0.25 with a, 0.43 with b; a alone below 0.3 stays unclaimed
    out, _ = books.claims({"f": [a]}, {"f": {"spines": [low]}})
    assert out["f"]["spines"][0]["title"] == ""
    assert a["title"] == ""  # inputs are not mutated


def test_crop_reader_rules(monkeypatch):
    stub_catalogue(monkeypatch)
    t = {"title": "Curtain", "author": "Agatha Christie", "publisher": "", "legibility": "partial"}
    words = lambda *ws: {"words": [[w, 0.5, 90] for w in ws], "rotation": 90}
    cand, conf, support = books.crop_reader(t, words("CURTAI"), set(), {})
    assert cand["title"] == "Curtain" and conf == 0.95 and support  # OCR fragment supports the title
    assert books.crop_reader(t, words("FROG"), set(), {})[0] is None  # OCR contradicts: rejected
    cand, conf, support = books.crop_reader(t, words(), set(), {})
    assert cand and conf == settings.CONF_MIN and not support  # OCR read nothing: capped, sent to review
    cand, conf, support = books.crop_reader(t, words("AGATHA", "CHRISTIE"), set(), {})
    assert cand and conf == settings.CONF_MIN and not support  # only the author's name: neither for nor against
    assert books.crop_reader({**t, "title": ""}, words(), set(), {})[0] is None
    cand, conf, support = books.crop_reader(t, words("REACHER"), {"reacher"}, {})
    assert cand and conf == settings.CONF_MIN and not support  # shelf boilerplate: neither for nor against


def test_still_photo_with_sam_spines(tmp_path, monkeypatch):
    sd = still_sweep(tmp_path)
    stub_catalogue(monkeypatch)
    # two rows, given out of order; the row-1 boxes jitter vertically
    boxes = [[100, 500, 400, 600], [100, 100, 400, 200], [110, 300, 390, 400], [550, 400, 900, 500], [560, 100, 900, 200]]
    spines = {"000001.jpg": [sam(b, i) for i, b in enumerate(boxes)]}
    reads = {"000001.jpg": {"spines": [gem([100, 105, 400, 205], "Tripwire"), gem([110, 300, 390, 400], "Die Trying"),
                                       gem([0, 900, 50, 1000], "Off Shelf")], "items": []}}
    # reading order: sam 1, 2, 0 (row 1) then 4, 3 (row 2)
    ocr = [[["TRIPWIRE", 0.95, 90]], [["GONE TOMORROW", 0.95, 90]], [], [["XQZT", 0.95, 90]], [["FROG", 0.5, 90]]]
    monkeypatch.setattr(spine_ocr, "read", lambda crops: [{"words": w, "rotation": 90} for w in ocr])
    sent = []

    def read_crops(js):
        sent.append(len(js))
        return [{"title": "Curtain", "author": "", "publisher": "", "legibility": "partial"} for _ in js]
    monkeypatch.setattr(vision, "read_crops", read_crops)

    out = books.run(sd, reads, NoGeo(), spines)
    bs = out["books"]
    assert [(b["shelf"], b["position"]) for b in bs] == [("A-1", 1), ("A-1", 2), ("A-1", 3), ("A-2", 1), ("A-2", 2)]
    assert [b["title"] for b in bs] == ["Tripwire", "gone tomorrow", "Curtain", "", ""]
    assert [b["id_source"] for b in bs] == ["spine_read", "ocr_fragments", "crop_reader", "", ""]
    assert bs[1]["ocr_veto"] == "Die Trying"  # the claim the spine's own words contradict
    assert bs[2]["id_confidence"] == settings.CONF_MIN and not bs[2]["ocr_support"]
    assert sent == [3]  # every unidentified crop; 'XQZT' and 'FROG' contradict the reader's 'Curtain': rejected
    assert [b["legibility"] for b in bs] == ["partial", "partial", "none", "partial", "none"]
    assert out["stats"]["spine_source"] == "sam" and out["stats"]["vision_spines_unmatched"] == 1
    q = review.queue({"scale": None}, bs, [], {}, {}, out["stats"])
    assert {"ref_id": bs[2]["id"], "reason": "title from vision model only, no OCR support"} in q
    assert not any("spines from vision model" in r["reason"] for r in q)


def test_vision_spines_path_unchanged_and_noted(tmp_path, monkeypatch):
    sd = still_sweep(tmp_path)
    stub_catalogue(monkeypatch)
    monkeypatch.setattr(spine_ocr, "read", lambda crops: [{"words": [], "rotation": None} for _ in crops])
    rereads = []
    monkeypatch.setattr(vision, "read_crops", lambda js: rereads.append(len(js)) or [
        {"title": "", "author": "", "publisher": "", "legibility": "none"} for _ in js])
    reads = {"000001.jpg": {"spines": [gem([100, 300, 400, 400], "Die Trying"), gem([100, 100, 400, 200], "Tripwire")],
                            "items": []}}
    out = books.run(sd, reads, NoGeo())
    assert [b["title"] for b in out["books"]] == ["Tripwire", "Die Trying"]  # left to right
    assert all(b["id_source"] == "spine_read" for b in out["books"])
    assert rereads == [2]  # the old pre-OCR reread of single-view claims still runs
    assert out["stats"]["spine_source"] == "vision"
    q = review.queue({"scale": None}, out["books"], [], {}, {}, out["stats"])
    assert any(r["ref_id"] == "books" and r["reason"].startswith("spines from vision model") for r in q)


def test_claim_without_ocr_support_is_kept_but_sent_to_review(tmp_path, monkeypatch):
    """flash-lite read 'Evil Under the Sun' under a price sticker; OCR read nothing there. Not confident."""
    sd = still_sweep(tmp_path)
    stub_catalogue(monkeypatch)
    spines = {"000001.jpg": [sam([100, 100, 400, 200], 0), sam([100, 300, 400, 400], 1)]}
    reads = {"000001.jpg": {"spines": [gem([100, 100, 400, 200], "Evil Under the Sun", "Agatha Christie"),
                                       gem([100, 300, 400, 400], "Tripwire")], "items": []}}
    ocr = [[], [["TRIPWIRE", 0.95, 90]]]
    monkeypatch.setattr(spine_ocr, "read", lambda crops: [{"words": w, "rotation": 90} for w in ocr])
    monkeypatch.setattr(vision, "read_crops", lambda js: [{"title": "", "author": "", "publisher": "",
                                                           "legibility": "none"} for _ in js])
    bs = books.run(sd, reads, NoGeo(), spines)["books"]
    assert [b["title"] for b in bs] == ["Evil Under the Sun", "Tripwire"]
    assert [b["ocr_support"] for b in bs] == [False, True]
    assert bs[0]["id_confidence"] == settings.CONF_MIN and bs[1]["id_confidence"] == 0.95
    q = review.queue({"scale": None}, bs, [], {}, {}, {})
    assert {"ref_id": bs[0]["id"], "reason": "title from vision model only, no OCR support"} in q
    assert not any(r["ref_id"] == bs[1]["id"] and "OCR" in r["reason"] for r in q)


def test_catalogue_timeout_is_no_candidates_not_a_dead_packet(monkeypatch):
    """A replay died here: Google Books rate-limited, then read timeouts outlasted net's retries."""
    import httpx
    from pipeline import net

    def boom(*a, **k):
        raise httpx.ReadTimeout("The read operation timed out")
    monkeypatch.setattr(net, "request", boom)
    assert books._gbooks("Tripwire", "Lee Child") == [] and books._openlibrary("Tripwire", "Lee Child") == []


def test_failed_catalogue_is_not_cached_as_no_such_book(monkeypatch):
    """Google Books' daily quota (429) cached 'Wild Wind' as unknown on photo A; a real empty answer is cached."""
    import httpx
    from pipeline import net
    status = {"code": 429}
    monkeypatch.setattr(net, "request", lambda m, url, **k: httpx.Response(
        status["code"], json={"items": [], "docs": []}, request=httpx.Request(m, url)))
    cache = {}
    assert books.resolve("Wild Wind", "Taslima Nasrin", "", 1.0, cache) == (None, 0.0, False)
    assert isinstance(books._gbooks("Wild Wind", ""), books.Unavailable)
    assert spine_ocr.fragment_lookup(["wild", "wind"], "", cache)[0] is None
    assert cache == {}
    status["code"] = 200
    books.resolve("Wild Wind", "Taslima Nasrin", "", 1.0, cache)
    spine_ocr.fragment_lookup(["wild", "wind"], "", cache)
    assert [v["cands"] for v in cache.values()] == [[], []]


def test_spent_daily_quota_skips_google_books(monkeypatch):
    """After a 'Queries per day' 429, Google Books is not called again in this run; Open Library still is."""
    import httpx
    from pipeline import net
    calls = []

    def req(m, url, **k):
        calls.append(url)
        return httpx.Response(429, text="Quota exceeded for limit 'Queries per day'", request=httpx.Request(m, url))
    monkeypatch.setattr(net, "request", req)
    monkeypatch.setattr(books, "_gbooks_spent", False)
    books._gbooks("Wild Wind", "")
    books._gbooks("Hema Malini", "")
    assert sum("googleapis" in u for u in calls) == 1


def test_checker_failure_is_unsure_not_a_dead_packet(tmp_path, monkeypatch):
    """The checker's daily quota running out mid-sweep sends those books to review instead of raising."""
    (tmp_path / "a.jpg").write_bytes(b"x")
    bs = [{"id": f"b_{i}", "title": t, "author": "", "excluded": False, "crop": "a.jpg"}
          for i, t in enumerate(["Killing Floor", "Die Trying"])]

    def check(claims):
        if any(t == "Die Trying" for _, t, _ in claims):
            raise RuntimeError("429 RESOURCE_EXHAUSTED")
        return [{"verdict": "match", "reason": "ok"} for _ in claims]
    monkeypatch.setattr(vision, "check_identities", check)
    monkeypatch.setattr(review, "CHECK_BATCH", 1)
    review.checker(tmp_path, bs)
    assert bs[0]["checker"]["verdict"] == "match"
    assert bs[1]["checker"] == {"verdict": "unsure", "reason": "checker unavailable (RuntimeError)"}
    assert bs[1]["title"] == "Die Trying"  # unsure keeps the identity; review.queue flags it


def _cl(title, xy, z=0.7, n=3, author="", ori="flat"):
    m = [{"title": title, "author": author, "orientation": ori} for _ in range(n)]
    return {"members": m, "center": np.array([*xy, z + 0.01]), "bottom_z": z, "title": title, "orientation": ori}


def test_merge_joins_one_book_read_differently_across_frames():
    """Live video: 'RAAVAN' and 'RAAVAN: ENEMY OF ARYAVARTA', an author line read as the title, and a flipped
    orientation are one book. 'IKIGAI' beside 'JAPONISME: IKIGAI, ...' and a stacked book 5 cm up are not."""
    out = books.merge([
        _cl("RAAVAN: ENEMY OF ARYAVARTA", (0, 0), n=8),
        _cl("RAAVAN", (0.03, 0), n=2, ori="upright"),
        _cl("EXAM WARRIORS", (1, 0), author="Narendra Modi", ori="upright"),
        _cl("Narendra Modi", (1.01, 0), n=1, ori="upright"),
        _cl("JAPONISME: IKIGAI, FOREST BATHING", (2, 0)),
        _cl("IKIGAI", (2, 0), z=0.73),
        _cl("RAAVAN: ENEMY OF ARYAVARTA", (0, 0), z=0.80),  # same title 10 cm up: another copy in the stack
    ])
    titles = sorted(c["title"] for c in out)
    assert titles == ["EXAM WARRIORS", "IKIGAI", "JAPONISME: IKIGAI, FOREST BATHING", "RAAVAN: ENEMY OF ARYAVARTA",
                      "RAAVAN: ENEMY OF ARYAVARTA"]
    raavan = max((c for c in out if c["title"].startswith("RAAVAN")), key=lambda c: len(c["members"]))
    assert len(raavan["members"]) == 10 and raavan["orientation"] == "flat"
    assert not books.titled("4") and not books.titled("THE") and books.titled("Sita")


def test_seen_once_drops_unread_single_views_only_in_a_long_sweep():
    unread, read = _cl("", (0, 0), n=1), _cl("SITA", (1, 0), n=1)
    assert books.seen_once([unread, read], frames=60) == ([read], [unread])
    assert books.seen_once([unread, read], frames=1) == ([unread, read], [])  # a still photo: every box is one view


def test_neighbour_words_on_an_angled_crop_do_not_veto():
    """The Secret of the Nagas crop also shows Ruskin Bond's spine: those words are the neighbour's, not a veto."""
    cand = {"title": "The Secret of the Nagas", "authors": ["Amish"]}
    ocr = {"words": [["RUSKIN BOND", 0.99], ["ROOM ON THE ROOF", 0.99]], "rotation": 0}
    _, conf, w = spine_ocr.witness(cand, 0.9, ocr, set(), "Amish", {})
    assert w["ocr_veto"]  # no neighbours known: vetoed, as before
    cand2, conf, w = spine_ocr.witness(cand, 0.9, ocr, set(), "Amish", {},
                                        neighbours=["The Room on the Roof Ruskin Bond"])
    assert cand2 is cand and conf == 0.9 and not w["ocr_veto"]


def _bk(title, c, h=20.0, t=3.0, ori="upright", views=5, read=None):
    return {"title": title, "spine_text": {"title": title if read is None else read, "author": "", "publisher": ""},
            "_center": list(c), "spine_height_cm": h, "spine_thickness_cm": t, "orientation": ori, "views": views,
            "excluded": "", "author": "", "ocr_text": ""}


def test_same_book_joins_pieces_of_one_identified_spine():
    """Live sweep 183215: Think Like a Monk came out 6 times (SAM boxed the top of the upright spine on its own, at
    another height). Same book within its height is one; a neighbour, another volume, or a far copy is not."""
    full = _bk("Think Like a Monk", (0.40, 0.30, 0.12), h=24, views=38)
    top = _bk("Think Like a Monk", (0.39, 0.31, 0.21), h=8, views=5, read="")
    piece = _bk("", (0.401, 0.301, 0.15), h=4, read="")
    neighbour = _bk("", (0.40, 0.33, 0.12), h=22, read="")  # one spine thickness along the shelf
    v3 = _bk("Volume", (0.40, 0.36, 0.12), read="Volume 3")
    v7 = _bk("Volume", (0.40, 0.39, 0.12), read="Volume 7")
    far = _bk("Think Like a Monk", (1.40, 0.30, 0.12))
    named = {**_bk("", (0.45, 0.30, 0.10), h=6, read=""), "ocr_text": "LIKE MONK"}  # its OCR names the book
    stub = _bk("", (0.80, 0.30, 0.10), h=4, read="")  # a 4 cm unread box: a piece of some spine, not a book
    out, folded, fragments = books.same_book([full, top, piece, neighbour, v3, v7, far, named, stub])
    assert out == [full, neighbour, v3, v7, far] and folded == 3 and fragments == 1
    assert full["views"] == 38 + 5 + 5 + 5


def test_same_book_folds_a_half_read_resolved_to_another_work():
    """Demo sweep 5ced: a half-spine box read 'THE OAT THE VAYUI' resolved to 'The Vayu Purana' and became a 13th
    book next to The Oath of the Vayuputras. A read that fits the neighbour's title better is part of it; a nearby book
    whose read fits its own title stays."""
    oath = _bk("The Oath of the Vayuputras", (0.13, 0.40, 0.57), h=18, ori="flat", views=53)
    half = _bk("The Vayu Purana", (0.15, 0.45, 0.63), h=10, ori="flat", views=4, read="THE OAT THE VAYUI")
    sita = _bk("Sita", (0.14, 0.42, 0.60), ori="flat", read="SITA WARRIOR OF MITHILA")
    out, folded, _ = books.same_book([oath, half, sita])
    assert out == [oath, sita] and folded == 1 and oath["views"] == 57
    # shelf sweep 203017: Exam Warriors, named by its Marathi title, is not The Ship of Secrets beside it
    ship = _bk("The Ship of Secrets", (0.5, 0.3, 0.12), views=41)
    exam = _bk("एक्झाम वॉरियर्स", (0.5, 0.33, 0.12), read="EXAM WARRIORS")
    assert books.same_book([ship, exam])[1] == 0


def test_same_book_flat_pieces_and_crop_reader_alias():
    """Demo sweep 5ced: a second box on the flat Othello, 1.3 cm along the spine and within its thickness, stayed a
    line, unread or named by the crop reader 'William Shakespeare's Othello'. The book above it in the stack stays."""
    oth = _bk("Othello", (0.156, 0.633, 0.567), h=15, t=2.5, ori="flat", views=31)
    unread = _bk("", (0.144, 0.628, 0.576), h=12, t=2.4, ori="flat", views=8, read="")
    above = _bk("", (0.156, 0.633, 0.592), h=15, t=2.5, ori="flat", read="")  # one thickness up the stack
    out, folded, _ = books.same_book([oth, unread, above])
    assert out == [oth, above] and folded == 1
    named = _bk("William Shakespeare's Othello", (0.144, 0.628, 0.576), h=12, ori="flat", views=8, read="")
    out, folded, _ = books.same_book([_bk("Othello", (0.156, 0.633, 0.567), ori="flat", views=31), named])
    assert len(out) == 1 and folded == 1
    # a garbled read no catalogue matched names one neighbour: part of it; an author shared by two neighbours is not
    gates = {**_bk("Tom Gates: Family, Friends and Furry Creatures", (0.133, 0.397, 0.606), ori="flat", views=55),
             "author": "Liz Pichon"}
    garbled = _bk("", (0.144, 0.462, 0.608), h=7, ori="flat", read="TOM GATES FAMI FURR")
    out, folded, _ = books.same_book([gates, garbled])
    assert out == [gates] and folded == 1
    sita, raavan = ({**_bk(t, c, ori="flat"), "author": "Amish Tripathi"} for t, c in
                    (("Sita", (0.1, 0.4, 0.50)), ("Raavan", (0.1, 0.4, 0.53))))
    amish = _bk("", (0.12, 0.45, 0.515), ori="flat", read="AMISH")
    assert books.same_book([sita, raavan, amish])[1] == 0


def test_pick_prefers_the_closest_edition_title_among_near_ties():
    """Demo sweep 5ced: an English spine got the Hindi edition's catalogue title at P 0.9805 vs 0.958 for the English
    one. Near-ties go to the title closest to the read; a clearly better probability still wins."""
    cands = [{"title": "Raavan", "subtitle": "Enemy of Aryavarta", "authors": ["Amish"]},
             {"title": "Raavan - Aryavart Ka Shatru (Raavan", "subtitle": "Enemy of Aryavarta)", "authors": ["Amish Tripathi"]}]
    cand, conf, _ = books.pick("RAAVAN ENEMY OF ARYAVARTA", "AMISH", {"cands": cands, "probs": {"c0": 0.958, "c1": 0.9805}})
    assert cand is cands[0] and conf == 0.958
    cand, _, _ = books.pick("RAAVAN ENEMY OF ARYAVARTA", "AMISH", {"cands": cands, "probs": {"c0": 0.6, "c1": 0.98}})
    assert cand is cands[1]


def test_frame_cap_drops_only_unread_lines_above_one_full_view():
    """Gemini saw 6 spines in a frame showing the whole unit; 3D held 4 identified + 1 weak title + 3 unread. Two
    unread lines go (weakest candidate, then fewest views); the weak title stays. Without a full view nothing is dropped."""
    def line(i, title, conf, views=5):
        return {"id": f"b{i}", "shelf": "A-1", "excluded": "", "title": title, "id_confidence": conf, "views": views}
    known = [line(i, f"Book {i}", 0.99) for i in range(4)]
    weak = line(4, "Maybe", 0.5)
    unread = [line(5, "", 0.0, 9), line(6, "", 0.0, 2), line(7, "", 0.1, 3)]
    clusters = [{"id": b["id"], "members": [{"frame": "f1"}]} for b in known + [weak] + unread]
    reads = {"f1": {"spines": [{}] * 6}, "f2": {"spines": [{}] * 2}}
    kept, n = books.frame_cap(known + [weak] + unread, clusters, reads)
    assert n == 2 and [b["id"] for b in kept] == ["b0", "b1", "b2", "b3", "b4", "b7"]  # b7 has a weak candidate
    clusters = [{"id": b["id"], "members": [{"frame": f"f{i}"}]} for i, b in enumerate(known + [weak] + unread)]
    assert books.frame_cap(known + [weak] + unread, clusters, reads)[1] == 0  # no frame shows the unit


def test_reconcile_guardrails():
    """The reconcile model only groups or drops existing lines: two confident titles stay two books, a confident
    line is never dropped, 'unsure' keeps the line, and an answer that does not list every entry once is ignored."""
    from pipeline import reconcile
    def line(i, title, conf, views=5):
        return {"id": f"b{i}", "title": title, "id_confidence": conf, "views": views, "excluded": ""}
    unit = [line(0, "Think Like a Monk", 0.99, 30), line(1, "", 0.0, 4), line(2, "Ikigai", 0.99), line(3, "", 0.0),
            line(4, "Sita", 0.98), line(5, "", 0.0)]
    A = reconcile.Answer
    ans = A(books=[reconcile.Book(entries=[0, 1, 2], reason="top half of Monk; Ikigai beside it"),
                   reconcile.Book(entries=[4], reason="")],
            not_books=[reconcile.Drop(entry=3, kind="not_a_book", reason="a box"),
                       reconcile.Drop(entry=5, kind="unsure", reason="?")])
    kept, acts = reconcile.apply(unit, ans)
    assert [b["id"] for b in kept] == ["b0", "b2", "b4", "b5"]
    assert unit[0]["views"] == 34 and [a[:3] for a in acts] == [("b1", "merged", "b0"), ("b3", "not_a_book", "")]
    strong_drop = A(books=[reconcile.Book(entries=[0, 1, 2, 3, 5], reason="")],
                    not_books=[reconcile.Drop(entry=4, kind="not_a_book", reason="")])
    assert "b4" in [b["id"] for b in reconcile.apply([dict(b) for b in unit], strong_drop)[0]]
    assert reconcile.apply(unit, A(books=[reconcile.Book(entries=[0, 1], reason="")], not_books=[])) is None
