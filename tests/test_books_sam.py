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
