"""OCR witness decisions, on words PP-OCRv5 actually read from a sale shelf (Lee Child / Agatha Christie, stickers)."""
from pipeline import books, drex, spine_ocr


def ocr(*words):
    return {"words": [[w, 0.97, 270] for w in words], "rotation": 270}


SHELF = [ocr("LEE", "CHILD", "TO", "ROW", "JACK", "REACHER"), ocr("LEE", "CHILD", "TRIPWIRE", "JACK REACHER"),
         ocr("LEE", "CHILD", "NO PLANB", "A JACK REACHER THRILLER"), ocr("JACK REACHER", "LEE CHILD", "MAK"),
         ocr("agalla Chuistie", "EVA"), ocr("Agalla Chuistie", "Nemes"), ocr("agallaChustie", "HERCULE POIRO", "CHRISTM"),
         ocr("Agalla Chuistie", "SADCY", "ESS")]


def test_shelf_boilerplate_is_learned_not_listed():
    boiler = spine_ocr.boilerplate(SHELF)
    assert {"lee", "child", "agalla", "chuistie"} <= boiler and "tripwire" not in boiler


def test_sticker_covered_title_is_vetoed_but_the_real_one_survives():
    boiler = spine_ocr.boilerplate(SHELF)
    toks = spine_ocr.distinct(SHELF[0], boiler, "Lee Child")
    assert toks == ["row"]
    assert spine_ocr.veto("Die Trying", toks)            # flash-lite's invention for 'TO… ROW'
    assert not spine_ocr.veto("Gone Tomorrow", toks)     # the book actually on the shelf
    eva = spine_ocr.distinct(SHELF[4], boiler, "Agatha Christie")
    assert spine_ocr.veto("Evil Under the Sun", eva) and not spine_ocr.veto("Why Didn't They Ask Evans?", eva)


def test_garbled_author_never_vetoes_a_correct_title():
    boiler = spine_ocr.boilerplate(SHELF)
    assert not spine_ocr.veto("Nemesis", spine_ocr.distinct(SHELF[5], boiler, "Agatha Christie"))
    assert not spine_ocr.veto("Nemesis", spine_ocr.distinct(SHELF[5], set(), ""))  # even without boilerplate
    assert not spine_ocr.veto("Tripwire", [])                                       # no text, no verdict


def test_too_little_text_is_never_looked_up():
    toks = spine_ocr.distinct(SHELF[4], spine_ocr.boilerplate(SHELF), "Agatha Christie")
    assert spine_ocr.fragment_lookup(toks, "Agatha Christie", {}) == (None, 0.0, [])


def test_fragments_pick_the_book_that_contains_them(monkeypatch):
    def cand(title):
        return {"title": title, "subtitle": "", "authors": ["Agatha Christie"], "publisher": "", "year": "", "isbn": "",
                "url": title, "source": "stub"}
    monkeypatch.setattr(books, "_gbooks", lambda t, a, free=False: [cand("Hercule Poirot's Christmas"),
                                                                    cand("The Mysterious Affair at Styles")])
    monkeypatch.setattr(books, "_openlibrary", lambda t, a: [cand("Hercule Poirot's Casebook")])
    asked = {}
    monkeypatch.setattr(drex, "ask", lambda state, qs: asked.update(state) or {"c0": 0.93, "c1": 0.2})
    toks = spine_ocr.distinct(SHELF[6], spine_ocr.boilerplate(SHELF), "Agatha Christie")
    c, p, used = spine_ocr.fragment_lookup(toks, "Agatha Christie", {})
    assert c["title"] == "Hercule Poirot's Christmas" and p == 0.93 and used == ["hercule", "poiro", "christm"]
    titles = [x["title"] for x in asked["catalogue_candidates"].values()]
    assert "The Mysterious Affair at Styles" not in titles  # contains none of the fragments: never shown to DREX


def test_editions_of_one_work_are_not_rivals(monkeypatch):
    def cand(title):
        return {"title": title, "subtitle": "", "authors": ["Agatha Christie"], "publisher": "", "year": "", "isbn": "",
                "url": title, "source": "stub"}
    monkeypatch.setattr(books, "_gbooks", lambda t, a, free=False: [cand("Curtain: Poirot's Last Case"),
                                                                    cand("Curtain: Poirot’s Last Case (Poirot)")])
    monkeypatch.setattr(books, "_openlibrary", lambda t, a: [])
    monkeypatch.setattr(drex, "ask", lambda state, qs: {"c0": 0.99, "c1": 0.98})
    c, p, _ = spine_ocr.fragment_lookup(["last", "case", "curtain", "poirots"], "", {})
    assert c["title"] == "Curtain: Poirot's Last Case" and p == 0.99  # two editions, one work: a clear winner
    assert not spine_ocr.same_work("Curtain", "Curtain Call")


def test_one_long_fragment_can_name_a_book(monkeypatch):
    def cand(title):
        return {"title": title, "subtitle": "", "authors": ["Lee Child"], "publisher": "", "year": "", "isbn": "",
                "url": title, "source": "stub"}
    monkeypatch.setattr(books, "_gbooks", lambda t, a, free=False: [cand("Tripwire"), cand("Die Trying")])
    monkeypatch.setattr(books, "_openlibrary", lambda t, a: [])
    asked = {}
    monkeypatch.setattr(drex, "ask", lambda state, qs: asked.update(state) or {"c0": 0.9})
    c, p, _ = spine_ocr.fragment_lookup(["tripwire"], "", {})
    assert c["title"] == "Tripwire" and len(asked["catalogue_candidates"]) == 1  # 'Die Trying' lacks the word
