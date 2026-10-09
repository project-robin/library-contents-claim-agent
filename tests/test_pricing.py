import numpy as np
import pytest

from pipeline import drex, pricing
from pipeline.books import cluster, score, sim, vote


@pytest.fixture
def fake_drex(monkeypatch):
    """Stand-in for DREX: yes for listings whose title starts with the book title and is not a different volume."""
    def ask(state, questions):
        subject = state.get("item_being_priced", {}).get("title", "").lower()
        listings = state.get("listings", {})
        return {k: (0.97 if listings.get(k, "").lower().startswith(subject) and "volume" not in listings.get(k, "").lower()
                    else 0.02) for k in questions}
    monkeypatch.setattr(drex, "ask", ask)


def test_parse_price_locale():
    assert pricing.parse_price("₹1,29,999.00", ("₹", "Rs", "INR")) == 129999.0
    assert pricing.parse_price("Rs. 450", ("₹", "Rs", "INR")) == 450.0
    assert pricing.parse_price("$12.99", ("₹", "Rs", "INR")) is None
    assert pricing.parse_price("$12.99", ("$", "USD")) == 12.99
    assert pricing.parse_price("CA$12.99", ("$", "USD")) is None


def test_prefilter_is_whole_word():
    assert pricing.not_the_book("Sapiens Kindle Edition")
    assert pricing.not_the_book("Harari Box Set of 3 books")
    assert not pricing.not_the_book("The Smuggler's Moon")  # 'mug' inside a word is not a mug


def test_listing_match_uses_drex_probability(tmp_path, fake_drex):
    p = pricing.Pricer(tmp_path / "c.json")
    offers = [{"title": "Sapiens: A Brief History", "amount": 1}, {"title": "Sapiens: A Graphic History, Volume 1", "amount": 2}]
    kept = p.matching({"title": "Sapiens", "author": "Harari"}, offers, pricing.BOOK_NEW_Q)
    assert [o["amount"] for o in kept] == [1] and kept[0]["match_p"] == 0.97


def test_new_price_is_a_real_offer_and_never_invented(tmp_path, monkeypatch, fake_drex):
    p = pricing.Pricer(tmp_path / "c.json")
    fake = {"retrieved_at": "2026-10-06T10:00:00+00:00", "results": [
        {"title": "Sapiens Paperback", "source": "Amazon.in", "link": "https://a/1", "price": "₹499"},
        {"title": "Sapiens (Hardcover)", "source": "Flipkart", "link": "https://f/2", "price": "₹899"},
        {"title": "Sapiens Kindle", "source": "Amazon.in", "link": "https://a/3", "price": "₹199"},
        {"title": "Sapiens", "source": "Shop", "link": "https://s/4", "price": "₹650"}]}
    monkeypatch.setattr(p, "shopping", lambda q, gl: fake if gl == "in" else {"retrieved_at": "", "results": []})
    new = p.book_new("Sapiens", "Yuval Noah Harari", "IN")
    assert new["amount"] == 650 and new["url"] == "https://s/4" and not new["converted"]
    assert new["retrieved_at"] == fake["retrieved_at"]
    monkeypatch.setattr(p, "shopping", lambda q, gl: {"retrieved_at": "", "results": []})
    assert p.book_new("Sapiens", "Harari", "IN") is None  # nothing found -> nothing made up


def test_converted_price_is_labelled(tmp_path, monkeypatch, fake_drex):
    p = pricing.Pricer(tmp_path / "c.json")
    us = {"retrieved_at": "t", "results": [{"title": "Dune", "source": "B&N", "link": "u", "price": "$10.00"}]}
    monkeypatch.setattr(p, "shopping", lambda q, gl: us if gl == "us" else {"retrieved_at": "", "results": []})
    monkeypatch.setattr(p, "fx", lambda b, t: {"rate": 84.0, "date": "2026-10-05", "source": "ECB", "url": "x"})
    new = p.book_new("Dune", "Herbert", "IN")
    assert new["converted"] and new["amount"] == 840.0 and new["converted_from"]["currency"] == "USD"


def test_identification_score_and_vote():
    cand = {"title": "Sapiens", "subtitle": "A Brief History of Humankind", "authors": ["Yuval Noah Harari"]}
    assert score("SAPIENS", "Harari", 1.0, cand) > 0.9  # surname-only spine
    assert score("Dune", "", 1.0, cand) < 0.5
    t, a, _, agree = vote([("Sapiens", "Harari", "", 2), ("Sapiens", "", "", 1), ("Sapens", "", "", 1)])
    assert t == "Sapiens" and agree == 1.0
    assert sim("Volume 2", "Volume 3") <= 0.5  # series volumes never merge


def test_dedup_merges_views_keeps_neighbours():
    def det(x, title, leg="full"):
        return {"center": np.array([x, 0.0, 1.0]), "orientation": "upright", "bottom_z": 0.9, "title": title,
                "thickness_cm": 3.0, "legibility": leg, "area": 100, "t": 0}
    dets = [det(0.000, "Dune"), det(0.005, "Dune", "partial"), det(0.030, "Emma"), det(0.034, ""),
            det(0.060, ""), det(0.12, "Dune")]
    cs = cluster(dets)
    assert sorted(len(c["members"]) for c in cs) == [1, 2, 3]  # Dune x3 (same title re-sighted 12 cm off), Emma x2, blank


def test_used_and_import_sellers_are_not_new_prices():
    from pipeline.pricing import used_listing
    assert used_listing("Biblio.com - Orphans Treasure Box", "Cross Country - Hardcover")
    assert used_listing("Bookchor", "Cross Country By James Patterson")
    assert used_listing("Some Shop", "Cross Country (Used - Good)")
    assert not used_listing("amazon.in", "Cross Country")
    assert not used_listing("Crossword Bookstores", "The Usual Suspects")  # 'used' only as a whole word


def test_price_lookup_failure_leaves_line_unpriced(tmp_path, monkeypatch, fake_drex):
    """Serper 504s that outlast the retries: the book is unpriced and goes to review; the packet still builds."""
    import httpx

    def down(self, q, gl):
        raise httpx.HTTPStatusError("504", request=httpx.Request("POST", "https://google.serper.dev/shopping"),
                                    response=httpx.Response(504))
    monkeypatch.setattr(pricing.Pricer, "shopping", down)
    monkeypatch.setattr(pricing.Pricer, "ebay_used", lambda self, q, m: {"retrieved_at": "", "results": []})
    b = {"id": "b_0", "title": "Killing Floor", "author": "Lee Child", "excluded": False, "marks": [], "notes": []}
    out = pricing.run(tmp_path, [b], [], "IN")
    p = out["books"]["b_0"]
    assert p["status"] == "identified" and "replacement_cost" not in p
    assert p["reasons"] == ["price lookup failed (HTTPStatusError): not priced"]


def test_used_price_from_second_hand_sellers_on_the_same_queries(tmp_path, monkeypatch, fake_drex):
    """India has no used-book API: second-hand sellers in the same Shopping results give the used price, sourced and
    dated. A used offer above the new price (a collector listing) is not the used value."""
    p = pricing.Pricer(tmp_path / "c.json")
    fake = {"retrieved_at": "2026-10-09T10:00:00+00:00", "results": [
        {"title": "Ikigai", "source": "Amazon.in", "link": "https://a/1", "price": "₹299"},
        {"title": "Ikigai", "source": "Kitabay", "link": "https://k/3", "price": "₹170"},
        {"title": "Ikigai (Used - Good)", "source": "Some Shop", "link": "https://s/4", "price": "₹190"},
        {"title": "Ikigai", "source": "AbeBooks", "link": "https://ab/5", "price": "₹2,400"}]}
    calls = []
    monkeypatch.setattr(p, "shopping", lambda q, gl: calls.append(q) or fake)
    new = p.book_new("Ikigai", "Hector Garcia", "IN")
    used = p.book_used("Ikigai", "Hector Garcia", "IN", new)
    assert new["amount"] == 299  # second-hand offers never set the replacement cost
    # lower median of 170, 190 (2,400 dropped); with the collector listing it would be 190
    assert used["amount"] == 170 and used["url"] == "https://k/3" and "Kitabay" in used["source"]
    assert used["retrieved_at"] == fake["retrieved_at"] and used["condition_assumed"].startswith("used")
    assert used["listing_title"] == "Ikigai" and not used["converted"]


def test_listing_title_must_be_this_book_in_this_language():
    f = pricing.same_book_title
    assert not f("The Oath of the Vayuputras", "Flyte By Mark Zug")
    assert not f("Think Like a Monk", "Think Like a Monk (Kannada)")
    assert not f("Raavan", "Raavan: Enemy of Aryavarta (Hindi Edition)")
    assert not f("Ikigai", "Ikigai (इकिगाई)")
    assert not f("Will Grayson, Will Grayson", "Libro Will Grayson Will Grayson (biblioteca John Green)", "John Green")
    assert not f("Will Grayson, Will Grayson", "The Essential John Green Collection 4 Books Set", "John Green")
    assert not f("Tom Gates: Family, Friends and Furry Creatures", "Tom Gates #24: Pesky Pets And Parties", "Liz Pichon")
    assert f("Ikigai", "Ikigai: The Japanese Secret to a Long and Happy Life")  # a subtitle word, not a language
    assert f("Othello - William Shakespeare", "William Shakespeare : Othello The Moor Of Venice")
    assert f("Japonisme: Ikigai, Forest Bathing, Wabi-sabi and More", "Japonisme | Erin Niimi Longhurst",
             "Erin Niimi Longhurst")  # the listing drops the subtitle
