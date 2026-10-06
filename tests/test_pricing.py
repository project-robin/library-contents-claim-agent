from pipeline import pricing
from pipeline.books import cluster, score, vote
import numpy as np


def test_parse_price_locale():
    assert pricing.parse_price("₹1,29,999.00", ("₹", "Rs", "INR")) == 129999.0
    assert pricing.parse_price("Rs. 450", ("₹", "Rs", "INR")) == 450.0
    assert pricing.parse_price("$12.99", ("₹", "Rs", "INR")) is None
    assert pricing.parse_price("$12.99", ("$", "USD")) == 12.99
    assert pricing.parse_price("CA$12.99", ("$", "USD")) is None


def test_title_match_rejects_ebooks_and_sets():
    assert pricing.title_matches("Sapiens", "Sapiens: A Brief History of Humankind (Paperback)")
    assert not pricing.title_matches("Sapiens", "Sapiens Kindle Edition")
    assert not pricing.title_matches("Sapiens", "Harari Box Set of 3 books")
    assert not pricing.title_matches("The Hobbit", "The Lord of the Rings")


def test_new_price_is_a_real_offer_and_never_invented(tmp_path, monkeypatch):
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


def test_converted_price_is_labelled(tmp_path, monkeypatch):
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


def test_dedup_merges_views_keeps_neighbours():
    def det(x, title, leg="full"):
        return {"center": np.array([x, 0.0, 1.0]), "orientation": "upright", "bottom_z": 0.9, "title": title,
                "thickness_cm": 3.0, "legibility": leg, "area": 100, "t": 0}
    dets = [det(0.000, "Dune"), det(0.005, "Dune", "partial"), det(0.030, "Emma"), det(0.034, ""),
            det(0.060, ""), det(0.12, "Dune")]
    cs = cluster(dets)
    assert sorted(len(c["members"]) for c in cs) == [1, 2, 3]  # Dune x3 (same title re-sighted 12 cm off), Emma x2, blank
