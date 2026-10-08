import pytest
from pydantic import ValidationError

from pipeline.contract import Book

BOOK = {"id": "b_001", "shelf": "A-1", "position": 1, "frame_ref": "frames/000001.jpg#t=1.0", "status": "identified",
        "title": "Dune", "author": "Frank Herbert", "edition": "", "isbn": "", "spine_height_cm": 21.0,
        "spine_thickness_cm": 3.1, "id_confidence": 0.93,
        "replacement_cost": {"amount": 499.0, "source": "Google Shopping via Serper: Amazon.in", "url": "https://x",
                             "retrieved_at": "2026-10-06T10:00:00+00:00", "converted": False},
        "used_value": {"amount": None, "source": "", "url": "", "retrieved_at": "", "condition_assumed": ""}}


def test_valid_book_passes():
    Book.model_validate(BOOK)


@pytest.mark.parametrize("change", [
    {"replacement_cost": {**BOOK["replacement_cost"], "url": ""}},           # price without a source
    {"status": "needs_appraisal"},                                         # appraisal item carrying a price
    {"status": "unidentified"},                                            # unidentified but titled
    {"frame_ref": ""},                                                     # no evidence frame
])
def test_contract_rejects_unsupported_claims(change):
    with pytest.raises(ValidationError):
        Book.model_validate(BOOK | change)


def test_dropped_field_fails():
    with pytest.raises(ValidationError):
        Book.model_validate({k: v for k, v in BOOK.items() if k != "isbn"})


def test_malformed_boxes_are_dropped_not_fatal():
    from pipeline.vision import clean_read
    sp = {"orientation": "upright", "legibility": "full", "title": "t", "author": "", "publisher": "", "marks": []}
    r = clean_read({"spines": [{**sp, "box_2d": [10, 20, 30, 40]}, {**sp, "box_2d": [47, 281, 71, 155, 308]},
                               {**sp, "box_2d": [30, 20, 10, 40]}], "items": [], "issue": "none"})
    assert [s["box_2d"] for s in r["spines"]] == [[10, 20, 30, 40]] and r["dropped_bad_boxes"] == 2


def test_author_name_alone_is_not_an_identification():
    from pipeline.books import author_only
    cands = [{"title": "Sidney Sheldon", "authors": ["Anon"]}, {"title": "The Other Side of Midnight", "authors": ["Sidney Sheldon"]}]
    assert author_only("SIDNEY SHELDON", "", cands)
    assert author_only("SHELDON", "", cands)
    assert not author_only("The Other Side of Midnight", "Sidney Sheldon", cands)
    assert not author_only("Midnight", "", cands)


def test_title_hidden_by_a_sticker_is_not_completed_into_another_book():
    from pipeline.vision import clean_read, title_supported
    assert not title_supported("Die Trying", "A JACK REACHER THRILLER LEE CHILD TO… ROW")
    assert not title_supported("Evil Under the Sun", "Agatha Christie EVA…")
    assert not title_supported("Death Comes as the End", "Agatha Christie DEATH O… …LE HARPER")
    assert title_supported("Never Go Back", "LEE CHILD JACK REACHER NEVER GO BACK")
    assert title_supported("Dream With Your Eyes Open", "DREAM WITH YOUR EYES OPEN RONNIE SCREWWALLA")
    assert title_supported("Curtain: Poirot's Last Case", "Agatha Christie CURTAIN: POIROT'S LAST CASE")
    sp = {"box_2d": [10, 20, 30, 40], "orientation": "upright", "legibility": "full", "author": "Lee Child",
          "publisher": "", "marks": [], "visible_text": "LEE CHILD TO… ROW", "title": "Die Trying"}
    s = clean_read({"spines": [sp], "items": [], "issue": "none"})["spines"][0]
    assert s["title"] == "" and s["title_unsupported"] == "Die Trying" and s["legibility"] == "partial"
    old = {k: v for k, v in sp.items() if k != "visible_text"}  # reads saved before the field existed: untouched
    assert clean_read({"spines": [old], "items": [], "issue": "none"})["spines"][0]["title"] == "Die Trying"


def test_primary_model_failure_trips_breaker_to_fallback(monkeypatch):
    import httpx
    from types import SimpleNamespace
    from pipeline import settings, vision
    calls = []

    def generate_content(model, contents, config):
        calls.append(model)
        if model == settings.VISION_MODEL:
            raise httpx.ReadTimeout("slow dense frame")
        usage = SimpleNamespace(prompt_token_count=1, candidates_token_count=1, thoughts_token_count=0)
        return SimpleNamespace(parsed="ok", usage_metadata=usage)

    monkeypatch.setattr(vision, "_primary_down_until", 0.0)
    monkeypatch.setattr(vision, "client", lambda: SimpleNamespace(models=SimpleNamespace(generate_content=generate_content)))
    assert vision._generate(settings.VISION_MODEL, [], None, "p") == "ok"
    assert vision._generate(settings.VISION_MODEL, [], None, "p") == "ok"
    assert calls == [settings.VISION_MODEL, settings.VISION_FALLBACK_MODEL, settings.VISION_FALLBACK_MODEL]
