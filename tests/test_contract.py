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
