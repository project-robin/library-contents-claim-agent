from pipeline.packet import totals


def book(status, rep=None, used=None, title="x"):
    return {"status": status, "title": title, "replacement_cost": {"amount": rep}, "used_value": {"amount": used}}


def item(status, lo=None, hi=None):
    return {"status": status, "replacement_cost": {"low": lo, "high": hi}}


def test_totals_add_up_from_lines_only():
    books = [book("identified", 499.10, 200.05), book("identified", 0.2, None), book("identified", None, 50),
             book("needs_appraisal", None, None), book("unidentified", None, None, title="")]
    items = [item("priced", 1000, 1000), item("range", 200.5, 400.25), item("needs_appraisal"), item("range")]
    t = totals(books, items, 2.5)
    assert t["books_replacement_cost"] == 499.30  # Decimal: no 499.29999
    assert t["books_used_value"] == 250.05
    assert t["items_replacement_cost_low"] == 1200.5 and t["items_replacement_cost_high"] == 1400.25
    assert t["book_count"] == 5 and t["books_identified"] == 4 and t["books_unidentified"] == 1
    assert t["excluded_from_totals"] == 3 + 2  # 3 books without replacement in total, 2 items unpriced
    assert t["books_needs_appraisal"] == 1 and t["items_needs_appraisal"] == 1


def test_totals_are_empty_not_zero_when_nothing_was_priced():
    t = totals([book("identified", 300, None), book("unidentified", title="")], [item("needs_appraisal")], 1.0)
    assert t["books_replacement_cost"] == 300 and t["books_used_value"] is None
    assert t["items_replacement_cost_low"] is None and t["items_replacement_cost_high"] is None
