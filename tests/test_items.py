"""Non-book items: the printed scale marker is not an item, and the item check may only merge or drop lines."""
import cv2
import numpy as np

from pipeline import geometry, items, reconcile


def test_marker_card_is_not_an_item(tmp_path):
    """Sweep 230330 listed the marker twice as art. A box mostly showing the marker is dropped; a big box with the
    card leaning in a corner (a shelf) is kept."""
    (tmp_path / "frames").mkdir()
    img = np.full((1080, 1920), 255, np.uint8)
    img[300:500, 300:500] = cv2.aruco.generateImageMarker(geometry.ARUCO, 0, 200)
    cv2.imwrite(str(tmp_path / "frames" / "f.jpg"), img)
    det = cv2.aruco.ArucoDetector(geometry.ARUCO, cv2.aruco.DetectorParameters())
    to_box = lambda x0, y0, x1, y1: [y0 / 1080 * 1000, x0 / 1920 * 1000, y1 / 1080 * 1000, x1 / 1920 * 1000]
    cache = {}
    assert items.shows_marker(tmp_path, "f.jpg", to_box(250, 250, 600, 650), cache, det)  # the clipboard
    assert not items.shows_marker(tmp_path, "f.jpg", to_box(200, 100, 1900, 1080), cache, det)  # a whole shelf
    assert not items.shows_marker(tmp_path, "f.jpg", to_box(1000, 300, 1400, 700), cache, det)  # elsewhere


def _it(i, cat, desc, views=1, **kw):
    return {"id": f"i_{i:03d}", "category": cat, "description": desc, "views": views, "confidence": 0.8,
            "is_print": None, "notes": [], **kw}


def test_item_check_merges_one_bed_and_drops_fixtures_but_keeps_what_the_user_answered():
    fresh = lambda: [_it(1, "furniture", "Bed with patterned mattress", 3), _it(2, "furniture", "Headboard with print"),
                     _it(3, "electronics", "White switch board"), _it(4, "art", "Spider-Man painting", is_print=True),
                     _it(5, "art", "Framed artwork, black and white geometric pattern")]
    lines = fresh()
    ans = reconcile.ItemAnswer(
        objects=[reconcile.Thing(entries=[0, 1], reason="headboard of the bed"), reconcile.Thing(entries=[4], reason="")],
        not_items=[reconcile.NotItem(entry=2, kind="building_fixture", reason="switch"),
                   reconcile.NotItem(entry=3, kind="building_fixture", reason="wrongly called a fixture"),
                   reconcile.NotItem(entry=4, kind="scale_marker", reason="the marker card")])
    kept, _ = reconcile.apply_items(lines, ans)  # entry 4 is both an object and a marker: its first place counts
    assert "i_005" in [i["id"] for i in kept]
    ans.objects[1].entries = []
    kept, acts = reconcile.apply_items(fresh(), ans)
    assert [i["id"] for i in kept] == ["i_001", "i_004"]  # the print the user confirmed is never dropped
    assert kept[0]["views"] == 4
    assert {(a[0], a[1]) for a in acts} == {("i_002", "merged"), ("i_003", "building_fixture"), ("i_005", "scale_marker")}


def test_item_check_keeps_lines_the_answer_left_out_and_ignores_an_answer_that_did_not_try():
    lines = [_it(i, "decor", f"thing {i}") for i in range(1, 7)]
    ans = reconcile.ItemAnswer(objects=[reconcile.Thing(entries=[0, 1], reason="same"),
                                        reconcile.Thing(entries=[2, 9], reason="9 does not exist")], not_items=[])
    kept, acts = reconcile.apply_items(lines, ans)
    assert [i["id"] for i in kept] == ["i_001", "i_003", "i_004", "i_005", "i_006"] and len(acts) == 1
    assert reconcile.apply_items(lines, reconcile.ItemAnswer(objects=[reconcile.Thing(entries=[0], reason="")],
                                                             not_items=[])) is None


def test_bedding_is_never_folded_into_the_bed_and_the_bed_is_the_line_kept():
    """Run 2 of 3 on sweep 230330: bed, bedsheet and pillows came back as one object and the bedsheet (8 views) was
    kept while the bed vanished."""
    lines = [_it(1, "textile", "Floral pattern bedsheet", 8), _it(2, "furniture", "Bed with patterned mattress", 3),
             _it(3, "furniture", "Bed seen from the side", 2), _it(4, "decor", "Bolster pillow")]
    ans = reconcile.ItemAnswer(objects=[reconcile.Thing(entries=[0, 1, 2, 3], reason="the bed with its bedding")],
                               not_items=[])
    kept, acts = reconcile.apply_items(lines, ans)
    assert [i["id"] for i in kept] == ["i_001", "i_002", "i_004"]
    assert acts == [("i_003", "merged", "i_002", "the bed with its bedding")]
