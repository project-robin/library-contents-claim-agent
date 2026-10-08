import json
from pathlib import Path

import httpx
import pytest

from pipeline import sam, settings

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "sam_photoB.json").read_text())
TEXT = FIXTURE["output"][0]["content"][0]["text"]


def test_parse_recorded_photo():
    boxes = sam.parse(TEXT)
    assert len(boxes) == 40  # hand count on photo B
    assert len({b["id"] for b in boxes}) == 40
    for b in boxes:
        x1, y1, x2, y2 = b["box"]
        assert 0 <= x1 < x2 <= b["w"] and 0 <= y1 < y2 <= b["h"]
        assert (b["w"], b["h"]) == (1080, 810)
        assert "mask" not in b


def test_parse_masks_kept_raw():
    b = sam.parse(TEXT, masks=True)[0]
    assert b["mask"]["data"].startswith("341,74,")  # H x W of the box, x2/y2 inclusive
    assert b["box"][3] - b["box"][1] + 1 == 341


def test_parse_empty_and_malformed():
    assert sam.parse("") == [] and sam.parse(None) == [] and sam.parse("<0f>\n") == []
    good = "0<|box;x1=10;y1=20;x2=30;y2=200;w=100;h=300|>"
    bad = ["1<|box;x1=10;y1=20;x2=30|>",  # truncated
           "2<|box;x1=50;y1=20;x2=40;y2=200;w=100;h=300|>",  # x2 < x1
           "3<|box;x1=10;y1=20;x2=130;y2=200;w=100;h=300|>"]  # outside the image
    assert sam.parse("<0f>" + ",".join([bad[0], good] + bad[1:])) == [
        {"id": 0, "box": [10, 20, 30, 200], "w": 100, "h": 300}]


def _fake_post(responses, calls):
    def post(url, json=None, headers=None, timeout=None):
        calls.append((url, json, headers))
        status, body = responses.pop(0)
        return httpx.Response(status, json=body, request=httpx.Request("POST", url))
    return post


def test_segment_calls_api(monkeypatch):
    calls = []
    monkeypatch.setattr(settings, "META_API_KEY", "k")
    monkeypatch.setattr(sam.time, "sleep", lambda s: None)
    monkeypatch.setattr(sam.httpx, "post", _fake_post([(503, {}), (200, FIXTURE)], calls))
    boxes = sam.segment(b"jpeg")
    assert len(boxes) == 40 and len(calls) == 2  # one retry after the 503
    url, body, headers = calls[0]
    assert url.endswith("/responses") and headers["Authorization"] == "Bearer k"
    content = body["input"][0]["content"]
    assert content[0]["text"] == settings.SAM_PROMPT and content[1]["image_url"].startswith("data:image/jpeg;base64,")


def test_segment_raises(monkeypatch):
    monkeypatch.setattr(sam.time, "sleep", lambda s: None)
    monkeypatch.setattr(settings, "META_API_KEY", "")
    with pytest.raises(RuntimeError):
        sam.segment(b"jpeg")
    monkeypatch.setattr(settings, "META_API_KEY", "k")
    monkeypatch.setattr(sam.httpx, "post", _fake_post([(500, {}), (500, {})], []))
    with pytest.raises(httpx.HTTPStatusError):
        sam.segment(b"jpeg")
    monkeypatch.setattr(sam.httpx, "post", _fake_post([(200, {"output": []})], []))
    with pytest.raises(RuntimeError):
        sam.segment(b"jpeg")


def test_available(monkeypatch):
    monkeypatch.setattr(settings, "META_API_KEY", "k")
    monkeypatch.setattr(settings, "SPINE_SOURCE", "sam")
    assert sam.available()
    monkeypatch.setattr(settings, "SPINE_SOURCE", "gemini")
    assert not sam.available()
