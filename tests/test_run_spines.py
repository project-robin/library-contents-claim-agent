"""The SAM spines stage and the runner's --skip/--until flags, offline (sam.segment stubbed)."""
import json

import pytest

from pipeline import books, geometry, review, run, sam, settings


def box(i, x1, y1, x2, y2, w=1000, h=500):
    return {"id": i, "box": [x1, y1, x2, y2], "w": w, "h": h}


def test_to_spines_normalises_inclusive_pixels():
    up, flat = sam.to_spines([box(0, 99, 49, 199, 449), box(1, 0, 0, 499, 99)])
    assert up["box_2d"] == [98.0, 99.0, 900.0, 200.0]  # [ymin, xmin, ymax, xmax] /1000, x2/y2 inclusive (+1)
    assert up["orientation"] == "upright" and flat["orientation"] == "flat"  # 499 wide > 1.6 * 99 high
    assert up["title"] == "" and up["legibility"] == "none" and up["source"] == "sam" and flat["sam_id"] == 1


@pytest.fixture
def sweep(tmp_path, monkeypatch):
    (tmp_path / "frames").mkdir()
    kfs = [{"frame": f"{n:06d}.jpg", "t": float(n)} for n in range(1, 11)]
    for k in kfs:
        (tmp_path / "frames" / k["frame"]).write_bytes(k["frame"].encode())  # frame name as the 'jpeg'
    monkeypatch.setattr(settings, "META_API_KEY", "k")
    monkeypatch.setattr(settings, "SPINE_SOURCE", "sam")
    return tmp_path, kfs


def stub_segment(monkeypatch, fail=()):
    calls = []

    def segment(jpeg):
        calls.append(jpeg.decode())
        if jpeg.decode() in fail:
            raise RuntimeError("503")
        return [box(0, 10, 10, 50, 400)]
    monkeypatch.setattr(sam, "segment", segment)
    return calls


def test_stage_caches_per_frame(sweep, monkeypatch):
    sd, kfs = sweep
    calls = stub_segment(monkeypatch)
    out = run.segment_spines(sd, kfs, log=lambda s: None)
    assert len(out) == 10 and len(calls) == 10 and out["000001.jpg"][0]["source"] == "sam"
    assert json.loads((sd / "spines" / "000001.jpg.json").read_text())["source"] == "sam"
    assert run.segment_spines(sd, kfs, log=lambda s: None) == out and len(calls) == 10  # rerun: all cached


def test_stage_off_without_key(sweep, monkeypatch):
    sd, kfs = sweep
    calls = stub_segment(monkeypatch)
    monkeypatch.setattr(settings, "META_API_KEY", "")
    assert run.segment_spines(sd, kfs, log=lambda s: None) is None and not calls


def test_stage_failures(sweep, monkeypatch):
    sd, kfs = sweep
    stub_segment(monkeypatch, fail={"000003.jpg"})  # 1 of 10 = 10%: keep SAM, that frame has no spines
    out = run.segment_spines(sd, kfs, log=lambda s: None)
    assert len(out) == 9 and "000003.jpg" not in out and not (sd / "spines" / "000003.jpg.json").exists()
    calls = stub_segment(monkeypatch, fail={"000003.jpg", "000004.jpg"})  # rerun retries only the failed frame
    assert len(run.segment_spines(sd, kfs, log=lambda s: None)) == 9 and calls == ["000003.jpg"]
    (sd / "spines" / "000004.jpg.json").unlink()
    calls = stub_segment(monkeypatch, fail={"000003.jpg", "000004.jpg"})  # 2 of 10 > 10%: whole sweep falls back
    assert run.segment_spines(sd, kfs, log=lambda s: None) is None and sorted(calls) == ["000003.jpg", "000004.jpg"]


def test_loader_for_reruns(sweep, monkeypatch):
    sd, kfs = sweep
    stub_segment(monkeypatch)
    run.segment_spines(sd, kfs, log=lambda s: None)
    assert len(run.load_spines(sd, kfs)) == 10
    monkeypatch.setattr(settings, "SPINE_SOURCE", "gemini")
    assert run.load_spines(sd, kfs) is None
    monkeypatch.setattr(settings, "SPINE_SOURCE", "sam")
    assert run.load_spines(sd / "missing", kfs) is None


def test_skip_and_until(sweep, monkeypatch):
    """A still photo: --from spines --skip recon,geometry --until checker runs spines, books, checker and stops."""
    sd, kfs = sweep
    kfs = kfs[:1]
    (sd / "sweep.json").write_text(json.dumps({"id": "still", "country": "IN"}))
    (sd / "keyframes.json").write_text(json.dumps(kfs))
    (sd / "reads").mkdir()
    (sd / "reads" / "000001.jpg.json").write_text(json.dumps({"spines": [], "items": [], "issue": ""}))
    (sd / "geometry.json").write_text(json.dumps({"scale": None}))
    stub_segment(monkeypatch)
    ran, required = [], []
    monkeypatch.setattr(settings, "require", required.append)
    monkeypatch.setattr(run, "recon", lambda *a: ran.append("recon"))
    monkeypatch.setattr(run, "read_missing", lambda *a: ran.append("reads") or {})
    monkeypatch.setattr(geometry, "run", lambda *a: ran.append("geometry"))
    monkeypatch.setattr(geometry, "Geo", lambda sd: type("G", (), {"ok": False})())

    def fake_books(sd_, reads, geo, spines):
        ran.append(("books", len(spines["000001.jpg"])))
        return {"books": [], "stats": {}}
    monkeypatch.setattr(books, "run", fake_books)
    monkeypatch.setattr(review, "checker", lambda sd_, bs: ran.append("checker"))
    out = run.run(sd, start="spines", skip=["recon", "geometry"], until="checker", log=lambda s: None)
    assert out == {"books": [], "stats": {}} and required == ["books"]  # no pricing: Serper key not needed
    assert ran == [("books", 1), "checker"]  # reads come before --from spines; recon/geometry skipped
    assert (sd / "metrics.json").exists() and not (sd / "claim_packet.json").exists()
    ran.clear()
    run.run(sd, start="books", skip=["geometry"], until="books", log=lambda s: None)
    assert ran == [("books", 1)]  # --until books: the checker does not run


def test_failed_frame_read_is_skipped_not_fatal(tmp_path, monkeypatch):
    """A live sweep on a spent Gemini quota died re-reading frames; now they are skipped, counted and retried later."""
    from pipeline import metrics, vision
    (tmp_path / "frames").mkdir()
    (tmp_path / "frames" / "000001.jpg").write_bytes(b"x")

    def spent(jpeg):
        raise RuntimeError("429 RESOURCE_EXHAUSTED")
    monkeypatch.setattr(vision, "read_frame", spent)
    metrics.reset()
    assert run.read_missing(tmp_path, [{"frame": "000001.jpg"}]) == {}
    assert metrics.get("read_failures") == 1 and not (tmp_path / "reads" / "000001.jpg.json").exists()
