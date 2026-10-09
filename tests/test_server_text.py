"""The phone sends typed replies, not mic audio: on Chrome for Android the agent's loudspeaker voice leaked back into
the mic and the intake looped on 'which country and currency?'. A typed reply must reach the agent as the user's turn
and unlock set_locale at once (no wait for a speech transcript)."""
import asyncio
import json
from types import SimpleNamespace

from app import server


class FakeWS:
    def __init__(self, msgs):
        self.msgs, self.sent = list(msgs), []

    async def receive(self):
        return self.msgs.pop(0) if self.msgs else {"type": "websocket.disconnect"}

    async def send_text(self, t):
        self.sent.append(json.loads(t))


class FakeLive:
    def __init__(self):
        self.texts, self.tool_responses = [], []

    async def send_realtime_input(self, text=None, **kw):
        self.texts.append(text)

    async def send_tool_response(self, function_responses):
        self.tool_responses += function_responses


def test_typed_reply_is_the_users_turn_and_unlocks_set_locale(tmp_path):
    ws = FakeWS([{"type": "websocket.receive", "text": json.dumps({"type": "text", "text": "India, rupees"})}])
    s = server.Session(ws)
    s.live, s.dir, s.t0 = FakeLive(), tmp_path, 0.0
    (tmp_path / "sweep.json").write_text(json.dumps({"country": "IN", "currency": "INR"}))

    async def go():
        await s.browser_loop()
        await s.tool(SimpleNamespace(args={"country_code": "GB"}, name="set_locale", id="1"))
    asyncio.run(go())
    assert s.live.texts == ["[USER] India, rupees"]
    assert s.live.tool_responses[0].response == {"ok": True, "currency": "GBP"}
    assert json.loads((tmp_path / "sweep.json").read_text())["country"] == "GB"


def test_same_art_seen_again_is_asked_about_once(tmp_path, monkeypatch):
    """Sweep 200647: every frame read re-detected the Spider-Man art as a new item and the agent asked 9 times
    ('Print i said'). The same piece under another description is not asked again; a new piece waits ART_GAP_S."""
    from pipeline import vision
    seen = iter(["Spider-Man headboard painting", "Spiderman poster", "abstract headboard artwork",
                 "framed portrait on the wall", "framed portrait on the wall"])
    monkeypatch.setattr(vision, "read_frame", lambda jpeg: {"spines": [], "issue": "none", "items": [
        {"category": "art", "description": next(seen), "material": "", "brand_model": "", "dimensions_cm": ""}]})
    s = server.Session(FakeWS([]))
    s.live, s.dir = FakeLive(), tmp_path
    (tmp_path / "reads").mkdir()
    clock = iter([0, 5, 10, 15, 40])
    s.now = lambda: next(clock)
    for i in range(5):
        asyncio.run(s.read(f"{i:06d}.jpg", b"x"))
    asked = [t for t in s.live.texts if "original or a print" in t]
    assert len(asked) == 2 and "Spider-Man" in asked[0] and "portrait" in asked[1]  # portrait at 15 s waited: 40 s
