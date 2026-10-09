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
