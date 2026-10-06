"""Live sweep server: browser <-> Gemini Live relay, evidence capture, capture-quality nudges, live inventory.

Run:  uv run uvicorn app.server:app --host 0.0.0.0 --port 8000
"""
import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import cv2
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from google import genai
from google.genai import types

from app import live
from pipeline import frames, metrics, settings, vision
from pipeline import run as pipeline
from pipeline.books import norm

log = logging.getLogger("sweep")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
STATIC = Path(__file__).parent / "static"
settings.SWEEPS_DIR.mkdir(exist_ok=True)

NUDGES = {
    "blur": "frames are blurry: ask the user to slow down and hold the phone steady",
    "glare": "glare is hiding the spines: ask the user to tilt the phone slightly",
    "fast": "the camera is moving too fast to read spines: ask the user to slow down",
}

app = FastAPI()


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    await Session(ws).run()


def warm_modal():
    try:
        from gpu.modal_recon import remote
        remote().ping.remote()
        log.info("modal warm")
    except Exception as e:  # no Modal credentials yet: pipeline will report it at recon time
        log.warning("modal warm-up failed: %s", e)


class Session:
    def __init__(self, ws):
        self.ws, self.lock = ws, asyncio.Lock()
        self.live = None
        self.closed = self.ended = False
        self.n, self.prev_small, self.selector = 0, None, frames.KeyframeSelector()
        self.last_live_frame = self.last_nudge = 0.0
        self.bad_streak = 0
        self.read_sem, self.read_tasks = asyncio.Semaphore(3), set()
        self.books, self.items, self.unreadable_now = {}, [], 0
        self.last_item_t = {}

    # ---------- plumbing ----------
    def now(self):
        return round(time.time() - self.t0, 2)

    async def send_json(self, obj):
        try:
            async with self.lock:
                await self.ws.send_text(json.dumps(obj))
        except Exception:
            pass

    async def send_audio(self, pcm):
        try:
            async with self.lock:
                await self.ws.send_bytes(b"\x01" + pcm)
        except Exception:
            pass

    async def monitor(self, text, force=False):
        """Tell the agent about capture problems. Rate-limited so it does not nag."""
        if not self.live or (not force and (self.ended or time.time() - self.last_nudge < 6)):
            return
        self.last_nudge = time.time()
        try:
            await self.live.send_realtime_input(text=f"[MONITOR] {text}")
        except Exception as e:
            log.warning("monitor send failed: %s", e)
        await self.send_json({"type": "monitor", "text": text})

    def sweep_update(self, **kv):
        p = self.dir / "sweep.json"
        p.write_text(json.dumps(json.loads(p.read_text()) | kv, indent=1))

    def note(self, **kv):
        with open(self.dir / "notes.jsonl", "a") as f:
            f.write(json.dumps({"t": self.now(), **kv}) + "\n")

    # ---------- session ----------
    async def run(self):
        start = await self.ws.receive_json()
        metrics.reset()
        self.t0 = time.time()
        self.id = datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
        self.dir = settings.SWEEPS_DIR / self.id
        (self.dir / "frames").mkdir(parents=True)
        (self.dir / "reads").mkdir()
        loc = settings.LOCALES[settings.DEFAULT_COUNTRY]
        (self.dir / "sweep.json").write_text(json.dumps({
            "id": self.id, "captured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "device": start.get("device", ""), "country": settings.DEFAULT_COUNTRY, "currency": loc["currency"]}))
        self.frames_log = open(self.dir / "frames.jsonl", "a")
        await self.send_json({"type": "sweep", "id": self.id})
        log.info("sweep %s started", self.id)
        tasks = [asyncio.create_task(self.live_loop()), asyncio.create_task(asyncio.to_thread(warm_modal))]
        try:
            await self.browser_loop()
        except WebSocketDisconnect:
            pass
        finally:
            self.closed = True
            tasks[0].cancel()
            self.frames_log.close()

    async def browser_loop(self):
        while True:
            msg = await self.ws.receive()
            if msg["type"] == "websocket.disconnect":
                return
            if msg.get("bytes"):
                kind, data = msg["bytes"][0], msg["bytes"][1:]
                if kind == 1 and self.live:
                    try:
                        await self.live.send_realtime_input(audio=types.Blob(data=data, mime_type="audio/pcm;rate=16000"))
                    except Exception:
                        pass  # reconnecting
                elif kind == 2 and not self.ended:
                    await self.handle_frame(data)
            elif msg.get("text"):
                j = json.loads(msg["text"])
                if j.get("type") == "end":
                    await self.monitor("The user pressed End sweep. Tell them the packet is being prepared.", force=True)
                    asyncio.create_task(self.finish())

    async def live_loop(self):
        client = genai.Client(api_key=settings.GEMINI_API_KEY)
        handle, first = None, True
        while not self.closed:
            try:
                async with client.aio.live.connect(model=settings.LIVE_MODEL, config=live.config(handle)) as s:
                    self.live = s
                    log.info("live connected (resumed=%s)", handle is not None)
                    if first:
                        await s.send_realtime_input(text="[MONITOR] The user has opened the app. Greet them and begin.")
                        first = False
                    go_away = False
                    while not self.closed and not go_away:
                        async for m in s.receive():
                            u = m.session_resumption_update
                            if u and u.resumable and u.new_handle:
                                handle = u.new_handle
                            if m.go_away:
                                go_away = True
                                break
                            if m.usage_metadata:
                                metrics.add_tokens(settings.LIVE_MODEL, m.usage_metadata)
                            if m.tool_call:
                                for fc in m.tool_call.function_calls:
                                    asyncio.create_task(self.tool(fc))
                            sc = m.server_content
                            if not sc:
                                continue
                            for part in (sc.model_turn.parts if sc.model_turn else []):
                                if part.inline_data and part.inline_data.data:
                                    await self.send_audio(part.inline_data.data)
                            if sc.interrupted:
                                await self.send_json({"type": "interrupted"})
                            if sc.input_transcription and sc.input_transcription.text:
                                await self.send_json({"type": "transcript", "role": "user", "text": sc.input_transcription.text})
                            if sc.output_transcription and sc.output_transcription.text:
                                await self.send_json({"type": "transcript", "role": "agent", "text": sc.output_transcription.text})
            except asyncio.CancelledError:
                return
            except Exception as e:
                log.warning("live session error, reconnecting: %s", e)
                await asyncio.sleep(1)
            finally:
                self.live = None

    # ---------- evidence + quality ----------
    async def handle_frame(self, jpeg):
        t = self.now()
        self.n += 1
        name = f"{self.n:06d}.jpg"
        (self.dir / "frames" / name).write_bytes(jpeg)
        img = await asyncio.to_thread(frames.decode, jpeg)
        if img is None:
            return
        q, small = await asyncio.to_thread(frames.quality, img, self.prev_small)
        self.prev_small = small
        key = self.selector.offer(t, q, small)
        self.frames_log.write(json.dumps({"file": name, "t": t, **q, "key": key}) + "\n")
        self.frames_log.flush()
        await self.send_json({"type": "quality", "issue": q["issue"], "frame": name})
        self.bad_streak = self.bad_streak + 1 if q["issue"] else 0
        if self.bad_streak >= 4:
            await self.monitor(NUDGES[q["issue"]])
        if key:
            task = asyncio.create_task(self.read(name, jpeg))
            self.read_tasks.add(task)
            task.add_done_callback(self.read_tasks.discard)
        if self.live and t - self.last_live_frame >= 1.0:  # Live API takes ~1 fps
            self.last_live_frame = t
            h, w = img.shape[:2]
            s = 768 / max(h, w)
            small_jpeg = cv2.imencode(".jpg", cv2.resize(img, (int(w * s), int(h * s))), [cv2.IMWRITE_JPEG_QUALITY, 70])[1]
            try:
                await self.live.send_realtime_input(video=types.Blob(data=small_jpeg.tobytes(), mime_type="image/jpeg"))
            except Exception:
                pass

    async def read(self, name, jpeg):
        """Live spine/item read on a keyframe. Saved, so the post-sweep pipeline does not redo it."""
        async with self.read_sem:
            try:
                r = await asyncio.to_thread(vision.read_frame, jpeg)
            except Exception as e:
                log.warning("read %s failed: %s", name, e)
                return
        pipeline.dump(self.dir / "reads" / f"{name}.json", r)
        t = self.now()
        spines = r["spines"]
        for s in spines:
            if s["legibility"] != "none" and norm(s["title"]) and norm(s["title"]) not in self.books:
                self.books[norm(s["title"])] = {"title": s["title"], "author": s["author"], "frame": name}
        unreadable = sum(s["legibility"] == "none" for s in spines)
        self.unreadable_now = unreadable
        if unreadable >= 3 and unreadable >= 0.4 * len(spines):
            await self.monitor(f"{unreadable} of {len(spines)} spines in view are unreadable: ask the user to step "
                               "closer to this row and hold still for a moment")
        for k, it in enumerate(r["items"]):
            if t - self.last_item_t.get(it["category"], -99) < 3:  # ponytail: provisional live dedup only
                continue
            self.last_item_t[it["category"]] = t
            iid = f"i{len(self.items) + 1}"
            self.items.append({"id": iid, "category": it["category"], "description": it["description"],
                               "det_id": f"{name}:{k}"})
            if it["category"] == "art":
                await self.monitor(f"New item {iid}: {it['description']}. Ask whether it is an original or a print, "
                                   f"then call set_item_fact with item_id {iid}.", force=True)
        await self.send_json({"type": "inventory", "books": list(self.books.values()),
                              "items": self.items, "unreadable_now": self.unreadable_now})

    # ---------- tools ----------
    async def tool(self, fc):
        a, name = dict(fc.args or {}), fc.name
        res = {"ok": True}
        if name == "set_locale":
            cc = str(a.get("country_code", "")).upper()
            if cc in settings.LOCALES:
                self.sweep_update(country=cc, currency=settings.LOCALES[cc]["currency"])
                res["currency"] = settings.LOCALES[cc]["currency"]
            else:
                res = {"ok": False, "error": f"unsupported country; supported: {sorted(settings.LOCALES)}"}
        elif name == "log_note":
            self.note(kind="note", text=a.get("text", ""), target=a.get("target", "book"))
        elif name == "exclude_shelf":
            self.note(kind="exclude", text=a.get("reason", ""), target="book")
        elif name == "set_item_fact":
            it = next((i for i in self.items if i["id"] == a.get("item_id")), None)
            if it:
                self.note(kind="fact", det_id=it["det_id"], is_print=bool(a.get("is_print")), item_id=it["id"])
            else:
                res = {"ok": False, "error": "unknown item id"}
        elif name == "end_sweep":
            res["status"] = "building the claim packet"
            asyncio.create_task(self.finish())
        await self.send_json({"type": "tool", "name": name, "args": a, "result": res})
        try:
            await self.live.send_tool_response(function_responses=[types.FunctionResponse(id=fc.id, name=name, response=res)])
        except Exception as e:
            log.warning("tool response failed: %s", e)

    # ---------- packet ----------
    async def finish(self):
        if self.ended:
            return
        self.ended = True
        self.sweep_update(t_end_epoch=time.time(), duration_s=self.now())
        await self.send_json({"type": "status", "text": "Sweep ended. Building the claim packet..."})
        if self.read_tasks:
            await asyncio.gather(*self.read_tasks, return_exceptions=True)
        loop = asyncio.get_running_loop()
        progress = lambda s: asyncio.run_coroutine_threadsafe(self.send_json({"type": "status", "text": s}), loop)
        try:
            p = await asyncio.to_thread(pipeline.run, self.dir, "keyframes", False, progress)
        except Exception as e:
            log.exception("pipeline failed")
            await self.send_json({"type": "status", "text": f"Packet failed: {e}"})
            await self.monitor(f"[PACKET READY] The packet failed to build ({e}). Apologise briefly; it will be "
                               "rerun from the saved evidence.", force=True)
            return
        t, cur = p["totals"], p["sweep"]["currency"]
        summary = (f"{t['book_count']} books counted, {t['books_identified']} identified, "
                   f"{t['books_unidentified']} unidentified, {t['books_needs_appraisal']} need appraisal. "
                   f"Books replacement {cur} {t['books_replacement_cost']:,.0f}, used value {cur} "
                   f"{t['books_used_value']:,.0f}. {len(p['items'])} other items, replacement {cur} "
                   f"{t['items_replacement_cost_low']:,.0f} to {t['items_replacement_cost_high']:,.0f}. "
                   f"Room floor {p['room']['floor_area_m2']} square metres. {len(p['review_queue'])} lines need review.")
        await self.send_json({"type": "packet", "report": f"/sweeps/{self.id}/report.html",
                              "json": f"/sweeps/{self.id}/claim_packet.json", "summary": summary})
        await self.monitor(f"[PACKET READY] {summary}", force=True)


app.mount("/sweeps", StaticFiles(directory=settings.SWEEPS_DIR), name="sweeps")
app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")
