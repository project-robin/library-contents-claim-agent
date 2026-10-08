"""Replay a recorded video through the live server as if it were the phone: same /ws protocol, same 3 fps JPEGs.
Tests the whole path (quality nudges, live reads, Live agent, pipeline, packet) without walking a room.

  uv run python -m eval.replay video.mp4                          # clean replay, ends the sweep when the video ends
  uv run python -m eval.replay video.mp4 --distort blur,glare --severity 0.6
  uv run python -m eval.replay video.mp4 --audio answers.m4a      # your recorded voice ("India", "it's a print")

Distortions: blur, motion, glare, dark, occlude, jpeg, drop. Agent speech is saved to agent.wav in the sweep dir.
"""
import argparse
import asyncio
import json
import subprocess
import time
import wave

import cv2
import numpy as np
import websockets

from pipeline import settings

FPS, MIC_RATE, AGENT_RATE, CHUNK_S = 3, 16000, 24000, 0.02


def distort(img, kinds, s, rng, i):
    """Deterministic for a given seed. s in [0, 1] is severity. Returns None for a dropped frame."""
    h, w = img.shape[:2]
    for k in kinds:
        if k == "drop" and rng.random() < s * 0.5:
            return None
        if k == "blur":
            r = 2 * int(1 + 8 * s) + 1
            img = cv2.GaussianBlur(img, (r, r), 0)
        elif k == "motion":
            n = int(5 + 30 * s)
            kern = np.zeros((n, n), np.float32)
            kern[n // 2] = 1 / n
            img = cv2.filter2D(img, -1, kern)
        elif k == "glare":  # a bright blob that drifts across the frame, like a window reflection
            cx, cy = int(w * (0.2 + 0.6 * ((i * 0.03) % 1))), int(h * 0.4)
            m = np.zeros((h, w), np.float32)
            cv2.ellipse(m, (cx, cy), (int(w * 0.25 * s + 20), int(h * 0.12 * s + 20)), 0, 0, 360, 1.0, -1)
            m = cv2.GaussianBlur(m, (0, 0), w * 0.04)[..., None]
            img = np.clip(img + m * 255 * (0.6 + 0.4 * s), 0, 255).astype(np.uint8)
        elif k == "dark":  # dim room: lower exposure plus sensor noise
            img = np.clip(img * (1 - 0.75 * s) + rng.normal(0, 12 * s, img.shape), 0, 255).astype(np.uint8)
        elif k == "occlude":  # a hand or a chair in front of the shelf
            ow, oh = int(w * (0.2 + 0.4 * s)), int(h * (0.2 + 0.4 * s))
            x, y = int(rng.integers(0, w - ow)), int(rng.integers(0, h - oh))
            img = img.copy()
            img[y:y + oh, x:x + ow] = (40, 45, 50)
        elif k == "jpeg":
            img = cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(90 - 80 * s)])[1], 1)
    return img


def frames(path):
    cap = cv2.VideoCapture(path)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    step, i = src_fps / FPS, 0
    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i * step))
        ok, img = cap.read()
        if not ok:
            return
        yield img
        i += 1


def pcm16(path):
    """Any audio/video file -> 16 kHz mono PCM16 (what the browser mic worklet sends)."""
    return subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-f", "s16le", "-ac", "1", "-ar", str(MIC_RATE), "-"],
                          capture_output=True, check=True).stdout


async def main(a):
    rng = np.random.default_rng(a.seed)
    kinds = [k for k in a.distort.split(",") if k]
    mic = pcm16(a.audio) if a.audio else b""
    log, agent_pcm, state = [], bytearray(), {}
    t0 = time.time()
    now = lambda: round(time.time() - t0, 2)

    async with websockets.connect(a.url, max_size=None) as ws:
        await ws.send(json.dumps({"type": "start", "device": f"replay:{a.video}"}))

        async def receive():
            async for m in ws:
                if isinstance(m, bytes):
                    if m[0] == 1:
                        agent_pcm.extend(m[1:])
                    continue
                j = json.loads(m)
                if j["type"] == "quality":
                    state["issues"] = state.get("issues", 0) + bool(j["issue"])
                    continue
                if j["type"] == "inventory":
                    state["live_books"], state["live_items"] = len(j["books"]), len(j["items"])
                    continue
                log.append({"t": now(), **j})
                print(f"[{now():7.2f}] {j['type']:9} {json.dumps({k: v for k, v in j.items() if k != 'type'})[:160]}")
                if j["type"] == "sweep":
                    state["id"] = j["id"]
                if j["type"] == "packet" or (j["type"] == "status" and j["text"].startswith("Packet failed")):
                    state["t_packet"] = now()
                    return

        async def send_audio():  # real-time mic stream: the recording, then silence so Live sees a live mic
            n, k = int(MIC_RATE * CHUNK_S) * 2, 0
            while True:
                chunk = mic[k:k + n] if k < len(mic) else b"\0" * n
                k += n
                await ws.send(b"\x01" + chunk.ljust(n, b"\0"))
                await asyncio.sleep(CHUNK_S)

        async def send_frames():
            await asyncio.sleep(a.lead)
            sent = dropped = 0
            for i, img in enumerate(frames(a.video)):
                t = time.time()
                img = distort(img, kinds, a.severity, rng, i) if kinds else img
                if img is None:
                    dropped += 1
                else:
                    await ws.send(b"\x02" + cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes())
                    sent += 1
                await asyncio.sleep(max(0.0, 1 / FPS - (time.time() - t)))
            state.update(frames_sent=sent, frames_dropped=dropped)
            await asyncio.sleep(a.tail)
            if not a.no_end:
                state["t_end"] = now()
                await ws.send(json.dumps({"type": "end"}))
                print(f"[{now():7.2f}] end sent")

        rx = asyncio.create_task(receive())
        tx = [asyncio.create_task(send_audio()), asyncio.create_task(send_frames())]
        try:
            await asyncio.wait_for(rx, a.timeout)
        except TimeoutError:
            print("timed out waiting for the packet")
        finally:
            for t in tx:
                t.cancel()

    sd = settings.SWEEPS_DIR / state.get("id", "_none")
    summary = {"video": a.video, "distort": kinds, "severity": a.severity, "seed": a.seed, **state,
               "time_to_packet_s": round(state["t_packet"] - state["t_end"], 1)
               if "t_packet" in state and "t_end" in state else None, "events": log}
    if sd.is_dir():  # server on this machine: keep the replay record next to the evidence
        (sd / "replay.json").write_text(json.dumps(summary, indent=1))
        with wave.open(str(sd / "agent.wav"), "wb") as w:
            w.setnchannels(1), w.setsampwidth(2), w.setframerate(AGENT_RATE), w.writeframes(bytes(agent_pcm))
    print(json.dumps({k: v for k, v in summary.items() if k != "events"}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--url", default="ws://localhost:8000/ws")
    ap.add_argument("--distort", default="", help="comma list: blur,motion,glare,dark,occlude,jpeg,drop")
    ap.add_argument("--severity", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--audio", help="your recorded voice, any format ffmpeg reads; silence after it ends")
    ap.add_argument("--lead", type=float, default=0, help="seconds before the first frame (agent greeting)")
    ap.add_argument("--tail", type=float, default=2, help="seconds after the last frame before End sweep")
    ap.add_argument("--no-end", action="store_true", help="never press End; wait for the agent to call end_sweep")
    ap.add_argument("--timeout", type=float, default=900)
    asyncio.run(main(ap.parse_args()))
