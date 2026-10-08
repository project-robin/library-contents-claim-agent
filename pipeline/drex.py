"""DREX decision model (Nace.AI): JSON state + yes/no questions in, calibrated probabilities out. Text only.
Used for every judgement call that decides money or identity, so each one carries a probability, not a guess."""
import random
import threading
import time

import httpx

from pipeline import metrics, settings


class _Usage:  # adapter so metrics.add_tokens can count DREX input tokens
    def __init__(self, u):
        self.prompt_token_count = u.get("input_tokens", 0)


_slots = threading.BoundedSemaphore(4)  # books and pricing call from thread pools; 8 at once drew sustained 429s
RETRY = (429, 529, 502, 503)


def _post(body):
    """Up to ~2 min of backoff: Retry-After when given, else exponential with jitter so threads do not retry in step."""
    for attempt in range(7):
        try:
            with _slots:
                r = httpx.post(f"{settings.DREX_BASE_URL}/v1/systemone", json=body, timeout=70,
                               headers={"Authorization": f"Bearer {settings.DREX_API_KEY}"})
        except httpx.TransportError:
            if attempt == 6:
                raise
            r = None
        if r is not None and r.status_code not in RETRY:
            return r
        metrics.count("drex_retries")
        wait = r.headers.get("retry-after", "") if r is not None else ""
        time.sleep(float(wait) if wait.replace(".", "", 1).isdigit() else min(30, 2 ** attempt) * random.uniform(0.5, 1.0))
    return r


def ask(state, questions):
    """questions: {id: instructions}. Returns {id: P(yes)}. Batches of 512, backs off on 429/529/5xx."""
    out = {}
    items = list(questions.items())
    for i in range(0, len(items), 512):
        body = {"model": settings.DREX_MODEL, "state": state,
                "questions": {k: {"type": "noul", "instructions": q} for k, q in items[i:i + 512]}}
        r = _post(body)
        r.raise_for_status()
        j = r.json()
        metrics.add_tokens("drex", _Usage(j.get("usage", {})))
        metrics.count("drex_calls")
        out.update({k: a["noul"] for k, a in j["answers"].items()})
    return out
