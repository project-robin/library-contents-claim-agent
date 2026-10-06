"""DREX decision model (Nace.AI): JSON state + yes/no questions in, calibrated probabilities out. Text only.
Used for every judgement call that decides money or identity, so each one carries a probability, not a guess."""
import time

import httpx

from pipeline import metrics, settings


class _Usage:  # adapter so metrics.add_tokens can count DREX input tokens
    def __init__(self, u):
        self.prompt_token_count = u.get("input_tokens", 0)


def ask(state, questions):
    """questions: {id: instructions}. Returns {id: P(yes)}. Batches of 512, retries on 429/529."""
    out = {}
    items = list(questions.items())
    for i in range(0, len(items), 512):
        body = {"model": settings.DREX_MODEL, "state": state,
                "questions": {k: {"type": "noul", "instructions": q} for k, q in items[i:i + 512]}}
        for attempt in range(4):
            r = httpx.post(f"{settings.DREX_BASE_URL}/v1/systemone", json=body, timeout=70,
                           headers={"Authorization": f"Bearer {settings.DREX_API_KEY}"})
            if r.status_code not in (429, 529, 502, 503):
                break
            time.sleep(1.5 * (attempt + 1))
        r.raise_for_status()
        j = r.json()
        metrics.add_tokens("drex", _Usage(j.get("usage", {})))
        metrics.count("drex_calls")
        out.update({k: a["noul"] for k, a in j["answers"].items()})
    return out
