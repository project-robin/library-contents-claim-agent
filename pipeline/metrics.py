"""Latency per stage and token/query cost per sweep."""
import json
import threading
import time
from contextlib import contextmanager

from pipeline import settings

_lock = threading.Lock()
_tokens = {}  # model -> [in, out]
_counters = {}  # e.g. serper_queries, gpu_seconds
_stages = {}  # stage -> seconds


def reset():  # ponytail: process-global meter, one sweep at a time
    with _lock:
        _tokens.clear(), _counters.clear(), _stages.clear()


def add_tokens(model, usage):
    if usage is None:
        return
    tin = (usage.prompt_token_count or 0)
    tout = (usage.candidates_token_count or 0) + (getattr(usage, "thoughts_token_count", 0) or 0)
    with _lock:
        t = _tokens.setdefault(model, [0, 0])
        t[0] += tin
        t[1] += tout


def count(name, n=1):
    with _lock:
        _counters[name] = _counters.get(name, 0) + n


@contextmanager
def stage(name):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        with _lock:
            _stages[name] = round(_stages.get(name, 0) + time.perf_counter() - t0, 2)


def snapshot():
    with _lock:
        cost = {}
        for model, (tin, tout) in _tokens.items():
            rin, rout = settings.TOKEN_RATES.get(model, (0, 0))
            cost[model] = round(tin / 1e6 * rin + tout / 1e6 * rout, 4)
        cost["serper"] = round(_counters.get("serper_queries", 0) * settings.SERPER_COST_PER_QUERY, 4)
        cost["gpu"] = round(_counters.get("gpu_seconds", 0) * settings.GPU_COST_PER_S, 4)
        return {
            "latency_s": dict(_stages),
            "tokens": {m: {"in": a, "out": b} for m, (a, b) in _tokens.items()},
            "counters": dict(_counters),
            "cost_usd": cost,
            "cost_usd_total": round(sum(cost.values()), 4),
        }


def save(path, extra=None):
    data = snapshot() | (extra or {})
    path.write_text(json.dumps(data, indent=2))
    return data
