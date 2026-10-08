"""All tunables in one place, typed and validated at import (pydantic-settings reads .env).
Defaults are calibration knobs, not facts about any test room."""
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", case_sensitive=False)

    GEMINI_API_KEY: str = ""
    SERPER_API_KEY: str = ""
    DREX_API_KEY: str = ""
    DREX_BASE_URL: str = "https://drex.nace.ai"
    DREX_MODEL: str = "drex-latest"
    HF_TOKEN: str = ""
    EBAY_CLIENT_ID: str = ""
    EBAY_CLIENT_SECRET: str = ""
    GOOGLE_BOOKS_API_KEY: str = ""  # optional, raises quota
    META_API_KEY: str = ""  # SAM 3.1 spine segmentation (pipeline/sam.py); empty -> spines from the vision model
    META_BASE_URL: str = "https://api.meta.ai/v1"

    LIVE_MODEL: str = "gemini-3.8-live"
    VISION_MODEL: str = "gemini-3.8-flash"
    VISION_FALLBACK_MODEL: str = "gemini-3.5-flash-lite"  # on 503 "high demand": 2x faster, 18/20 titles vs 3.8 Flash
    VISION_THINKING: str = "low"
    VISION_BREAKER_S: int = Field(600, ge=0)  # after a primary-model failure, use the fallback this long
    VISION_TIMEOUT_S: int = Field(120, ge=10)  # a dense close-up once ran past 4 min on 3.8 Flash; then fall back
    CHECKER_MODEL: str = "gemini-pro-latest"
    LIVE_VOICE: str = "Kore"
    SAM_MODEL: str = "sam-3.1"
    SAM_PROMPT: str = "book spine"  # one short noun phrase; a comma list returns nothing
    SPINE_SOURCE: str = "sam"  # sam | gemini ("gemini" = old behaviour, also the automatic fallback when SAM fails)
    SAM_CONCURRENCY: int = Field(8, ge=1)
    RECON_MODEL: str = "mapanything"  # mapanything | vggt (both on Modal; pick by the geometry spike)

    SWEEPS_DIR: Path = Path("sweeps")
    DEFAULT_COUNTRY: str = "IN"
    COMPARISON_COUNTRY: str = "US"
    USED_FALLBACK_MARKET: str = "EBAY_US"

    # Metric scale. Measure the printed marker with a ruler and set this; printers rescale.
    MARKER_SIZE_CM: float = Field(15.0, gt=1, lt=60)
    BOOK_PRIOR_HEIGHT_CM: float = 21.0  # median upright spine, fallback only

    # Capture quality (tune on your phone: Laplacian variance at 640 px width)
    BLUR_MIN: float = 60
    GLARE_MAX: float = Field(0.06, ge=0, le=1)
    MOTION_MAX: float = 28
    NOVELTY_MIN: float = 10
    MAX_RECON_FRAMES: int = Field(150, ge=8)
    MIN_ROOM_SIDE_M: float = 1.2  # narrower than this means the walls were not captured, not a real room
    LIVE_READ_CONCURRENCY: int = Field(8, ge=1)

    # Decisions (DREX calibrated probabilities)
    CONF_IDENTIFIED: float = Field(0.75, ge=0, le=1)
    CONF_MIN: float = Field(0.5, ge=0, le=1)
    LISTING_MATCH_MIN: float = Field(0.8, ge=0, le=1)
    OCR_ENABLED: bool = True  # PP-OCRv5 spine witness on Modal CPU (gpu/modal_ocr.py)
    OCR_MIN_SCORE: float = Field(0.8, ge=0, le=1)
    OCR_VETO_SCORE: float = Field(0.9, ge=0, le=1)  # a veto needs 4+ char tokens at this confidence
    MIN_PRICE_LISTINGS: int = Field(3, ge=1)  # fewer matched new listings -> priced but sent to review
    APPRAISAL_MIN: float = Field(0.5, ge=0, le=1)

    # $ per 1M tokens (input, output). List prices change: update before quoting cost.
    RATE_LIVE_IN: float = 3.0
    RATE_LIVE_OUT: float = 12.0
    RATE_VISION_IN: float = 0.75
    RATE_VISION_OUT: float = 3.75
    RATE_CHECKER_IN: float = 2.0
    RATE_CHECKER_OUT: float = 12.0
    RATE_DREX_IN: float = 0.5
    SERPER_COST_PER_QUERY: float = 0.001
    GPU_COST_PER_S: float = 0.00054  # Modal L40S
    SAM_COST_PER_IMAGE: float = 0.0  # placeholder: no list price published for SAM 3.1 on the Meta API yet
    OCR_COST_PER_S: float = 0.0001  # Modal CPU container, 4 cores (unverified list price; ~$0.0000131/core/s)


# ponytail: module attributes keep call sites as `settings.X`
globals().update(Settings().model_dump())

REQUIRED = {"live": ["GEMINI_API_KEY"], "pipeline": ["GEMINI_API_KEY", "SERPER_API_KEY", "DREX_API_KEY"],
            "books": ["GEMINI_API_KEY", "DREX_API_KEY"]}  # a run that stops before pricing (--until)


def require(what):
    """Fail at startup with a clear message instead of three minutes into a sweep."""
    missing = [k for k in REQUIRED[what] if not globals()[k]]
    if missing:
        raise SystemExit(f"Missing in .env: {', '.join(missing)} (see .env.example)")


# Locale = market settings, not prices. Add a row to support a new country.
LOCALES = {
    "IN": {"currency": "INR", "gl": "in", "symbols": ("₹", "Rs", "INR"), "ebay": None, "appraisal_threshold": 5000},
    "US": {"currency": "USD", "gl": "us", "symbols": ("$", "USD"), "ebay": "EBAY_US", "appraisal_threshold": 75},
    "GB": {"currency": "GBP", "gl": "uk", "symbols": ("£", "GBP"), "ebay": "EBAY_GB", "appraisal_threshold": 60},
}

# Google list prices, $ per 1M tokens (input, output incl. thinking), ai.google.dev/gemini-api/docs/pricing on
# 2026-10-07. Gemini 3.8 Flash doubles to (1.50, 7.50) on 2027-01-01. Prices belong to the model, not its role:
# keyed by role, a flash-lite checker was billed at Pro rates (6x too high).
MODEL_PRICES = {
    "gemini-3.8-flash": (0.75, 3.75),
    "gemini-3.5-flash-lite": (0.30, 2.50),
    "gemini-3.1-pro": (2.0, 12.0),
    "gemini-pro-latest": (2.0, 12.0),
}

TOKEN_RATES = {  # role defaults for models not in MODEL_PRICES, then the per-model list price wins
    LIVE_MODEL: (RATE_LIVE_IN, RATE_LIVE_OUT),  # noqa: F821 (set via globals above)
    VISION_MODEL: (RATE_VISION_IN, RATE_VISION_OUT),  # noqa: F821
    VISION_FALLBACK_MODEL: (RATE_VISION_IN, RATE_VISION_OUT),  # noqa: F821
    CHECKER_MODEL: (RATE_CHECKER_IN, RATE_CHECKER_OUT),  # noqa: F821
    "drex": (RATE_DREX_IN, 0.0),  # noqa: F821 (only input tokens are billed)
} | MODEL_PRICES
