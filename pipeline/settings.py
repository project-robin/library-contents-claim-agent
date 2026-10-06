"""All tunables in one place. Values read from .env; defaults are calibration knobs, not test-room facts."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _f(name, default):
    return float(os.getenv(name, default))


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
SERPER_API_KEY = os.getenv("SERPER_API_KEY", "")
EBAY_CLIENT_ID = os.getenv("EBAY_CLIENT_ID", "")
EBAY_CLIENT_SECRET = os.getenv("EBAY_CLIENT_SECRET", "")
GOOGLE_BOOKS_API_KEY = os.getenv("GOOGLE_BOOKS_API_KEY", "")  # optional, raises quota

LIVE_MODEL = os.getenv("LIVE_MODEL", "gemini-3.1-flash-live-preview")
VISION_MODEL = os.getenv("VISION_MODEL", "gemini-flash-latest")
CHECKER_MODEL = os.getenv("CHECKER_MODEL", "gemini-pro-latest")
LIVE_VOICE = os.getenv("LIVE_VOICE", "Kore")

SWEEPS_DIR = Path(os.getenv("SWEEPS_DIR", "sweeps"))
DEFAULT_COUNTRY = os.getenv("DEFAULT_COUNTRY", "IN")
COMPARISON_COUNTRY = os.getenv("COMPARISON_COUNTRY", "US")

# Locale = market settings, not prices. Add a row to support a new country.
LOCALES = {
    "IN": {"currency": "INR", "gl": "in", "symbols": ("₹", "Rs", "INR"), "ebay": None, "appraisal_threshold": 5000},
    "US": {"currency": "USD", "gl": "us", "symbols": ("$", "USD"), "ebay": "EBAY_US", "appraisal_threshold": 75},
    "GB": {"currency": "GBP", "gl": "uk", "symbols": ("£", "GBP"), "ebay": "EBAY_GB", "appraisal_threshold": 60},
}
USED_FALLBACK_MARKET = os.getenv("USED_FALLBACK_MARKET", "EBAY_US")

# Metric scale. Measure the printed marker with a ruler and set this; printers rescale.
MARKER_SIZE_CM = _f("MARKER_SIZE_CM", "15.0")
BOOK_PRIOR_HEIGHT_CM = _f("BOOK_PRIOR_HEIGHT_CM", "21.0")  # median upright spine, fallback only

# Capture quality (tune on your phone: Laplacian variance at 640 px width)
BLUR_MIN = _f("BLUR_MIN", "60")
GLARE_MAX = _f("GLARE_MAX", "0.06")
MOTION_MAX = _f("MOTION_MAX", "28")
NOVELTY_MIN = _f("NOVELTY_MIN", "10")
MAX_RECON_FRAMES = int(os.getenv("MAX_RECON_FRAMES", "150"))

# Identification
CONF_IDENTIFIED = _f("CONF_IDENTIFIED", "0.75")
CONF_MIN = _f("CONF_MIN", "0.5")
APPRAISAL_YEAR = int(os.getenv("APPRAISAL_YEAR", "1950"))

# $ per 1M tokens (input, output). List prices change: update before quoting cost.
TOKEN_RATES = {
    LIVE_MODEL: (_f("RATE_LIVE_IN", "3.0"), _f("RATE_LIVE_OUT", "12.0")),
    VISION_MODEL: (_f("RATE_VISION_IN", "0.5"), _f("RATE_VISION_OUT", "3.0")),
    CHECKER_MODEL: (_f("RATE_CHECKER_IN", "2.0"), _f("RATE_CHECKER_OUT", "12.0")),
}
SERPER_COST_PER_QUERY = _f("SERPER_COST_PER_QUERY", "0.001")
GPU_COST_PER_S = _f("GPU_COST_PER_S", "0.00054")  # Modal L40S
