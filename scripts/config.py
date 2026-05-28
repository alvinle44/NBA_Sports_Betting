import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the repo root (one level above this file)
load_dotenv(Path(__file__).parent.parent / ".env")


class Config:
    # ── API keys ──────────────────────────────────────────────────────────────
    # Set these in your .env file — never hard-code them here.
    ODDS_API_KEY: str = os.getenv("ODDS_API_KEY", "")
    RAPIDAPI_KEY: str = os.getenv("RAPIDAPI_KEY", "")
    RAPIDAPI_HOST: str = os.getenv("RAPIDAPI_HOST", "nba-injuries-reports.p.rapidapi.com")

    # ── Directories ───────────────────────────────────────────────────────────
    DATA_DIR    = Path("data")
    MODELS_DIR  = Path("models")

    DATA_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)

    # ── Bookmakers ────────────────────────────────────────────────────────────
    BOOKMAKERS = ["draftkings", "fanduel", "betmgm"]

    # ── Markets ───────────────────────────────────────────────────────────────
    MARKETS = [
        "player_points",
        "player_assists",
        "player_rebounds",
        "player_threes",
        "player_steals",
        "player_blocks",
        "player_points_rebounds_assists",
        "player_points_rebounds",
        "player_points_assists",
        "player_rebounds_assists",
    ]

    # ── Market → NBA stat column ──────────────────────────────────────────────
    MARKET_TO_STAT = {
        "player_points":   "PTS",
        "player_assists":  "AST",
        "player_rebounds": "REB",
        "player_threes":   "FG3M",
        "player_blocks":   "BLK",
        "player_steals":   "STL",
    }

    # ── Betting thresholds ────────────────────────────────────────────────────
    MIN_EDGE = 2.0
    MIN_PROB = 0.55

    # ── Season ────────────────────────────────────────────────────────────────
    CURRENT_SEASON = "2025-26"

    # ── Teammate analyzer ─────────────────────────────────────────────────────
    MIN_TEAMMATE_MINUTES = 20

    # ── Playoff configuration ─────────────────────────────────────────────────
    # Models are trained on regular season data. During playoffs the uncertainty
    # band is widened by this multiplier to reduce overconfidence and shrink
    # Kelly bet sizes. Set to 1.0 to disable.
    PLAYOFF_SIGMA_MULTIPLIER = 1.6

    # (start, end) inclusive date strings for each playoff window.
    # Update the end date each year once the Finals conclude.
    PLAYOFF_DATE_RANGES = [
        ("2024-04-20", "2024-06-30"),  # 2023-24 playoffs
        ("2025-04-19", "2025-06-30"),  # 2024-25 playoffs
        ("2026-04-18", "2026-06-30"),  # 2025-26 playoffs
    ]

    @classmethod
    def is_playoff_period(cls, check_date=None):
        """Return True if check_date (default today) falls within a known playoff window."""
        from datetime import date as _date
        d = check_date or _date.today()
        if isinstance(d, str):
            d = _date.fromisoformat(d)
        for start_str, end_str in cls.PLAYOFF_DATE_RANGES:
            if _date.fromisoformat(start_str) <= d <= _date.fromisoformat(end_str):
                return True
        return False
