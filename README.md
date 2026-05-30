
# NBA Player Prop Betting Model

A machine learning system for predicting NBA player prop outcomes, managing a betting bankroll, and detecting value in sportsbook lines.

## Overview

- Fetches live player props from [The Odds API](https://the-odds-api.com)
- Collects and stores historical NBA game logs and odds data
- Trains XGBoost models per prop market (points, rebounds, assists, blocks, steals, 3-pointers)
- Calculates confidence scores and quarter-Kelly bet sizing based on your live bankroll
- Tracks bets, results, and P&L over time
- Detects arbitrage opportunities across bookmakers
- Supports combo props (PRA, PR, PA, RA)
- Streamlit dashboard with dark navy/gold theme

> **No pre-trained models or historical data are included.** You collect your own data using your own API key and train your own models from scratch.

## Requirements

- Python 3.9+
- [The Odds API](https://the-odds-api.com) key — sign up for a free account (500 requests/month)
- Internet connection for NBA data via `nba_api` (no key needed)

## Setup

```bash
git clone <this-repo>
cd NBA_Sports_Betting
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file in the repo root with your API key:

```
ODDS_API_KEY=your_key_here
```

## Data Collection & Training

**Complete these steps in order before using the dashboard.**

### Step 1 — Collect NBA game logs

Downloads player game logs from 2020 to present via `nba_api`. Free, no API key needed.

```bash
python -m scripts.update_game_logs
```

### Step 2 — Collect historical odds

Fetches today's player props from The Odds API and appends them to a local CSV. Run this daily during the season to build up history.

```bash
python -m scripts.live_odds_scraper
```

> The free tier gives ~500 requests/month. A typical day with 5–10 games uses around 10–20 requests. You need at least a few weeks of history before training will have enough rows per market.

### Step 3 — Prepare training data

Merges game logs with historical odds into one training dataset.

```bash
python -m scripts.data_preparation
```

### Step 4 — Train models

```bash
python -m scripts.model_trainer
```

Trains one model per market using TimeSeriesSplit cross-validation. Requires at least ~100 rows per market. Models are saved to `models/`.

## Running the Dashboard

```bash
streamlit run app.py
```

## Dashboard Pages

| Page | Description |
|------|-------------|
| Dashboard | KPIs, recent bets, bankroll summary |
| Charts | P&L over time, hit rates, ROI by market and bookmaker |
| Run Predictions | Fetch live props and generate today's predictions |
| Predictions Table | Browse and filter model predictions |
| Combo Props | PRA / PR / PA / RA combo prop predictions |
| Arbitrage & Lines | Cross-bookmaker arbitrage scanner |
| Select Model Bets | Mark predictions as bets to track |
| Manual / Parlay | Log bets placed outside the model (parlays, manual picks) |
| Settle Bets | Record results and update bankroll |
| Bet Tracker | Full bet history with filters and export |
| Bankroll | Set starting bankroll, view transaction history |
| Historical Odds | Browse collected odds data |
| Maintenance | Retrain models, update game logs |

## Bookmakers

Configured in `scripts/config.py`. Defaults to DraftKings, FanDuel, and BetMGM. Any bookmaker key supported by The Odds API can be added.

## Betting Logic

- **Confidence** — normal CDF centred on the model's prediction, using out-of-sample CV RMSE as the uncertainty band
- **Playoff mode** — confidence intervals are widened by 1.6x during the postseason (see below)
- **Kelly sizing** — quarter-Kelly, capped at 5% of bankroll per bet
- **Break-even hit rate** at -110 juice: 52.4%

## Playoff Mode

Models are trained on regular season data and have no knowledge of playoff rotations, defensive adjustments, or intensity shifts. To reduce overconfidence during the postseason:

- The uncertainty band is multiplied by `PLAYOFF_SIGMA_MULTIPLIER` (default 1.6x)
- Confidence scores drop and Kelly bet sizes shrink proportionally
- A warning banner appears on the prediction pages when playoff mode is active

Playoff windows are defined in `Config.PLAYOFF_DATE_RANGES` in `scripts/config.py`. Update the end date each year once the Finals conclude. To disable entirely, set `PLAYOFF_SIGMA_MULTIPLIER = 1.0`.

## Notes

- All data is stored locally as CSV/JSON — no database required
- `nba_api` is rate-limited; game log collection may take a few minutes on first run
- The free Odds API tier is sufficient for daily use on a normal slate. Running the dashboard multiple times a day will consume quota faster

## Demo Video

https://github.com/user-attachments/assets/bc682509-703e-4d0c-b7e5-1047ae744963


