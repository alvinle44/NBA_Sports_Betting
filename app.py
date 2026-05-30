"""
NBA Props Dashboard  —  streamlit run app.py
"""
import json
import sys
import traceback
from datetime import datetime, date
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── project root on sys.path so scripts.* imports work ──────────────────────
ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── paths ────────────────────────────────────────────────────────────────────
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
TRACKER_FILE = DATA_DIR / "bet_tracker.csv"
BANKROLL_FILE = DATA_DIR / "bankroll.json"
ONGOING_FILE = DATA_DIR / "ongoing_odds_collection.csv"

TRACKER_COLS = [
    "date", "bet_type", "player_name", "market", "line", "bet_direction", "odds",
    "stake", "bookmaker", "predicted_value", "confidence", "expected_value",
    "result_status", "actual_value", "profit_loss", "bankroll_after",
    "parlay_legs", "notes",
]

# ── CSS theme ─────────────────────────────────────────────────────────────────
CSS = """
<style>
/* ── base ── */
[data-testid="stAppViewContainer"] { background: #0e1117; }
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1f2e 0%, #0e1117 100%);
    border-right: 1px solid #1d428a33;
}

/* ── sidebar title ── */
.sidebar-title {
    font-size: 1.5rem; font-weight: 800; letter-spacing: 1px;
    color: #ffc72c; text-align: center; padding: 0.5rem 0 0.25rem;
}
.sidebar-bankroll {
    text-align: center; font-size: 1.1rem; color: #e0e0e0;
    background: #1d428a22; border-radius: 8px; padding: 6px 12px;
    margin-bottom: 0.5rem;
}

/* ── KPI cards ── */
.kpi-grid { display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 1rem; }
.kpi-card {
    flex: 1; min-width: 140px;
    background: #1a1f2e;
    border: 1px solid #1d428a55;
    border-radius: 12px;
    padding: 16px 20px;
    text-align: center;
}
.kpi-label { font-size: 0.75rem; color: #8892a4; text-transform: uppercase; letter-spacing: 1px; }
.kpi-value { font-size: 1.8rem; font-weight: 700; color: #ffffff; margin: 4px 0; }
.kpi-delta { font-size: 0.8rem; }
.kpi-pos { color: #2ecc71; } .kpi-neg { color: #e74c3c; } .kpi-neu { color: #8892a4; }

/* ── section headers ── */
h1, h2 { color: #ffffff !important; }
h3 { color: #ffc72c !important; }

/* ── primary buttons ── */
[data-testid="stButton"] > button[kind="primary"] {
    background: linear-gradient(135deg, #1d428a, #2756b3);
    color: white; border: none; border-radius: 8px;
    font-weight: 600; letter-spacing: 0.5px;
    transition: opacity 0.2s;
}
[data-testid="stButton"] > button[kind="primary"]:hover { opacity: 0.85; }

/* ── divider ── */
hr { border-color: #1d428a44 !important; }

/* ── expander ── */
[data-testid="stExpander"] {
    background: #1a1f2e; border: 1px solid #1d428a44;
    border-radius: 8px;
}

/* ── tag badges ── */
.badge {
    display: inline-block; padding: 2px 10px; border-radius: 20px;
    font-size: 0.75rem; font-weight: 600;
}
.badge-green { background: #1a4731; color: #2ecc71; }
.badge-red   { background: #4a1a1a; color: #e74c3c; }
.badge-gold  { background: #3a2e00; color: #ffc72c; }
.badge-grey  { background: #1e2130; color: #8892a4; }

/* ── nav radio buttons ── */
[data-testid="stRadio"] label {
    padding: 6px 12px; border-radius: 6px;
    transition: background 0.15s;
}
[data-testid="stRadio"] label:hover { background: #1d428a22; }
</style>
"""

# ── helpers ───────────────────────────────────────────────────────────────────

def load_bankroll() -> dict:
    if BANKROLL_FILE.exists():
        with open(BANKROLL_FILE) as f:
            return json.load(f)
    return {"starting": 1000.0, "current": 1000.0}


def save_bankroll(data: dict):
    BANKROLL_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(BANKROLL_FILE, "w") as f:
        json.dump(data, f, indent=2)


def load_tracker() -> pd.DataFrame:
    if TRACKER_FILE.exists():
        df = pd.read_csv(TRACKER_FILE)
        for col in TRACKER_COLS:
            if col not in df.columns:
                df[col] = None
        return df
    return pd.DataFrame(columns=TRACKER_COLS)


def save_tracker(df: pd.DataFrame):
    TRACKER_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(TRACKER_FILE, index=False)


def load_predictions(dt=None) -> pd.DataFrame:
    if dt is None:
        dt = date.today()
    f = DATA_DIR / f"predictions_{dt.strftime('%Y-%m-%d')}.csv"
    if f.exists():
        return pd.read_csv(f)
    return pd.DataFrame()


def calc_pnl(stake: float, odds: float, won: bool, push: bool) -> float:
    if push:
        return 0.0
    if not won:
        return -stake
    if odds > 0:
        return round(stake * odds / 100, 2)
    return round(stake * 100 / abs(odds), 2)


def load_ongoing_odds() -> pd.DataFrame:
    if ONGOING_FILE.exists():
        return pd.read_csv(ONGOING_FILE)
    return pd.DataFrame()


def american_to_decimal(odds: float) -> float:
    return (odds / 100 + 1) if odds > 0 else (100 / abs(odds) + 1)


def decimal_to_american(dec: float) -> int:
    if dec >= 2.0:
        return int(round((dec - 1) * 100))
    return int(round(-100 / (dec - 1)))


def parlay_combined_odds(leg_odds) -> int:
    dec = 1.0
    for o in leg_odds:
        dec *= american_to_decimal(o)
    return decimal_to_american(dec)


def kpi(col, label: str, value: str, delta: str = "", positive=None):
    """Render a styled KPI card inside a Streamlit column."""
    if delta:
        cls = "kpi-pos" if positive else ("kpi-neg" if positive is False else "kpi-neu")
        delta_html = f'<div class="kpi-delta {cls}">{delta}</div>'
    else:
        delta_html = ""
    col.markdown(
        f'<div class="kpi-card">'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>'
        f'{delta_html}'
        f'</div>',
        unsafe_allow_html=True,
    )


def dedup_odds(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicates; preserve line movement and multi-book snapshots."""
    dedup_cols = [c for c in [
        "player_name", "market", "line", "odds_over", "odds_under",
        "bookmaker", "timestamp",
    ] if c in df.columns]
    if dedup_cols:
        df = df.drop_duplicates(subset=dedup_cols)
    return df


# ── page: Run Predictions ─────────────────────────────────────────────────────

def _playoff_warning():
    """Show a warning banner if today is in a known playoff window."""
    try:
        from scripts.config import Config
        if Config.is_playoff_period():
            st.warning(
                "**Playoff mode active** — models were trained on regular season data. "
                f"Confidence intervals are widened by {Config.PLAYOFF_SIGMA_MULTIPLIER}x "
                "and bet sizes are reduced accordingly. Use extra caution.",
                icon="⚠️",
            )
    except Exception:
        pass


def page_run_predictions():
    st.header("Run Daily Predictions")
    _playoff_warning()

    today = date.today()
    pred_file = DATA_DIR / f"predictions_{today.strftime('%Y-%m-%d')}.csv"
    if pred_file.exists():
        df = pd.read_csv(pred_file)
        st.success(f"Predictions already exist for {today} — {len(df)} props loaded.")
        st.caption("Re-run to refresh with new odds.")

    if st.button("Run Daily Predictions", type="primary"):
        with st.spinner("Fetching props, building features, predicting…"):
            try:
                from scripts.daily_predictor import DailyPredictor
                predictor = DailyPredictor()
                predictions = predictor.run_daily_predictions(save_to_ongoing=True)

                if predictions.empty:
                    st.warning("No predictions returned. Check that today's props are available.")
                else:
                    out = DATA_DIR / f"predictions_{today.strftime('%Y-%m-%d')}.csv"
                    predictions.to_csv(out, index=False)
                    pos_ev = predictions[predictions["expected_value"] > 0]
                    st.success(
                        f"Done! {len(predictions)} props analysed — "
                        f"{len(pos_ev)} positive-EV bets found."
                    )
                    st.dataframe(
                        pos_ev.sort_values("expected_value", ascending=False).head(20),
                        use_container_width=True,
                    )
            except Exception:
                st.error("Prediction run failed:")
                st.code(traceback.format_exc())

    st.divider()
    st.subheader("Collect Latest Odds Only")
    st.caption("Scrapes today's props and appends to the ongoing odds collection (no predictions).")
    if st.button("Collect Latest Odds"):
        with st.spinner("Scraping…"):
            try:
                from scripts.live_odds_scraper import LiveOddsScraper
                scraper = LiveOddsScraper()
                wide = scraper.get_todays_props()
                if wide.empty:
                    st.warning("No props returned from scraper.")
                else:
                    # Save in long format to match historical_odds_combined.csv
                    long_df = scraper.to_long_format(wide)
                    long_df["collection_date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    existing = load_ongoing_odds()
                    combined = pd.concat([existing, long_df], ignore_index=True)
                    combined = dedup_odds(combined)
                    combined.to_csv(ONGOING_FILE, index=False)
                    new_rows = len(combined) - len(existing)
                    st.success(
                        f"Collected {len(wide)} props ({len(long_df)} rows in long format)"
                        f" — {new_rows} new rows added."
                    )
            except Exception:
                st.error("Odds collection failed:")
                st.code(traceback.format_exc())


# ── page: Predictions Table ───────────────────────────────────────────────────

def page_predictions():
    st.header("Today's Predictions")
    _playoff_warning()

    sel_date = st.date_input("Prediction date", value=date.today())
    df = load_predictions(sel_date)

    if df.empty:
        st.info("No predictions found for this date. Run predictions first.")
        return

    # ── filters ──────────────────────────────────────────────────────────────
    with st.expander("Filters", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            markets = ["All"] + sorted(df["market"].dropna().unique().tolist())
            sel_market = st.selectbox("Market", markets)
            books = ["All"] + sorted(df["bookmaker"].dropna().unique().tolist())
            sel_book = st.selectbox("Bookmaker", books)
        with col2:
            min_conf = st.slider("Min confidence (%)", 50, 90, 55)
            min_ev = st.slider("Min EV (%)", -10, 30, 0)
        with col3:
            min_edge = st.slider("Min edge vs line", -5.0, 10.0, 0.0, 0.1)
            direction = st.selectbox("Bet direction", ["All", "OVER", "UNDER"])

        sort_col = st.selectbox(
            "Sort by",
            ["expected_value", "confidence", "edge_vs_line", "kelly_bet_size"],
        )

    filt = df.copy()
    if sel_market != "All":
        filt = filt[filt["market"] == sel_market]
    if sel_book != "All":
        filt = filt[filt["bookmaker"] == sel_book]
    if "confidence" in filt.columns:
        filt = filt[pd.to_numeric(filt["confidence"], errors="coerce").fillna(0) >= min_conf]
    if "expected_value" in filt.columns:
        filt = filt[pd.to_numeric(filt["expected_value"], errors="coerce").fillna(-999) >= min_ev]
    if "edge_vs_line" in filt.columns:
        filt = filt[pd.to_numeric(filt["edge_vs_line"], errors="coerce").fillna(-999) >= min_edge]
    if direction != "All" and "bet_direction" in filt.columns:
        filt = filt[filt["bet_direction"] == direction]

    if sort_col in filt.columns:
        filt = filt.sort_values(sort_col, ascending=False)

    display_cols = [c for c in [
        "player_name", "market", "line", "predicted_value", "edge_vs_line",
        "bet_direction", "confidence", "expected_value", "odds",
        "kelly_percent", "units", "rec_units", "kelly_bet_size",
        "bookmaker", "recommendation",
    ] if c in filt.columns]

    st.caption(f"{len(filt)} props shown")
    st.dataframe(filt[display_cols].reset_index(drop=True), use_container_width=True)

    csv = filt[display_cols].to_csv(index=False)
    st.download_button("Download filtered CSV", csv, "filtered_predictions.csv", "text/csv")


# ── page: Select Bets ─────────────────────────────────────────────────────────

def page_select_bets():
    st.header("Select Bets & Build Parlays")
    _playoff_warning()

    sel_date = st.date_input("Prediction date", value=date.today())
    df = load_predictions(sel_date)

    if df.empty:
        st.info("No predictions for this date. Run predictions first.")
        return

    # ── filters ───────────────────────────────────────────────────────────────
    with st.expander("Filters", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            only_pos_ev = st.checkbox("Positive EV only", value=True)
            markets = ["All"] + sorted(df["market"].dropna().unique().tolist())
            sel_market = st.selectbox("Market", markets, key="sb_market")
        with col2:
            min_conf = st.slider("Min confidence (%)", 50, 90, 55, key="sb_conf")
            direction = st.selectbox("Direction", ["All", "OVER", "UNDER"], key="sb_dir")
        with col3:
            books = ["All"] + sorted(df["bookmaker"].dropna().unique().tolist())
            sel_book = st.selectbox("Bookmaker", books, key="sb_book")
            sort_col = st.selectbox("Sort by", ["expected_value", "confidence", "edge_vs_line"], key="sb_sort")

    filt = df.copy()
    if only_pos_ev:
        filt = filt[pd.to_numeric(filt.get("expected_value", pd.Series(dtype=float)), errors="coerce") > 0]
    if sel_market != "All":
        filt = filt[filt["market"] == sel_market]
    if sel_book != "All":
        filt = filt[filt["bookmaker"] == sel_book]
    if "confidence" in filt.columns:
        filt = filt[pd.to_numeric(filt["confidence"], errors="coerce").fillna(0) >= min_conf]
    if direction != "All" and "bet_direction" in filt.columns:
        filt = filt[filt["bet_direction"] == direction]
    if sort_col in filt.columns:
        filt = filt.sort_values(sort_col, ascending=False)

    display_cols = [c for c in [
        "player_name", "market", "line", "predicted_value", "edge_vs_line",
        "bet_direction", "confidence", "expected_value", "odds",
        "rec_units", "kelly_bet_size", "bookmaker",
    ] if c in filt.columns]

    st.caption(f"{len(filt)} props shown — check rows to select, then add as singles or build a parlay below")

    filt_display = filt[display_cols].copy().reset_index(drop=True)
    filt_display.insert(0, "Select", False)

    edited = st.data_editor(
        filt_display,
        column_config={"Select": st.column_config.CheckboxColumn("Select", default=False)},
        disabled=[c for c in filt_display.columns if c != "Select"],
        use_container_width=True,
        hide_index=True,
        key="bet_selector",
    )

    selected_indices = edited[edited["Select"]].index.tolist()

    if not selected_indices:
        st.info("Check rows above to select bets.")
        return

    selected_rows = filt.iloc[selected_indices].reset_index(drop=True)

    st.divider()
    tab_single, tab_parlay = st.tabs([
        f"Add as {len(selected_indices)} Single Bet(s)",
        f"Build Parlay from {len(selected_indices)} Leg(s)",
    ])

    # ── tab: single bets ──────────────────────────────────────────────────────
    with tab_single:
        st.subheader(f"Configure {len(selected_indices)} bet(s)")
        bet_configs = []
        for i, row in selected_rows.iterrows():
            with st.expander(
                f"{row.get('player_name','?')} — {row.get('market','?')} "
                f"{row.get('bet_direction','?')} {row.get('line','?')}  "
                f"@ {row.get('bookmaker','?')}",
                expanded=True,
            ):
                c1, c2, c3 = st.columns(3)
                with c1:
                    kelly = float(row.get("kelly_bet_size", 0) or 0)
                    stake = st.number_input(
                        "Stake ($)", min_value=1.0, value=max(float(kelly), 5.0),
                        step=1.0, key=f"stake_{i}",
                    )
                with c2:
                    default_odds = float(row.get("odds", -110) or -110)
                    odds_taken = st.number_input(
                        "Odds taken", value=default_odds, step=1.0, key=f"odds_{i}",
                    )
                with c3:
                    notes = st.text_input("Notes", key=f"notes_{i}")
                bet_configs.append({"row": row, "stake": stake, "odds": odds_taken, "notes": notes})

        if st.button("Add Single Bets to Tracker", type="primary", key="add_singles"):
            tracker = load_tracker()
            today_str = sel_date.strftime("%Y-%m-%d")
            new_rows = []
            for bc in bet_configs:
                r = bc["row"]
                new_rows.append({
                    "date": today_str,
                    "bet_type": "model",
                    "player_name": r.get("player_name", ""),
                    "market": r.get("market", ""),
                    "line": r.get("line", ""),
                    "bet_direction": r.get("bet_direction", ""),
                    "odds": bc["odds"],
                    "stake": bc["stake"],
                    "bookmaker": r.get("bookmaker", ""),
                    "predicted_value": r.get("predicted_value", ""),
                    "confidence": r.get("confidence", ""),
                    "expected_value": r.get("expected_value", ""),
                    "result_status": "pending",
                    "actual_value": None,
                    "profit_loss": None,
                    "bankroll_after": None,
                    "notes": bc["notes"],
                })
            tracker = pd.concat([tracker, pd.DataFrame(new_rows)], ignore_index=True)
            save_tracker(tracker)
            st.success(f"Added {len(new_rows)} bet(s) to tracker.")
            st.rerun()

    # ── tab: parlay builder ───────────────────────────────────────────────────
    with tab_parlay:
        st.subheader("Parlay from Selected Predictions")

        legs = []
        for i, row in selected_rows.iterrows():
            direction = str(row.get("bet_direction", "OVER"))
            auto_odds = float(row.get("odds", -110) or -110)
            player = row.get("player_name", "?")
            market = row.get("market", "?")
            line = row.get("line", "?")
            book = row.get("bookmaker", "")

            c1, c2, c3 = st.columns([4, 2, 2])
            with c1:
                st.markdown(
                    f"**{player}** — {market} {direction} {line}"
                    + (f"  @ {book}" if book else "")
                )
            with c2:
                leg_odds = st.number_input(
                    "Odds", value=auto_odds, step=1.0,
                    key=f"parlay_odds_{i}", label_visibility="collapsed",
                )
            with c3:
                include = st.checkbox("Include", value=True, key=f"parlay_inc_{i}")

            if include:
                desc = f"{player} {market} {direction} {line}"
                legs.append({"desc": desc, "odds": int(leg_odds), "book": book, "row": row})

        if not legs:
            st.info("No legs included — check the Include boxes above.")
        else:
            combo_odds = parlay_combined_odds([l["odds"] for l in legs])
            all_books = ", ".join(set(l["book"] for l in legs if l["book"])) or "—"
            legs_str = " | ".join(f"{l['desc']} ({l['odds']:+d})" for l in legs)

            st.divider()
            m1, m2, m3 = st.columns(3)
            m1.metric("Legs", len(legs))
            m2.metric("Combined Odds", f"{combo_odds:+d}")

            p_stake = st.number_input(
                "Parlay stake ($)", min_value=1.0, value=10.0, step=1.0, key="parlay_stake"
            )
            to_win = calc_pnl(p_stake, float(combo_odds), True, False)
            m3.metric("To Win", f"${to_win:.2f}")

            p_notes = st.text_input("Notes", key="parlay_notes")
            st.caption(f"Legs: {legs_str}")

            if st.button("Add Parlay to Tracker", type="primary", key="add_parlay_from_preds"):
                tracker = load_tracker()
                new_row = {
                    "date": sel_date.strftime("%Y-%m-%d"),
                    "bet_type": "parlay",
                    "player_name": f"{len(legs)}-leg parlay",
                    "market": "parlay",
                    "line": None,
                    "bet_direction": "WIN",
                    "odds": float(combo_odds),
                    "stake": p_stake,
                    "bookmaker": all_books,
                    "predicted_value": None,
                    "confidence": round(float(np.mean([float(l["row"].get("confidence", 50) or 50) for l in legs])), 1),
                    "expected_value": None,
                    "result_status": "pending",
                    "actual_value": None,
                    "profit_loss": None,
                    "bankroll_after": None,
                    "parlay_legs": legs_str,
                    "notes": p_notes,
                }
                tracker = pd.concat([tracker, pd.DataFrame([new_row])], ignore_index=True)
                save_tracker(tracker)
                st.success(f"{len(legs)}-leg parlay added @ {combo_odds:+d} — to win ${to_win:.2f}")
                st.rerun()


# ── page: Manual Bet Entry ────────────────────────────────────────────────────

def page_manual_bets():
    st.markdown("## Manual Bet Entry")
    st.caption("Log parlays, outside-app bets, or any bet not generated by the model.")

    bankroll_data = load_bankroll()

    tab_single, tab_parlay = st.tabs(["Single Bet", "Parlay"])

    # ── single manual bet ─────────────────────────────────────────────────────
    with tab_single:
        st.markdown("### Single Bet")
        c1, c2 = st.columns(2)
        with c1:
            s_date = st.date_input("Date", value=date.today(), key="s_date")
            s_player = st.text_input("Player / Team / Description", key="s_player",
                                     placeholder="e.g. LeBron James or Lakers ML")
            s_market = st.text_input("Market / Bet Type", key="s_market",
                                     placeholder="e.g. player_points, moneyline, spread")
            s_line = st.number_input("Line (0 if N/A)", value=0.0, step=0.5, key="s_line")
        with c2:
            s_direction = st.selectbox("Direction", ["OVER", "UNDER", "WIN", "LOSE", "OTHER"], key="s_dir")
            s_odds = st.number_input("Odds (American)", value=-110, step=1, key="s_odds")
            s_stake = st.number_input("Stake ($)", min_value=1.0, value=10.0, step=1.0, key="s_stake")
            s_book = st.text_input("Bookmaker / App", key="s_book",
                                   placeholder="e.g. DraftKings, PrizePicks, Sleeper")
        s_notes = st.text_area("Notes", key="s_notes", height=60)

        # show implied prob
        try:
            dec = american_to_decimal(float(s_odds))
            implied = 1 / dec * 100
            to_win = calc_pnl(s_stake, float(s_odds), True, False)
            st.caption(f"Implied prob: {implied:.1f}%  ·  To win: ${to_win:.2f}  ·  Risk: ${s_stake:.2f}")
        except Exception:
            pass

        if st.button("Add Single Bet to Tracker", type="primary", key="add_single"):
            tracker = load_tracker()
            new_row = {
                "date": s_date.strftime("%Y-%m-%d"),
                "bet_type": "manual",
                "player_name": s_player,
                "market": s_market,
                "line": s_line if s_line != 0 else None,
                "bet_direction": s_direction,
                "odds": float(s_odds),
                "stake": s_stake,
                "bookmaker": s_book,
                "predicted_value": None, "confidence": None, "expected_value": None,
                "result_status": "pending",
                "actual_value": None, "profit_loss": None, "bankroll_after": None,
                "parlay_legs": None,
                "notes": s_notes,
            }
            tracker = pd.concat([tracker, pd.DataFrame([new_row])], ignore_index=True)
            save_tracker(tracker)
            st.success(f"Added: {s_player} — {s_market} {s_direction} @ {s_odds}")
            st.rerun()

    # ── parlay ────────────────────────────────────────────────────────────────
    with tab_parlay:
        st.markdown("### Parlay Builder")
        st.caption("Add each leg below, then enter your stake. Odds are calculated automatically.")

        if "parlay_legs" not in st.session_state:
            st.session_state["parlay_legs"] = []

        # add a leg
        with st.expander("Add a leg", expanded=True):
            lc1, lc2, lc3 = st.columns(3)
            with lc1:
                leg_desc = st.text_input("Leg description",
                                         placeholder="LeBron OVER 25.5 pts", key="leg_desc")
            with lc2:
                leg_odds = st.number_input("Leg odds (American)", value=-110, step=1, key="leg_odds")
            with lc3:
                leg_book = st.text_input("Bookmaker", placeholder="DraftKings", key="leg_book")

            if st.button("Add Leg"):
                if leg_desc:
                    st.session_state["parlay_legs"].append({
                        "desc": leg_desc, "odds": int(leg_odds), "book": leg_book,
                    })
                    st.rerun()

        if st.session_state["parlay_legs"]:
            st.markdown("**Current legs:**")
            legs = st.session_state["parlay_legs"]
            for i, leg in enumerate(legs):
                col_l, col_r = st.columns([5, 1])
                col_l.markdown(f"**{i+1}.** {leg['desc']}  —  `{leg['odds']:+d}`"
                               + (f"  @ {leg['book']}" if leg['book'] else ""))
                if col_r.button("✕", key=f"rm_leg_{i}"):
                    st.session_state["parlay_legs"].pop(i)
                    st.rerun()

            combo_odds = parlay_combined_odds([l["odds"] for l in legs])
            all_books = ", ".join(set(l["book"] for l in legs if l["book"])) or "—"
            legs_str = " | ".join(f"{l['desc']} ({l['odds']:+d})" for l in legs)

            st.divider()
            pc1, pc2 = st.columns(2)
            pc1.metric("Combined Parlay Odds", f"{combo_odds:+d}")
            p_date = st.date_input("Date", value=date.today(), key="p_date")
            p_stake = st.number_input("Stake ($)", min_value=1.0, value=10.0, step=1.0, key="p_stake")
            p_notes = st.text_area("Notes", key="p_notes", height=60)

            to_win = calc_pnl(p_stake, float(combo_odds), True, False)
            pc2.metric("To Win", f"${to_win:.2f}")
            st.caption(f"Bookmakers: {all_books}")

            bcol1, bcol2 = st.columns(2)
            if bcol1.button("Add Parlay to Tracker", type="primary", key="add_parlay"):
                tracker = load_tracker()
                new_row = {
                    "date": p_date.strftime("%Y-%m-%d"),
                    "bet_type": "parlay",
                    "player_name": f"{len(legs)}-leg parlay",
                    "market": "parlay",
                    "line": None,
                    "bet_direction": "WIN",
                    "odds": float(combo_odds),
                    "stake": p_stake,
                    "bookmaker": all_books,
                    "predicted_value": None, "confidence": None, "expected_value": None,
                    "result_status": "pending",
                    "actual_value": None, "profit_loss": None, "bankroll_after": None,
                    "parlay_legs": legs_str,
                    "notes": p_notes,
                }
                tracker = pd.concat([tracker, pd.DataFrame([new_row])], ignore_index=True)
                save_tracker(tracker)
                st.session_state["parlay_legs"] = []
                st.success(f"{len(legs)}-leg parlay added @ {combo_odds:+d}  (to win ${to_win:.2f})")
                st.rerun()

            if bcol2.button("Clear All Legs", key="clear_legs"):
                st.session_state["parlay_legs"] = []
                st.rerun()
        else:
            st.info("No legs added yet.")

    st.divider()
    st.markdown("### Recent Manual / Parlay Bets")
    tracker = load_tracker()
    manual = tracker[tracker.get("bet_type", pd.Series(dtype=str)).isin(["manual", "parlay"])] \
        if "bet_type" in tracker.columns else pd.DataFrame()
    if not manual.empty:
        cols = [c for c in ["date", "bet_type", "player_name", "market",
                             "odds", "stake", "bookmaker", "result_status",
                             "profit_loss", "parlay_legs", "notes"] if c in manual.columns]
        st.dataframe(manual[cols].sort_values("date", ascending=False).reset_index(drop=True),
                     use_container_width=True)
    else:
        st.caption("No manual bets yet.")


# ── page: Bankroll Settings ───────────────────────────────────────────────────

def page_bankroll():
    st.header("Bankroll Settings")

    data = load_bankroll()
    tracker = load_tracker()

    settled = tracker[tracker["result_status"].isin(["won", "lost", "push"])]
    total_pnl = pd.to_numeric(settled["profit_loss"], errors="coerce").sum()

    col1, col2, col3 = st.columns(3)
    col1.metric("Starting Bankroll", f"${data['starting']:,.2f}")
    col2.metric("Current Bankroll", f"${data['current']:,.2f}")
    col3.metric("Total P&L", f"${total_pnl:,.2f}", delta=f"${total_pnl:,.2f}")

    st.divider()
    st.subheader("Update Settings")

    new_starting = st.number_input(
        "Set starting bankroll ($)", min_value=1.0,
        value=float(data["starting"]), step=50.0,
    )
    new_current = st.number_input(
        "Override current bankroll ($) — use only to correct errors",
        min_value=0.0, value=float(data["current"]), step=10.0,
    )

    if st.button("Save Bankroll Settings"):
        save_bankroll({"starting": new_starting, "current": new_current})
        st.success("Bankroll saved.")
        st.rerun()


# ── page: Bet Tracker ─────────────────────────────────────────────────────────

def _hit_rate_table(df: pd.DataFrame) -> pd.DataFrame:
    """Build a per-market hit rate summary from settled prop bets."""
    prop_bets = df[df["result_status"].isin(["won", "lost", "push"])].copy()
    prop_bets = prop_bets[~prop_bets["market"].isin(["parlay", ""])]
    if prop_bets.empty:
        return pd.DataFrame()
    prop_bets["profit_loss_num"] = pd.to_numeric(prop_bets["profit_loss"], errors="coerce").fillna(0)
    prop_bets["stake_num"] = pd.to_numeric(prop_bets["stake"], errors="coerce").fillna(0)

    rows = []
    for market, g in prop_bets.groupby("market"):
        non_push = g[g["result_status"] != "push"]
        wins = (non_push["result_status"] == "won").sum()
        losses = (non_push["result_status"] == "lost").sum()
        pushes = (g["result_status"] == "push").sum()
        total = len(non_push)
        hit_rate = wins / total * 100 if total > 0 else 0
        staked = g["stake_num"].sum()
        pnl = g["profit_loss_num"].sum()
        roi = pnl / staked * 100 if staked > 0 else 0
        rows.append({
            "Market": market,
            "Bets": total,
            "W": wins, "L": losses, "P": pushes,
            "Hit Rate": f"{hit_rate:.1f}%",
            "Hit Rate Num": hit_rate,
            "P&L": f"${pnl:+.2f}",
            "ROI": f"{roi:+.1f}%",
        })
    return pd.DataFrame(rows).sort_values("Hit Rate Num", ascending=False)


def page_tracker():
    st.markdown("## Bet Tracker")

    tracker = load_tracker()
    if tracker.empty:
        st.info("No bets tracked yet.")
        return

    # ── hit rate summary ──────────────────────────────────────────────────────
    hit_df = _hit_rate_table(tracker)
    if not hit_df.empty:
        st.markdown("### Hit Rates by Market")
        display_cols = ["Market", "Bets", "W", "L", "P", "Hit Rate", "P&L", "ROI"]
        st.dataframe(
            hit_df[display_cols].reset_index(drop=True),
            use_container_width=True,
            hide_index=True,
        )

        # direction hit rate
        settled_props = tracker[
            tracker["result_status"].isin(["won", "lost"]) &
            ~tracker["market"].isin(["parlay", ""])
        ].copy()
        if not settled_props.empty and "bet_direction" in settled_props.columns:
            dir_rows = []
            for direction, g in settled_props.groupby("bet_direction"):
                wins = (g["result_status"] == "won").sum()
                total = len(g)
                dir_rows.append({
                    "Direction": direction,
                    "Bets": total,
                    "Wins": wins,
                    "Hit Rate": f"{wins/total*100:.1f}%" if total else "—",
                })
            if dir_rows:
                st.markdown("### Hit Rate: OVER vs UNDER")
                st.dataframe(pd.DataFrame(dir_rows), use_container_width=True, hide_index=True)

        st.divider()

    # ── full table with filters ───────────────────────────────────────────────
    st.markdown("### All Bets")
    with st.expander("Filters"):
        c1, c2, c3 = st.columns(3)
        with c1:
            statuses = ["All"] + sorted(tracker["result_status"].dropna().unique().tolist())
            sel_status = st.selectbox("Status", statuses)
        with c2:
            markets = ["All"] + sorted(tracker["market"].dropna().unique().tolist())
            sel_market = st.selectbox("Market", markets)
        with c3:
            if "bet_type" in tracker.columns:
                types = ["All"] + sorted(tracker["bet_type"].dropna().unique().tolist())
                sel_type = st.selectbox("Bet Type", types)
            else:
                sel_type = "All"

    filt = tracker.copy()
    if sel_status != "All":
        filt = filt[filt["result_status"] == sel_status]
    if sel_market != "All":
        filt = filt[filt["market"] == sel_market]
    if sel_type != "All" and "bet_type" in filt.columns:
        filt = filt[filt["bet_type"] == sel_type]

    st.dataframe(filt.sort_values("date", ascending=False).reset_index(drop=True),
                 use_container_width=True)
    csv = filt.to_csv(index=False)
    st.download_button("Download tracker CSV", csv, "bet_tracker.csv", "text/csv")


# ── page: Settle Bets ────────────────────────────────────────────────────────

def page_settle():
    st.markdown("## Settle Pending Bets")

    tracker = load_tracker()
    pending = tracker[tracker["result_status"] == "pending"].copy()

    if pending.empty:
        st.info("No pending bets.")
        return

    st.caption(f"{len(pending)} pending bet(s)")
    bankroll_data = load_bankroll()
    running_bankroll = bankroll_data["current"]

    updates = []
    for _, row in pending.iterrows():
        bet_type = str(row.get("bet_type", "model") or "model")
        is_prop = bet_type in ("model", "combo")
        player = row.get("player_name", "?")
        market = row.get("market", "?")
        direction = str(row.get("bet_direction", "OVER"))
        stake = row.get("stake", "?")
        label = (
            f"{row.get('date','?')}  ·  "
            + (f"[{bet_type.upper()}] " if bet_type not in ("model",) else "")
            + f"{player}  {market}  {direction}"
            + (f"  line {row.get('line','?')}" if is_prop else "")
            + f"  · stake ${stake}"
        )
        with st.expander(label):
            if bet_type == "parlay":
                legs_text = row.get("parlay_legs", "")
                if legs_text:
                    st.caption(f"Legs: {legs_text}")
                result_sel = st.selectbox(
                    "Result", ["pending", "won", "lost", "push"],
                    key=f"settle_result_{row.name}",
                )
                updates.append({
                    "index": row.name, "row": row,
                    "mode": "direct", "result": result_sel, "actual": None,
                })
            elif not is_prop:
                # Manual single bet — direct won/lost/push
                result_sel = st.selectbox(
                    "Result", ["pending", "won", "lost", "push"],
                    key=f"settle_result_{row.name}",
                )
                updates.append({
                    "index": row.name, "row": row,
                    "mode": "direct", "result": result_sel, "actual": None,
                })
            else:
                # Model/combo prop — enter actual stat value
                actual_val = st.number_input(
                    "Actual stat value", value=0.0, step=0.5,
                    key=f"settle_{row.name}",
                )
                updates.append({
                    "index": row.name, "row": row,
                    "mode": "stat", "actual": actual_val, "result": None,
                })

    if st.button("Settle All Above", type="primary"):
        for u in updates:
            idx = u["index"]
            row = u["row"]
            stake = float(row.get("stake", 0) or 0)
            odds = float(row.get("odds", -110) or -110)

            if u["mode"] == "direct":
                status = u["result"]
                if status == "pending":
                    continue
                won = status == "won"
                push = status == "push"
                actual = None
            else:
                actual = u["actual"]
                line = float(row.get("line", 0) or 0)
                direction = str(row.get("bet_direction", "OVER"))
                push = actual == line
                won = (actual > line) if direction == "OVER" else (actual < line)
                status = "push" if push else ("won" if won else "lost")

            pnl = calc_pnl(stake, odds, won, push)
            running_bankroll += pnl

            tracker.at[idx, "actual_value"] = actual
            tracker.at[idx, "result_status"] = status
            tracker.at[idx, "profit_loss"] = pnl
            tracker.at[idx, "bankroll_after"] = round(running_bankroll, 2)

        save_tracker(tracker)
        bankroll_data["current"] = round(running_bankroll, 2)
        save_bankroll(bankroll_data)
        st.success("Bets settled and bankroll updated.")
        st.rerun()


# ── page: Dashboard Summary ───────────────────────────────────────────────────

def page_dashboard():
    st.markdown("## Dashboard")

    bankroll_data = load_bankroll()
    tracker = load_tracker()

    settled = tracker[tracker["result_status"].isin(["won", "lost", "push"])].copy()
    pending = tracker[tracker["result_status"] == "pending"]

    pnl_series = pd.to_numeric(settled["profit_loss"], errors="coerce").fillna(0)
    total_pnl = pnl_series.sum()
    starting = bankroll_data["starting"]
    current = bankroll_data["current"]
    roi = (total_pnl / (starting + 1e-9)) * 100

    wins = (settled["result_status"] == "won").sum()
    losses = (settled["result_status"] == "lost").sum()
    pushes = (settled["result_status"] == "push").sum()
    total_settled = len(settled)
    win_rate = (wins / total_settled * 100) if total_settled else 0

    avg_conf = pd.to_numeric(tracker["confidence"], errors="coerce").mean() if not tracker.empty else 0
    avg_ev = pd.to_numeric(tracker["expected_value"], errors="coerce").mean() if not tracker.empty else 0
    total_staked = pd.to_numeric(settled["stake"], errors="coerce").sum() if not settled.empty else 0

    # ── KPI row 1 ─────────────────────────────────────────────────────────────
    c = st.columns(4)
    kpi(c[0], "Current Bankroll", f"${current:,.2f}",
        f"{'▲' if total_pnl >= 0 else '▼'} ${abs(total_pnl):,.2f} all-time",
        positive=total_pnl >= 0)
    kpi(c[1], "ROI", f"{roi:.1f}%",
        "on starting bankroll", positive=roi >= 0)
    kpi(c[2], "Win Rate", f"{win_rate:.1f}%",
        f"{wins}W  {losses}L  {pushes}P", positive=win_rate >= 52.4)
    kpi(c[3], "Pending Bets", str(len(pending)),
        "awaiting results", positive=None)

    # ── KPI row 2 ─────────────────────────────────────────────────────────────
    c2 = st.columns(4)
    kpi(c2[0], "Total Settled", str(total_settled), "bets", positive=None)
    kpi(c2[1], "Total Staked", f"${total_staked:,.0f}", "", positive=None)
    kpi(c2[2], "Avg Confidence", f"{avg_conf:.1f}%", "", positive=avg_conf >= 55)
    kpi(c2[3], "Avg EV", f"{avg_ev:.1f}%", "", positive=avg_ev >= 0)

    if settled.empty:
        st.info("No settled bets yet — results will appear here once you settle bets.")
        return

    st.divider()

    settled["profit_loss_num"] = pnl_series

    # ── charts row ────────────────────────────────────────────────────────────
    left, right = st.columns(2)

    with left:
        mkt_pnl = settled.groupby("market")["profit_loss_num"].sum().reset_index()
        mkt_pnl.columns = ["market", "profit_loss"]
        mkt_pnl["color"] = mkt_pnl["profit_loss"].apply(lambda x: "#2ecc71" if x >= 0 else "#e74c3c")
        fig_mkt = go.Figure(go.Bar(
            x=mkt_pnl["market"], y=mkt_pnl["profit_loss"],
            marker_color=mkt_pnl["color"],
            text=mkt_pnl["profit_loss"].apply(lambda x: f"${x:+.0f}"),
            textposition="outside",
        ))
        fig_mkt.update_layout(
            title="P&L by Market", plot_bgcolor="#1a1f2e", paper_bgcolor="#1a1f2e",
            font_color="#e0e0e0", title_font_color="#ffc72c",
            xaxis=dict(tickangle=-30, gridcolor="#2a2f3e"),
            yaxis=dict(gridcolor="#2a2f3e"),
            margin=dict(t=50, b=80),
        )
        st.plotly_chart(fig_mkt, use_container_width=True)

    with right:
        book_pnl = settled.groupby("bookmaker")["profit_loss_num"].sum().reset_index()
        book_pnl.columns = ["bookmaker", "profit_loss"]
        book_pnl["color"] = book_pnl["profit_loss"].apply(lambda x: "#2ecc71" if x >= 0 else "#e74c3c")
        fig_book = go.Figure(go.Bar(
            x=book_pnl["bookmaker"], y=book_pnl["profit_loss"],
            marker_color=book_pnl["color"],
            text=book_pnl["profit_loss"].apply(lambda x: f"${x:+.0f}"),
            textposition="outside",
        ))
        fig_book.update_layout(
            title="P&L by Bookmaker", plot_bgcolor="#1a1f2e", paper_bgcolor="#1a1f2e",
            font_color="#e0e0e0", title_font_color="#ffc72c",
            xaxis=dict(gridcolor="#2a2f3e"),
            yaxis=dict(gridcolor="#2a2f3e"),
            margin=dict(t=50, b=40),
        )
        st.plotly_chart(fig_book, use_container_width=True)

    # ── recent bets table ─────────────────────────────────────────────────────
    st.markdown("### Recent Results")
    recent = settled.sort_values("date", ascending=False).head(10)

    def status_badge(s):
        m = {"won": "badge-green", "lost": "badge-red", "push": "badge-gold"}
        return f'<span class="badge {m.get(s, "badge-grey")}">{s.upper()}</span>'

    recent_display = recent[[c for c in [
        "date", "bet_type", "player_name", "market", "line",
        "bet_direction", "odds", "stake", "profit_loss", "result_status",
    ] if c in recent.columns]].copy()
    st.dataframe(recent_display.reset_index(drop=True), use_container_width=True)


# ── page: Charts ─────────────────────────────────────────────────────────────

_CHART_LAYOUT = dict(
    plot_bgcolor="#1a1f2e", paper_bgcolor="#1a1f2e",
    font_color="#e0e0e0", title_font_color="#ffc72c",
    xaxis=dict(gridcolor="#2a2f3e"), yaxis=dict(gridcolor="#2a2f3e"),
    margin=dict(t=50, b=60, l=50, r=20),
)


def page_charts():
    st.markdown("## Performance Charts")

    tracker = load_tracker()
    bankroll_data = load_bankroll()
    settled = tracker[tracker["result_status"].isin(["won", "lost", "push"])].copy()

    if settled.empty:
        st.info("No settled bets to chart yet.")
        return

    settled["date"] = pd.to_datetime(settled["date"], errors="coerce")
    settled["profit_loss_num"] = pd.to_numeric(settled["profit_loss"], errors="coerce").fillna(0)
    settled["bankroll_after_num"] = pd.to_numeric(settled["bankroll_after"], errors="coerce")
    settled["stake_num"] = pd.to_numeric(settled["stake"], errors="coerce").fillna(0)

    # ── bankroll over time ────────────────────────────────────────────────────
    bankroll_ts = settled.dropna(subset=["bankroll_after_num"]).sort_values("date")
    if not bankroll_ts.empty:
        fig_br = go.Figure()
        fig_br.add_trace(go.Scatter(
            x=bankroll_ts["date"], y=bankroll_ts["bankroll_after_num"],
            mode="lines+markers", name="Bankroll",
            line=dict(color="#2ecc71", width=2),
            fill="tozeroy", fillcolor="rgba(46,204,113,0.08)",
        ))
        fig_br.add_hline(y=bankroll_data["starting"], line_dash="dash",
                         line_color="#ffc72c", annotation_text="Starting",
                         annotation_font_color="#ffc72c")
        fig_br.update_layout(title="Bankroll Over Time",
                              xaxis_title="Date", yaxis_title="$", **_CHART_LAYOUT)
        st.plotly_chart(fig_br, use_container_width=True)

    # ── cumulative P&L ────────────────────────────────────────────────────────
    settled_sorted = settled.sort_values("date").copy()
    settled_sorted["cumulative_pnl"] = settled_sorted["profit_loss_num"].cumsum()
    fig_cum = go.Figure()
    fig_cum.add_trace(go.Scatter(
        x=settled_sorted["date"], y=settled_sorted["cumulative_pnl"],
        mode="lines+markers", name="Cumulative P&L",
        line=dict(color="#3498db", width=2),
        fill="tozeroy", fillcolor="rgba(52,152,219,0.12)",
    ))
    fig_cum.add_hline(y=0, line_dash="dash", line_color="#8892a4")
    fig_cum.update_layout(title="Cumulative P&L", xaxis_title="Date",
                          yaxis_title="$", **_CHART_LAYOUT)
    st.plotly_chart(fig_cum, use_container_width=True)

    st.divider()

    # ── hit rate charts (props only) ──────────────────────────────────────────
    prop_settled = settled[~settled["market"].isin(["parlay", ""])].copy()
    if not prop_settled.empty:
        mkt_stats = prop_settled.groupby("market").apply(
            lambda g: pd.Series({
                "wins":   (g["result_status"] == "won").sum(),
                "bets":   len(g[g["result_status"] != "push"]),
                "pnl":    g["profit_loss_num"].sum(),
                "staked": g["stake_num"].sum(),
            })
        ).reset_index()
        mkt_stats["hit_rate"] = mkt_stats.apply(
            lambda r: r["wins"] / r["bets"] * 100 if r["bets"] > 0 else 0, axis=1
        )
        mkt_stats["roi"] = mkt_stats.apply(
            lambda r: r["pnl"] / r["staked"] * 100 if r["staked"] > 0 else 0, axis=1
        )

        left, right = st.columns(2)
        with left:
            bar_colors = mkt_stats["hit_rate"].apply(
                lambda x: "#2ecc71" if x >= 52.4 else "#e74c3c"
            )
            fig_hr = go.Figure(go.Bar(
                x=mkt_stats["market"], y=mkt_stats["hit_rate"],
                marker_color=bar_colors,
                text=mkt_stats["hit_rate"].apply(lambda x: f"{x:.1f}%"),
                textposition="outside",
            ))
            fig_hr.add_hline(y=52.4, line_dash="dash", line_color="#ffc72c",
                             annotation_text="Break-even (~52.4%)",
                             annotation_font_color="#ffc72c")
            fig_hr.update_layout(title="Hit Rate by Market",
                                 yaxis_title="%", yaxis_range=[0, 100], **_CHART_LAYOUT)
            st.plotly_chart(fig_hr, use_container_width=True)

        with right:
            roi_colors = mkt_stats["roi"].apply(lambda x: "#2ecc71" if x >= 0 else "#e74c3c")
            fig_roi_mkt = go.Figure(go.Bar(
                x=mkt_stats["market"], y=mkt_stats["roi"],
                marker_color=roi_colors,
                text=mkt_stats["roi"].apply(lambda x: f"{x:+.1f}%"),
                textposition="outside",
            ))
            fig_roi_mkt.add_hline(y=0, line_dash="dash", line_color="#8892a4")
            fig_roi_mkt.update_layout(title="ROI by Market", yaxis_title="%", **_CHART_LAYOUT)
            st.plotly_chart(fig_roi_mkt, use_container_width=True)

        # ── OVER vs UNDER hit rate ────────────────────────────────────────────
        if "bet_direction" in prop_settled.columns:
            dir_stats = prop_settled[prop_settled["result_status"] != "push"].groupby(
                ["market", "bet_direction"]
            ).apply(
                lambda g: pd.Series({"wins": (g["result_status"] == "won").sum(), "bets": len(g)})
            ).reset_index()
            dir_stats["hit_rate"] = dir_stats.apply(
                lambda r: r["wins"] / r["bets"] * 100 if r["bets"] > 0 else 0, axis=1
            )
            fig_dir = px.bar(
                dir_stats, x="market", y="hit_rate", color="bet_direction",
                barmode="group",
                color_discrete_map={"OVER": "#3498db", "UNDER": "#e67e22"},
                text=dir_stats["hit_rate"].apply(lambda x: f"{x:.0f}%"),
                title="Hit Rate: OVER vs UNDER by Market",
            )
            fig_dir.add_hline(y=52.4, line_dash="dash", line_color="#ffc72c",
                              annotation_text="Break-even", annotation_font_color="#ffc72c")
            fig_dir.update_traces(textposition="outside")
            fig_dir.update_layout(yaxis_range=[0, 100], yaxis_title="%", **_CHART_LAYOUT)
            st.plotly_chart(fig_dir, use_container_width=True)

        # ── rolling hit rate ──────────────────────────────────────────────────
        prop_sorted = prop_settled[prop_settled["result_status"] != "push"].sort_values("date").copy()
        if len(prop_sorted) >= 5:
            prop_sorted["won_bin"] = (prop_sorted["result_status"] == "won").astype(int)
            window = min(20, len(prop_sorted))
            prop_sorted["rolling_hr"] = (
                prop_sorted["won_bin"].rolling(window, min_periods=3).mean() * 100
            )
            fig_roll = go.Figure()
            fig_roll.add_trace(go.Scatter(
                x=prop_sorted["date"], y=prop_sorted["rolling_hr"],
                mode="lines", name="Rolling Hit Rate",
                line=dict(color="#9b59b6", width=2),
                fill="tozeroy", fillcolor="rgba(155,89,182,0.1)",
            ))
            fig_roll.add_hline(y=52.4, line_dash="dash", line_color="#ffc72c",
                               annotation_text="Break-even", annotation_font_color="#ffc72c")
            fig_roll.update_layout(
                title=f"Rolling Hit Rate (last {window} bets)",
                xaxis_title="Date", yaxis_title="%", yaxis_range=[0, 100], **_CHART_LAYOUT,
            )
            st.plotly_chart(fig_roll, use_container_width=True)

    st.divider()

    # ── ROI by bookmaker ──────────────────────────────────────────────────────
    book_grp = settled.groupby("bookmaker").agg(
        total_staked=("stake_num", "sum"),
        total_pnl=("profit_loss_num", "sum"),
    ).reset_index()
    book_grp["roi"] = book_grp.apply(
        lambda r: r["total_pnl"] / r["total_staked"] * 100 if r["total_staked"] > 0 else 0,
        axis=1,
    )
    roi_bar_colors = book_grp["roi"].apply(lambda x: "#2ecc71" if x >= 0 else "#e74c3c")
    fig_roi = go.Figure(go.Bar(
        x=book_grp["bookmaker"], y=book_grp["roi"],
        marker_color=roi_bar_colors,
        text=book_grp["roi"].apply(lambda x: f"{x:+.1f}%"),
        textposition="outside",
    ))
    fig_roi.add_hline(y=0, line_dash="dash", line_color="#8892a4")
    fig_roi.update_layout(title="ROI by Bookmaker", yaxis_title="%", **_CHART_LAYOUT)
    st.plotly_chart(fig_roi, use_container_width=True)


# ── page: Historical Odds ─────────────────────────────────────────────────────

def page_odds_collection():
    st.header("Historical Odds Collection")

    PLAYOFF_ONGOING_FILE = DATA_DIR / "ongoing_odds_playoffs.csv"

    # ── stats row ─────────────────────────────────────────────────────────────
    hist_file = DATA_DIR / "historical_odds_combined.csv"

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Historical (Regular Season)")
        if hist_file.exists():
            @st.cache_data(ttl=300)
            def _load_hist_stats():
                df = pd.read_csv(hist_file, usecols=["timestamp", "player_name", "market", "bookmaker"])
                return {
                    "total": len(df),
                    "latest": df["timestamp"].max(),
                    "markets": df["market"].value_counts().to_dict(),
                    "books": df["bookmaker"].value_counts().to_dict(),
                }
            stats = _load_hist_stats()
            st.metric("Total Props", f"{stats['total']:,}")
            st.metric("Latest Entry", stats["latest"])
            st.caption("Markets:")
            st.json(stats["markets"])
        else:
            st.info("historical_odds_combined.csv not found in data/")

    with col2:
        st.subheader("Ongoing (Regular Season)")
        ongoing = load_ongoing_odds()
        if not ongoing.empty:
            today_str = date.today().strftime("%Y-%m-%d")
            if "collection_date" in ongoing.columns:
                today_rows = ongoing[ongoing["collection_date"].astype(str).str.startswith(today_str)]
                st.metric("Total Rows", f"{len(ongoing):,}")
                st.metric("Collected Today", f"{len(today_rows):,}")
                st.metric("Latest Collection", ongoing["collection_date"].max())
            else:
                st.metric("Total Rows", f"{len(ongoing):,}")
        else:
            st.info("No regular season collection yet.")

    with col3:
        st.subheader("Ongoing (Playoffs)")
        if PLAYOFF_ONGOING_FILE.exists():
            po_df = pd.read_csv(PLAYOFF_ONGOING_FILE)
            today_str = date.today().strftime("%Y-%m-%d")
            if "collection_date" in po_df.columns:
                today_po = po_df[po_df["collection_date"].astype(str).str.startswith(today_str)]
                st.metric("Total Rows", f"{len(po_df):,}")
                st.metric("Collected Today", f"{len(today_po):,}")
                st.metric("Latest Collection", po_df["collection_date"].max())
            else:
                st.metric("Total Rows", f"{len(po_df):,}")
            if st.button("Remove Duplicates (Playoffs)"):
                cleaned = dedup_odds(po_df)
                cleaned.to_csv(PLAYOFF_ONGOING_FILE, index=False)
                st.success(f"Removed {len(po_df) - len(cleaned):,} duplicates.")
                st.rerun()
        else:
            st.info("No playoff odds collected yet. Playoff mode auto-saves here during the postseason.")

    st.divider()

    # ── regular season dedup ─────────────────────────────────────────────────
    if not ongoing.empty:
        before = len(ongoing)
        if st.button("Remove Exact Duplicates from Ongoing Collection"):
            cleaned = dedup_odds(ongoing)
            cleaned.to_csv(ONGOING_FILE, index=False)
            removed = before - len(cleaned)
            st.success(f"Removed {removed:,} exact duplicate rows. {len(cleaned):,} rows remain.")
            st.rerun()

    # ── Historical Odds Explorer ─────────────────────────────────────────────
    st.subheader("Historical Odds Explorer")
    st.caption("Explore line movement and sportsbook comparisons. Loads from historical_odds_combined.csv.")

    if not hist_file.exists():
        st.info("No historical odds file found.")
        return

    with st.form("odds_explorer"):
        c1, c2, c3 = st.columns(3)
        with c1:
            player_query = st.text_input("Player name (partial OK)", "")
        with c2:
            market_opts = [
                "player_points", "player_rebounds", "player_assists",
                "player_threes", "player_steals", "player_blocks",
                "player_points_rebounds_assists", "player_points_rebounds",
                "player_points_assists", "player_rebounds_assists",
            ]
            sel_market_exp = st.selectbox("Market", market_opts)
        with c3:
            book_opts = ["All", "betmgm", "draftkings", "fanduel", "bovada",
                         "betrivers", "fanatics", "williamhill_us", "betonlineag"]
            sel_book_exp = st.selectbox("Bookmaker", book_opts)
        submitted = st.form_submit_button("Load")

    if submitted and player_query:
        with st.spinner("Loading…"):

            @st.cache_data(ttl=120, show_spinner=False)
            def _load_player_odds(player: str, market: str, book: str):
                df = pd.read_csv(hist_file)
                df = df[df["player_name"].str.contains(player, case=False, na=False)]
                df = df[df["market"] == market]
                if book != "All":
                    df = df[df["bookmaker"] == book]
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
                df = df.sort_values("timestamp")
                return df

            exp_df = _load_player_odds(player_query, sel_market_exp, sel_book_exp)

        if exp_df.empty:
            st.warning("No data found for that player / market / book combination.")
        else:
            st.caption(f"{len(exp_df):,} rows found for '{player_query}'")

            # line movement chart — over lines per bookmaker
            over_df = exp_df[exp_df["bet_type"].str.lower() == "over"] if "bet_type" in exp_df.columns else exp_df
            if not over_df.empty:
                fig_line = px.line(
                    over_df, x="timestamp", y="line",
                    color="bookmaker" if "bookmaker" in over_df.columns else None,
                    title=f"Line Movement — {player_query} ({sel_market_exp})",
                    markers=True,
                )
                st.plotly_chart(fig_line, use_container_width=True)

                # odds comparison
                fig_odds = px.line(
                    over_df, x="timestamp", y="odds",
                    color="bookmaker" if "bookmaker" in over_df.columns else None,
                    title=f"Over Odds Movement — {player_query} ({sel_market_exp})",
                    markers=True,
                )
                st.plotly_chart(fig_odds, use_container_width=True)

            st.dataframe(exp_df.tail(100).reset_index(drop=True), use_container_width=True)


# ── page: Combo Props ─────────────────────────────────────────────────────────

def page_combo_props():
    st.header("Combo Prop Predictions")
    st.caption(
        "Combines individual market predictions (pts/reb/ast) to evaluate "
        "PRA, PR, PA, RA combo props from today's odds."
    )

    sel_date = st.date_input("Prediction date", value=date.today(), key="combo_date")
    individual_df = load_predictions(sel_date)

    if individual_df.empty:
        st.warning("No individual predictions found for this date. Run Daily Predictions first.")
        return

    bankroll_data = load_bankroll()
    bankroll = bankroll_data["current"]

    if st.button("Generate Combo Predictions", type="primary"):
        with st.spinner("Fetching combo props and predicting…"):
            try:
                from scripts.live_odds_scraper import LiveOddsScraper
                from scripts.combo_prop_predictor import ComboPropPredictor

                scraper = LiveOddsScraper()
                all_props = scraper.get_todays_props()

                if all_props.empty:
                    st.warning("No props fetched from The Odds API.")
                    return

                combo_markets = [
                    "player_points_rebounds_assists",
                    "player_points_rebounds",
                    "player_points_assists",
                    "player_rebounds_assists",
                ]
                combo_props = all_props[all_props["market"].isin(combo_markets)].copy()

                if combo_props.empty:
                    st.warning("No combo props available in today's odds.")
                    return

                st.info(f"Found {len(combo_props)} combo props across {combo_props['market'].nunique()} markets.")

                predictor = ComboPropPredictor()
                combo_preds = predictor.predict_combo_props(individual_df, combo_props, bankroll=bankroll)

                if combo_preds.empty:
                    st.warning("No combo predictions generated — individual predictions may not cover all components.")
                    return

                st.session_state["combo_preds"] = combo_preds
                st.success(f"Generated {len(combo_preds)} combo predictions.")

            except Exception:
                st.error("Combo prediction failed:")
                st.code(traceback.format_exc())

    combo_preds = st.session_state.get("combo_preds", pd.DataFrame())
    if combo_preds.empty:
        return

    # ── filters ──────────────────────────────────────────────────────────────
    with st.expander("Filters", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            markets = ["All"] + sorted(combo_preds["market"].dropna().unique().tolist())
            sel_mkt = st.selectbox("Market", markets, key="combo_mkt_filter")
        with c2:
            min_conf = st.slider("Min confidence (%)", 40, 80, 50, key="combo_conf")
        with c3:
            only_positive = st.checkbox("Positive EV only", value=True, key="combo_ev")

    filt = combo_preds.copy()
    if sel_mkt != "All":
        filt = filt[filt["market"] == sel_mkt]
    filt = filt[pd.to_numeric(filt["confidence"], errors="coerce").fillna(0) >= min_conf]
    if only_positive:
        filt = filt[pd.to_numeric(filt.get("expected_value_pct", pd.Series(dtype=float)), errors="coerce").fillna(-999) > 0]

    display_cols = [c for c in [
        "player_name", "market", "line", "predicted_value", "edge_vs_line",
        "bet_direction", "confidence", "win_probability", "expected_value_pct",
        "recommended_bet", "recommendation", "bookmaker",
        "points_pred", "rebounds_pred", "assists_pred",
    ] if c in filt.columns]

    filt_display = filt[display_cols].sort_values(
        "expected_value_pct" if "expected_value_pct" in filt.columns else display_cols[0],
        ascending=False,
    ).reset_index(drop=True)

    st.caption(f"{len(filt_display)} combo predictions shown")
    st.dataframe(filt_display, use_container_width=True)

    # ── add to tracker ────────────────────────────────────────────────────────
    st.subheader("Add Combo Bets to Tracker")
    filt_display.insert(0, "Select", False)
    edited = st.data_editor(
        filt_display,
        column_config={"Select": st.column_config.CheckboxColumn("Select", default=False)},
        disabled=[c for c in filt_display.columns if c != "Select"],
        use_container_width=True,
        hide_index=True,
        key="combo_selector",
    )
    selected = edited[edited["Select"]].index.tolist()

    if selected and st.button("Add Selected Combo Bets to Tracker"):
        tracker = load_tracker()
        new_rows = []
        for idx in selected:
            r = filt.iloc[idx]
            is_over = str(r.get("bet_direction", "")) == "OVER"
            odds = r.get("odds_over" if is_over else "odds_under", -110)
            new_rows.append({
                "date": sel_date.strftime("%Y-%m-%d"),
                "player_name": r.get("player_name", ""),
                "market": r.get("market", ""),
                "line": r.get("line", ""),
                "bet_direction": r.get("bet_direction", ""),
                "odds": odds,
                "stake": float(r.get("recommended_bet", 10) or 10),
                "bookmaker": r.get("bookmaker", ""),
                "predicted_value": r.get("predicted_value", ""),
                "confidence": r.get("confidence", ""),
                "expected_value": r.get("expected_value_pct", ""),
                "result_status": "pending",
                "actual_value": None,
                "profit_loss": None,
                "bankroll_after": None,
                "notes": "combo",
            })
        tracker = pd.concat([tracker, pd.DataFrame(new_rows)], ignore_index=True)
        save_tracker(tracker)
        st.success(f"Added {len(new_rows)} combo bet(s) to tracker.")
        st.rerun()


# ── page: Arbitrage & Line Shopping ──────────────────────────────────────────

def page_arbitrage():
    st.header("Arbitrage & Line Shopping")
    st.caption(
        "Loads today's props across all bookmakers and finds arbitrage opportunities "
        "or best odds per side."
    )

    if st.button("Load Today's Odds", type="primary"):
        with st.spinner("Fetching odds from all bookmakers…"):
            try:
                from scripts.live_odds_scraper import LiveOddsScraper
                scraper = LiveOddsScraper()
                props = scraper.get_todays_props()
                if props.empty:
                    st.warning("No props returned.")
                else:
                    st.session_state["arb_props"] = props
                    st.success(f"Loaded {len(props)} props across {props['bookmaker'].nunique()} bookmakers.")
            except Exception:
                st.error("Failed to fetch odds:")
                st.code(traceback.format_exc())

    props = st.session_state.get("arb_props", pd.DataFrame())
    if props.empty:
        return

    from scripts.arbitrage_detector import ArbitrageDetector
    detector = ArbitrageDetector()
    best_lines = detector.find_best_lines(props)

    if best_lines.empty:
        st.info("No props with multiple bookmakers found (need 2+ books per prop).")
        return

    tab1, tab2 = st.tabs(["Arbitrage Opportunities", "Best Lines / Line Shopping"])

    # ── tab 1: arbitrage ─────────────────────────────────────────────────────
    with tab1:
        arb = best_lines[best_lines["has_arbitrage"]].copy()
        if arb.empty:
            st.info("No arbitrage opportunities right now.")
        else:
            st.success(f"{len(arb)} arbitrage opportunity/ies found!")
            for _, row in arb.iterrows():
                with st.expander(
                    f"{row['player_name']} — {row['market']}  line {row['line']}"
                    f"  |  profit {row['arb_profit_pct']:.2f}%",
                    expanded=True,
                ):
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Guaranteed Profit", f"{row['arb_profit_pct']:.2f}%")
                    c2.metric(
                        f"Best OVER — {row['best_over_book']}",
                        f"{row['best_over_odds']:+.0f}",
                    )
                    c3.metric(
                        f"Best UNDER — {row['best_under_book']}",
                        f"{row['best_under_odds']:+.0f}",
                    )

                    bankroll = load_bankroll()["current"]
                    over_stake = bankroll * (row.get("over_stake_pct", 50) / 100)
                    under_stake = bankroll * (row.get("under_stake_pct", 50) / 100)

                    st.caption(
                        f"Suggested stakes on ${bankroll:,.0f} bankroll:  "
                        f"OVER ${over_stake:.0f} @ {row['best_over_book']}  |  "
                        f"UNDER ${under_stake:.0f} @ {row['best_under_book']}"
                    )

    # ── tab 2: line shopping ─────────────────────────────────────────────────
    with tab2:
        st.subheader("Best Odds Per Side")

        with st.expander("Filters"):
            c1, c2 = st.columns(2)
            with c1:
                markets = ["All"] + sorted(best_lines["market"].dropna().unique().tolist())
                sel_mkt = st.selectbox("Market", markets, key="arb_mkt")
                min_books = st.slider("Min bookmakers", 2, 6, 2, key="arb_books")
            with c2:
                min_value = st.slider("Min odds advantage vs avg (units)", 0, 30, 5, key="arb_value")
                show_all = st.checkbox("Show all props (not just value)", value=False)

        filt = best_lines.copy()
        if sel_mkt != "All":
            filt = filt[filt["market"] == sel_mkt]
        filt = filt[filt["num_books"] >= min_books]
        if not show_all:
            filt = filt[
                (filt["over_value"].abs() >= min_value) |
                (filt["under_value"].abs() >= min_value)
            ]

        display_cols = [c for c in [
            "player_name", "market", "line", "num_books",
            "best_over_book", "best_over_odds", "over_value",
            "best_under_book", "best_under_odds", "under_value",
            "vig",
        ] if c in filt.columns]

        st.caption(f"{len(filt)} props shown")
        st.dataframe(
            filt[display_cols].reset_index(drop=True),
            use_container_width=True,
        )

        # ── quick per-prop odds comparison ───────────────────────────────────
        st.subheader("Odds Comparison — Single Prop")
        if not best_lines.empty:
            players = sorted(best_lines["player_name"].dropna().unique().tolist())
            sel_player = st.selectbox("Player", players, key="arb_player")
            player_props = best_lines[best_lines["player_name"] == sel_player]
            if not player_props.empty:
                sel_prop_key = st.selectbox(
                    "Market / line",
                    player_props.apply(lambda r: f"{r['market']} {r['line']}", axis=1).tolist(),
                    key="arb_prop",
                )
                sel_row = player_props.iloc[
                    player_props.apply(lambda r: f"{r['market']} {r['line']}", axis=1)
                    .tolist().index(sel_prop_key)
                ]
                all_books = sel_row.get("all_books", [])
                if all_books:
                    books_df = pd.DataFrame(all_books)
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=books_df["bookmaker"], y=books_df["odds_over"],
                        name="Over", marker_color="#2ca02c",
                    ))
                    fig.add_trace(go.Bar(
                        x=books_df["bookmaker"], y=books_df["odds_under"],
                        name="Under", marker_color="#d62728",
                    ))
                    fig.update_layout(
                        title=f"{sel_player} — {sel_prop_key} odds by bookmaker",
                        barmode="group",
                        xaxis_title="Bookmaker",
                        yaxis_title="American Odds",
                    )
                    st.plotly_chart(fig, use_container_width=True)


# ── page: Maintenance ─────────────────────────────────────────────────────────

def page_maintenance():
    st.header("Maintenance")

    # ── daily predictor fresh data note ──────────────────────────────────────
    st.subheader("Game Logs")
    st.info(
        "The **daily predictor** fetches fresh per-player game logs automatically "
        "each time you run predictions (cached once per day). "
        "The **game_results CSV files** (used for training data) need a separate update."
    )

    col1, col2 = st.columns(2)

    with col1:
        for season, fname in [("2024-25", "game_results_2024-25.csv"), ("2025-26", "game_results_2025-26.csv")]:
            p = DATA_DIR / fname
            if p.exists():
                df = pd.read_csv(p, usecols=["GAME_DATE"])
                df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")
                st.metric(
                    f"{season} game log",
                    f"{len(df):,} rows",
                    f"latest: {df['GAME_DATE'].max().date()}",
                )
            else:
                st.metric(f"{season} game log", "not found")

    with col2:
        season_sel = st.selectbox("Season to update", ["2025-26", "2024-25"])
        if st.button("Update Game Logs", type="primary"):
            with st.spinner("Fetching new game results from NBA API…"):
                try:
                    from scripts.nba_log_scrapper import NBAResultsScraper
                    from datetime import timedelta

                    season_file_map = {"2025-26": "game_results_2025-26.csv", "2024-25": "game_results_2024-25.csv"}
                    season_start_map = {"2025-26": "2025-10-21", "2024-25": "2024-10-22"}
                    fname = season_file_map[season_sel]
                    existing_path = DATA_DIR / fname

                    if existing_path.exists():
                        existing = pd.read_csv(existing_path)
                        existing["GAME_DATE"] = pd.to_datetime(existing["GAME_DATE"], errors="coerce")
                        last_date = existing["GAME_DATE"].max()
                        start_date = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
                    else:
                        existing = pd.DataFrame()
                        start_date = season_start_map[season_sel]

                    today_str = date.today().strftime("%Y-%m-%d")
                    if start_date >= today_str:
                        st.success("Game logs already up to date.")
                    else:
                        st.write(f"Fetching from {start_date} → {today_str}…")
                        odds_df = pd.read_csv(DATA_DIR / "historical_odds_combined.csv", usecols=["player_name"])
                        players_list = odds_df["player_name"].dropna().unique().tolist()

                        scraper = NBAResultsScraper()
                        new_games = scraper.get_results_for_date_range(
                            start_date=start_date,
                            end_date=today_str,
                            players_list=players_list,
                            season=season_sel,
                        )

                        if new_games is not None and not new_games.empty:
                            combined = pd.concat([existing, new_games], ignore_index=True) if not existing.empty else new_games
                            combined.to_csv(existing_path, index=False)
                            st.success(f"Added {len(new_games):,} rows. Total: {len(combined):,}")
                        else:
                            st.info("No new games found.")
                except Exception:
                    st.error("Update failed:")
                    st.code(traceback.format_exc())

    st.divider()

    # ── defense cache ─────────────────────────────────────────────────────────
    st.subheader("Team Defense Cache")
    defense_file = DATA_DIR / "team_defense_features.csv"
    if defense_file.exists():
        df = pd.read_csv(defense_file, usecols=["game_date", "team_abbrev"])
        df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
        st.metric(
            "team_defense_features.csv",
            f"{len(df):,} rows  ({df['team_abbrev'].nunique()} teams)",
            f"latest: {df['game_date'].max().date()}",
        )
    else:
        st.warning("team_defense_features.csv not found. Run build_team_defense_cache.py to create it.")

    if st.button("Rebuild Defense Cache"):
        with st.spinner("Building team defense features…"):
            try:
                import importlib.util, sys
                spec = importlib.util.spec_from_file_location(
                    "build_defense", ROOT / "scripts" / "build_team_defense_cache.py"
                )
                mod = importlib.util.load_from_spec(spec) if hasattr(importlib.util, "load_from_spec") else None
                # Fall back to subprocess for safety
                import subprocess
                result = subprocess.run(
                    [sys.executable, "-m", "scripts.build_team_defense_cache"],
                    capture_output=True, text=True, cwd=str(ROOT),
                )
                if result.returncode == 0:
                    st.success("Defense cache rebuilt.")
                    st.code(result.stdout[-2000:] if result.stdout else "")
                else:
                    st.error("Build failed.")
                    st.code(result.stderr[-2000:])
            except Exception:
                st.error("Failed:")
                st.code(traceback.format_exc())

    st.divider()

    # ── validate training data ────────────────────────────────────────────────
    st.subheader("Validate Training Data")
    if st.button("Run Validation"):
        with st.spinner("Validating…"):
            try:
                import subprocess, sys
                result = subprocess.run(
                    [sys.executable, "-m", "scripts.validate_training_data"],
                    capture_output=True, text=True, cwd=str(ROOT),
                )
                st.code((result.stdout + result.stderr)[-3000:])
            except Exception:
                st.code(traceback.format_exc())


# ── navigation & layout ───────────────────────────────────────────────────────

PAGES = {
    # ── Analysis ──────────────────────────────
    "Dashboard":         page_dashboard,
    "Charts":            page_charts,
    # ── Predictions ───────────────────────────
    "Run Predictions":   page_run_predictions,
    "Predictions Table": page_predictions,
    "Combo Props":       page_combo_props,
    "Arbitrage & Lines": page_arbitrage,
    # ── Betting ───────────────────────────────
    "Select Model Bets": page_select_bets,
    "Manual / Parlay":   page_manual_bets,
    "Settle Bets":       page_settle,
    "Bet Tracker":       page_tracker,
    "Bankroll":          page_bankroll,
    # ── Data ──────────────────────────────────
    "Historical Odds":   page_odds_collection,
    "Maintenance":       page_maintenance,
}


def main():
    st.set_page_config(
        page_title="NBA Props Dashboard",
        page_icon="🏀",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(CSS, unsafe_allow_html=True)

    bankroll_data = load_bankroll()
    tracker = load_tracker()
    pending_count = int((tracker["result_status"] == "pending").sum()) if not tracker.empty else 0
    total_pnl = pd.to_numeric(
        tracker.loc[tracker["result_status"].isin(["won","lost","push"]), "profit_loss"],
        errors="coerce",
    ).sum() if not tracker.empty else 0.0

    with st.sidebar:
        st.markdown('<div class="sidebar-title">🏀 NBA Props</div>', unsafe_allow_html=True)
        pnl_sign = "▲" if total_pnl >= 0 else "▼"
        pnl_color = "#2ecc71" if total_pnl >= 0 else "#e74c3c"
        st.markdown(
            f'<div class="sidebar-bankroll">'
            f'<b style="color:#ffc72c">${bankroll_data["current"]:,.2f}</b><br>'
            f'<span style="font-size:0.8rem;color:{pnl_color}">'
            f'{pnl_sign} ${abs(total_pnl):,.2f} all-time</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
        if pending_count:
            st.warning(f"{pending_count} pending bet(s) to settle")

        st.markdown("---")

        # grouped nav labels
        GROUPS = {
            "Analysis":    ["Dashboard", "Charts"],
            "Predictions": ["Run Predictions", "Predictions Table",
                            "Combo Props", "Arbitrage & Lines"],
            "Betting":     ["Select Model Bets", "Manual / Parlay",
                            "Settle Bets", "Bet Tracker", "Bankroll"],
            "Data":        ["Historical Odds", "Maintenance"],
        }
        flat_pages = list(PAGES.keys())

        if "sel_page" not in st.session_state:
            st.session_state["sel_page"] = flat_pages[0]

        for group, items in GROUPS.items():
            st.markdown(
                f'<div style="font-size:0.68rem;color:#8892a4;text-transform:uppercase;'
                f'letter-spacing:1.5px;padding:8px 4px 2px">{group}</div>',
                unsafe_allow_html=True,
            )
            for item in items:
                active = st.session_state["sel_page"] == item
                bg = "#1d428a33" if active else "transparent"
                border = "2px solid #1d428a" if active else "2px solid transparent"
                if st.sidebar.button(
                    item,
                    key=f"nav_{item}",
                    use_container_width=True,
                ):
                    st.session_state["sel_page"] = item
                    st.rerun()

    PAGES[st.session_state["sel_page"]]()


if __name__ == "__main__":
    main()
