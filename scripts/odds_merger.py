"""
Functions for merging sportsbook odds with actual game results and labeling outcomes.
"""

import numpy as np
import pandas as pd

from scripts.config import Config


def merge_odds_with_results(odds_df, results_df):
    """
    Merge sportsbook odds with actual NBA game results.

    Returns a DataFrame with odds info, actual stat result, and hit/miss labels.
    """
    print("\nMerging odds with game results...")

    odds_df = odds_df.copy()
    results_df = results_df.copy()

    results_df.columns = results_df.columns.str.upper()

    required_cols = ['PLAYER_NAME', 'GAME_DATE']
    missing = [c for c in required_cols if c not in results_df.columns]
    if missing:
        print(f"Error: Missing required columns: {missing}")
        print(f"Available columns: {list(results_df.columns)}")
        return pd.DataFrame()

    odds_df["game_date"] = (
        pd.to_datetime(odds_df["game_date"], errors="coerce", utc=True)
        .dt.tz_convert(None)
        .dt.strftime("%Y-%m-%d")
    )
    results_df["GAME_DATE"] = pd.to_datetime(
        results_df["GAME_DATE"], errors="coerce"
    ).dt.strftime("%Y-%m-%d")

    odds_df = odds_df.dropna(subset=["game_date"])
    results_df = results_df.dropna(subset=["GAME_DATE"])

    odds_df['normalized_player_name'] = odds_df['player_name'].astype(str).str.lower().str.strip()
    results_df['normalized_player_name'] = results_df['PLAYER_NAME'].astype(str).str.lower().str.strip()

    odds_df['merge_key'] = odds_df['normalized_player_name'] + '_' + odds_df['game_date']
    results_df['merge_key'] = results_df['normalized_player_name'] + '_' + results_df['GAME_DATE']

    merged_frames = []

    for market in sorted(odds_df['market'].dropna().unique()):
        market_odds = odds_df[odds_df['market'] == market].copy()
        stat_col = Config.MARKET_TO_STAT.get(market)

        if stat_col is None:
            print(f"Skipping {market}: no stat mapping found")
            continue

        stat_col = stat_col.upper()

        if stat_col not in results_df.columns:
            print(f"Skipping {market}: '{stat_col}' not found in results")
            continue

        market_merged = market_odds.merge(
            results_df[['merge_key', stat_col]],
            on='merge_key',
            how='left',
        ).rename(columns={stat_col: 'actual_value'})

        market_merged['actual_value'] = pd.to_numeric(
            market_merged['actual_value'], errors='coerce'
        )
        merged_frames.append(market_merged)

    if not merged_frames:
        print("No markets were successfully merged")
        return pd.DataFrame()

    merged = pd.concat(merged_frames, ignore_index=True)

    dedupe_cols = [
        c for c in [
            "player_name", "game_date", "market", "line", "bet_type",
            "bookmaker", "price", "odds", "american_odds",
            "commence_time", "last_update", "timestamp",
        ]
        if c in merged.columns
    ]

    before = len(merged)
    merged = merged.drop_duplicates(subset=dedupe_cols, keep="last")
    print(f"Removed exact duplicate odds rows: {before - len(merged):,}")

    before_rows = len(merged)
    merged = merged.dropna(subset=['actual_value'])
    dropped = before_rows - len(merged)
    drop_pct = dropped / before_rows * 100 if before_rows else 0

    print(f"Merged props: {len(merged):,}")
    print(f"Dropped props without results: {dropped:,} ({drop_pct:.1f}%)")
    print("\nMerge results by market:")
    for market, count in merged['market'].value_counts().items():
        print(f"  {market}: {count:,}")

    return merged


def label_results(merged_df):
    """
    Add hit (1/0) and edge (actual - line) columns to merged props.
    """
    print("\nLabeling prop results...")

    if merged_df.empty:
        print("No data to label")
        return pd.DataFrame()

    labeled = merged_df.copy()
    labeled['actual_value'] = pd.to_numeric(labeled['actual_value'], errors='coerce')
    labeled['line'] = pd.to_numeric(labeled['line'], errors='coerce')

    before = len(labeled)
    labeled = labeled.dropna(subset=['actual_value', 'line'])
    dropped = before - len(labeled)

    if labeled.empty:
        print("All rows removed due to missing values")
        return pd.DataFrame()

    labeled['bet_type'] = (
        labeled.get('bet_type', 'Over').astype(str).str.lower().str.strip()
    )
    labeled['edge'] = labeled['actual_value'] - labeled['line']
    labeled['hit'] = np.where(
        ((labeled['bet_type'] == 'over') & (labeled['actual_value'] > labeled['line']))
        | ((labeled['bet_type'] == 'under') & (labeled['actual_value'] < labeled['line'])),
        1,
        0,
    )

    print(f"Labeled props: {len(labeled):,}")
    if dropped:
        print(f"Dropped rows with missing values: {dropped:,}")
    print(f"Historical hit rate: {labeled['hit'].mean() * 100:.2f}%")

    return labeled
