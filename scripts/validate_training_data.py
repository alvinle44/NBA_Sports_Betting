import pandas as pd
from scripts.config import Config


EXPECTED_BY_MARKET = {
    "player_points": [
        "fg_pct_L5",
        "fg_pct_trend",
        "fta_rate_L5",
        "efficiency_last_5",
        "pts_per_minute",
    ],
    "player_assists": [
        "ast_to_tov_ratio",
        "ast_per_minute",
    ],
    "player_rebounds": [
        "reb_per_minute",
        "oreb_rate",
        "dreb_rate",
        "oreb_L5",
        "dreb_L5",
    ],
    "player_threes": [
        "three_point_pct_L5",
        "three_point_attempts_L5",
    ],
}


def validate_market_features(data_file="data/prepared_training_data.csv"):
    df = pd.read_csv(data_file)

    print("\nTraining Data Validation")
    print("=" * 60)
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns):,}")

    print("\nMarkets:")
    print(df["market"].value_counts())

    print("\nChecking market-specific features...")
    for market, features in EXPECTED_BY_MARKET.items():
        market_df = df[df["market"] == market]

        if market_df.empty:
            print(f"\n{market}: no rows")
            continue

        print(f"\n{market}: {len(market_df):,} rows")

        for feature in features:
            if feature not in market_df.columns:
                print(f"  MISSING COLUMN: {feature}")
                continue

            missing_pct = market_df[feature].isna().mean() * 100

            if missing_pct > 25:
                print(f"  BAD: {feature} is {missing_pct:.1f}% missing")
            else:
                print(f"  OK:  {feature} is {missing_pct:.1f}% missing")

    print("\nChecking duplicate odds rows...")
    dedupe_cols = [
        "player_name",
        "game_date",
        "market",
        "line",
        "bet_type",
        "bookmaker",
    ]

    dedupe_cols = [c for c in dedupe_cols if c in df.columns]

    duplicate_count = df.duplicated(subset=dedupe_cols).sum()

    if duplicate_count > 0:
        print(f"BAD: Found {duplicate_count:,} duplicate rows")
    else:
        print("OK: No duplicate rows found")

    print("\nChecking target leakage columns...")
    leakage_cols = ["actual_value", "hit", "edge"]

    print("These should exist in CSV, but must NOT be used as model features:")
    for col in leakage_cols:
        print(f"  {col}: {'exists' if col in df.columns else 'missing'}")

    print("\nDone.")

    


if __name__ == "__main__":
    validate_market_features()

    df = pd.read_csv("data/prepared_training_data.csv")

    dedupe_cols = [
        "player_name",
        "game_date",
        "market",
        "line",
        # "bookmaker",
    ]

    dedupe_cols = [c for c in dedupe_cols if c in df.columns]

    dupes = df[df.duplicated(subset=dedupe_cols, keep=False)]

    print(f"Duplicate rows found: {len(dupes):,}")

    show_cols = [
        "player_name",
        "game_date",
        "market",
        "line",
        # "bookmaker",
    ]

    extra_cols = [
        "bookmaker",
        "price",
        "odds",
        "american_odds",
        "last_update",
    ]

    for col in extra_cols:
        if col in dupes.columns:
            show_cols.append(col)

    print(
        dupes[show_cols]
        .sort_values(show_cols[:5])
        .head(100)
        .to_string(index=False)
    )