"""
Stateless utility functions for player feature calculation.
"""

import numpy as np
import pandas as pd


TEAM_NAME_MAP = {
    'ATL': 'Atlanta Hawks',
    'BOS': 'Boston Celtics',
    'BKN': 'Brooklyn Nets',
    'CHA': 'Charlotte Hornets',
    'CHI': 'Chicago Bulls',
    'CLE': 'Cleveland Cavaliers',
    'DAL': 'Dallas Mavericks',
    'DEN': 'Denver Nuggets',
    'DET': 'Detroit Pistons',
    'GSW': 'Golden State Warriors',
    'HOU': 'Houston Rockets',
    'IND': 'Indiana Pacers',
    'LAC': 'LA Clippers',
    'LAL': 'Los Angeles Lakers',
    'MEM': 'Memphis Grizzlies',
    'MIA': 'Miami Heat',
    'MIL': 'Milwaukee Bucks',
    'MIN': 'Minnesota Timberwolves',
    'NOP': 'New Orleans Pelicans',
    'NYK': 'New York Knicks',
    'OKC': 'Oklahoma City Thunder',
    'ORL': 'Orlando Magic',
    'PHI': 'Philadelphia 76ers',
    'PHX': 'Phoenix Suns',
    'POR': 'Portland Trail Blazers',
    'SAC': 'Sacramento Kings',
    'SAS': 'San Antonio Spurs',
    'TOR': 'Toronto Raptors',
    'UTA': 'Utah Jazz',
    'WAS': 'Washington Wizards',
}


def get_full_team_name(team_code):
    return TEAM_NAME_MAP.get(team_code, team_code)


def extract_opponent_from_matchup(matchup):
    """
    Parse opponent team code from a matchup string.
    "LAL vs. PHI" → "PHI",  "GSW @ OKC" → "OKC"
    """
    if not matchup:
        return None
    try:
        if 'vs.' in matchup:
            opp = matchup.split('vs.')[1].strip()
            return opp.split()[0] if opp else None
        elif '@' in matchup:
            opp = matchup.split('@')[1].strip()
            return opp.split()[0] if opp else None
    except Exception:
        pass
    return None


def calculate_home_away_splits(player_stats, stat_col, is_home):
    """Return home/away split dict or None if < 5 games in either context."""
    home = player_stats[~player_stats['MATCHUP'].str.contains('@', na=False)]
    away = player_stats[player_stats['MATCHUP'].str.contains('@', na=False)]

    if len(home) < 5 or len(away) < 5:
        return None

    home_avg = home[stat_col].mean()
    away_avg = away[stat_col].mean()
    context_games = home if is_home else away

    return {
        'home_away_split': home_avg - away_avg,
        'context_avg': home_avg if is_home else away_avg,
        'context_consistency': context_games[stat_col].std(),
    }


def calculate_vs_opponent_history(player_stats, opponent, stat_col):
    """Return weighted average vs opponent or None if < 3 historical matchups."""
    if not opponent or player_stats.empty:
        return None

    required = {'MATCHUP', 'GAME_DATE', stat_col}
    if not required.issubset(player_stats.columns):
        return None

    stats = player_stats.copy()
    stats['GAME_DATE'] = pd.to_datetime(stats['GAME_DATE'], errors='coerce')
    stats = stats.sort_values('GAME_DATE', ascending=False)

    vs_games = stats[
        stats['MATCHUP'].str.contains(
            rf'(\bvs\.\s+{opponent}\b|\@\s+{opponent}\b)',
            regex=True,
            na=False,
        )
    ].head(8)

    if len(vs_games) < 3:
        return None

    weights = np.linspace(1.0, 0.6, len(vs_games))
    values = pd.to_numeric(vs_games[stat_col], errors='coerce').fillna(0)

    return {
        'vs_opponent_avg': round(float(np.average(values, weights=weights)), 3),
        'vs_opponent_games_count': len(vs_games),
    }


def calculate_recent_trend_features(player_stats, stat_col):
    """Return momentum/trend dict or None if < 10 games."""
    if len(player_stats) < 10:
        return None

    last_5 = player_stats.head(5)[stat_col]
    games_6_10 = player_stats.iloc[5:10][stat_col]

    weights = np.array([0.3, 0.25, 0.2, 0.15, 0.1])

    return {
        'recent_vs_older': last_5.mean() - games_6_10.mean(),
        'momentum_score': np.average(last_5.values, weights=weights),
        'peak_recent': last_5.max(),
    }


def build_market_features(player_stats, market, avg_minutes, opp_stats):
    """Build market-specific engineered features."""
    features = {}

    if market == "player_points":
        if "FG_PCT" in player_stats.columns:
            fg_recent = pd.to_numeric(player_stats.head(5)["FG_PCT"], errors="coerce")
            fg_old = pd.to_numeric(player_stats.iloc[5:10]["FG_PCT"], errors="coerce")
            features["fg_pct_L5"] = fg_recent.mean()
            features["fg_pct_trend"] = fg_recent.mean() - fg_old.mean()

        if "FTA" in player_stats.columns:
            features["fta_rate_L5"] = pd.to_numeric(
                player_stats.head(5)["FTA"], errors="coerce"
            ).mean()

        if {"FGA", "PTS"}.issubset(player_stats.columns):
            fga_l5 = pd.to_numeric(player_stats.head(5)["FGA"], errors="coerce").mean()
            if fga_l5 > 0:
                pts_l5 = pd.to_numeric(player_stats.head(5)["PTS"], errors="coerce").mean()
                features["efficiency_last_5"] = pts_l5 / fga_l5
            features["pts_per_minute"] = (
                pd.to_numeric(player_stats["PTS"], errors="coerce").mean()
                / (avg_minutes + 0.1)
            )

    elif market == "player_assists":
        if {"AST", "TOV"}.issubset(player_stats.columns):
            ast_mean = pd.to_numeric(player_stats["AST"], errors="coerce").mean()
            tov_mean = pd.to_numeric(player_stats["TOV"], errors="coerce").mean()
            if tov_mean > 0.1:
                features["ast_to_tov_ratio"] = ast_mean / tov_mean
            features["ast_per_minute"] = ast_mean / (avg_minutes + 0.1)

        if opp_stats is not None and opp_stats.get("pace") is not None:
            features["pace_factor_deviation"] = opp_stats["pace"] - 100.0

    elif market == "player_rebounds":
        if "REB" in player_stats.columns:
            reb_mean = pd.to_numeric(player_stats["REB"], errors="coerce").mean()
            features["reb_per_minute"] = reb_mean / (avg_minutes + 0.1)

            if {"OREB", "DREB"}.issubset(player_stats.columns) and reb_mean > 0.1:
                features["oreb_rate"] = (
                    pd.to_numeric(player_stats["OREB"], errors="coerce").mean() / reb_mean
                )
                features["dreb_rate"] = (
                    pd.to_numeric(player_stats["DREB"], errors="coerce").mean() / reb_mean
                )
                features["oreb_L5"] = pd.to_numeric(
                    player_stats.head(5)["OREB"], errors="coerce"
                ).mean()
                features["dreb_L5"] = pd.to_numeric(
                    player_stats.head(5)["DREB"], errors="coerce"
                ).mean()

    elif market == "player_threes":
        if {"FG3A", "FG3M"}.issubset(player_stats.columns):
            fg3a_l5 = pd.to_numeric(player_stats.head(5)["FG3A"], errors="coerce").mean()
            fg3m_l5 = pd.to_numeric(player_stats.head(5)["FG3M"], errors="coerce").mean()
            if fg3a_l5 > 0:
                features["three_point_pct_L5"] = fg3m_l5 / fg3a_l5
                features["three_point_attempts_L5"] = fg3a_l5

    return features


def validate_feature_consistency(features_df):
    """Print a summary of potential feature quality issues."""
    print("\n" + "=" * 60)
    print("FEATURE VALIDATION")
    print("=" * 60)

    if features_df.empty:
        print("Feature dataframe is empty")
        return

    metadata_cols = {'actual_value', 'hit', 'edge', 'line', 'market', 'player_name', 'game_date'}
    feature_cols = [c for c in features_df.columns if c not in metadata_cols]

    issues = []

    for col in feature_cols:
        series = pd.to_numeric(features_df[col], errors='coerce')
        total = len(series)
        if total == 0:
            continue

        non_null = series.dropna()
        if non_null.empty:
            issues.append(f"{col}: all values missing")
            continue

        if non_null.nunique() <= 1:
            issues.append(f"{col}: constant feature")
            continue

        missing_pct = series.isna().sum() / total * 100
        if missing_pct > 50:
            issues.append(f"{col}: {missing_pct:.1f}% missing")

        zero_pct = (non_null == 0).sum() / len(non_null) * 100
        if zero_pct > 85:
            issues.append(f"{col}: {zero_pct:.1f}% zeros")

        if non_null.var() < 1e-4:
            issues.append(f"{col}: extremely low variance ({non_null.var():.6f})")

    if issues:
        print("\nPotential feature issues:\n")
        for issue in issues[:25]:
            print(f"  - {issue}")
        if len(issues) > 25:
            print(f"\n... and {len(issues) - 25} more")
    else:
        print("\nNo major feature issues detected")

    print("\nFeature Summary")
    print("-" * 60)
    print(f"Training samples: {len(features_df):,}")
    print(f"Feature count: {len(feature_cols)}")
    if 'market' in features_df.columns:
        print(f"Markets: {features_df['market'].nunique()}")
    print("=" * 60)
