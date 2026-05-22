import pandas as pd
import numpy as np
from scripts.config import Config


TEAM_ABBREV_TO_FULL = {
    "ATL": "Atlanta Hawks",
    "BOS": "Boston Celtics",
    "BKN": "Brooklyn Nets",
    "CHA": "Charlotte Hornets",
    "CHI": "Chicago Bulls",
    "CLE": "Cleveland Cavaliers",
    "DAL": "Dallas Mavericks",
    "DEN": "Denver Nuggets",
    "DET": "Detroit Pistons",
    "GSW": "Golden State Warriors",
    "HOU": "Houston Rockets",
    "IND": "Indiana Pacers",
    "LAC": "LA Clippers",
    "LAL": "Los Angeles Lakers",
    "MEM": "Memphis Grizzlies",
    "MIA": "Miami Heat",
    "MIL": "Milwaukee Bucks",
    "MIN": "Minnesota Timberwolves",
    "NOP": "New Orleans Pelicans",
    "NYK": "New York Knicks",
    "OKC": "Oklahoma City Thunder",
    "ORL": "Orlando Magic",
    "PHI": "Philadelphia 76ers",
    "PHX": "Phoenix Suns",
    "POR": "Portland Trail Blazers",
    "SAC": "Sacramento Kings",
    "SAS": "San Antonio Spurs",
    "TOR": "Toronto Raptors",
    "UTA": "Utah Jazz",
    "WAS": "Washington Wizards",
}


def get_season_from_date(game_date):
    return "2024-25" if game_date < pd.Timestamp("2025-10-01") else "2025-26"


def build_team_game_table(game_logs):

    game_logs = game_logs.copy()

    # Normalize columns
    game_logs.columns = game_logs.columns.str.upper()

    # Parse dates
    game_logs["GAME_DATE"] = pd.to_datetime(
        game_logs["GAME_DATE"],
        errors="coerce"
    )

    # Remove invalid rows
    game_logs = game_logs.dropna(
        subset=["GAME_DATE", "MATCHUP"]
    )

    # Extract team abbreviation
    def extract_team(matchup):

        if pd.isna(matchup):
            return None

        matchup = str(matchup).strip()


        return matchup.split()[0]

    # Extract opponent abbreviation
    def extract_opponent(matchup):

        if pd.isna(matchup):
            return None

        matchup = str(matchup).strip()

        if "vs." in matchup:
            return matchup.split("vs.")[1].strip().split()[0]

        if "@" in matchup:
            return matchup.split("@")[1].strip().split()[0]

        return None

    game_logs["TEAM_ABBREV"] = (
        game_logs["MATCHUP"]
        .apply(extract_team)
    )

    game_logs["OPP_ABBREV"] = (
        game_logs["MATCHUP"]
        .apply(extract_opponent)
    )

    # Numeric columns used for defense calculations
    numeric_cols = [
        "PTS",
        "REB",
        "AST",
        "OREB",
        "DREB",
        "TOV",
        "FG3M",
        "FG3A",
        "FGA",
        "FTA",
        "MIN"
    ]

    # Convert safely to numeric
    for col in numeric_cols:

        if col in game_logs.columns:

            game_logs[col] = pd.to_numeric(
                game_logs[col],
                errors="coerce"
            ).fillna(0)

    # Aggregate player logs into team game stats
    team_games = (
        game_logs
        .groupby(
            [
                "GAME_ID",
                "GAME_DATE",
                "TEAM_ABBREV",
                "OPP_ABBREV"
            ],
            as_index=False
        )
        .agg({
            "PTS": "sum",
            "REB": "sum",
            "AST": "sum",
            "OREB": "sum",
            "DREB": "sum",
            "TOV": "sum",
            "FG3M": "sum",
            "FG3A": "sum",
            "FGA": "sum",
            "FTA": "sum",
        })
    )

    # Sort chronologically
    team_games = team_games.sort_values(
        "GAME_DATE"
    ).reset_index(drop=True)

    print(f"Built team games table: {len(team_games):,} rows")

    return team_games


def build_defense_features(team_games, window=10):
    rows = []

    team_games = team_games.sort_values("GAME_DATE")

    for team in sorted(team_games["TEAM_ABBREV"].dropna().unique()):
        allowed_games = team_games[
            team_games["OPP_ABBREV"] == team
        ].sort_values("GAME_DATE")

        for _, game in allowed_games.iterrows():
            game_date = game["GAME_DATE"]

            prior_games = allowed_games[
                allowed_games["GAME_DATE"] < game_date
            ].tail(window)

            if prior_games.empty:
                continue

            team_name = TEAM_ABBREV_TO_FULL.get(team, team)

            rows.append({
                "team_name": team_name,
                "team_abbrev": team,
                "game_date": game_date.strftime("%Y-%m-%d"),
                "season": get_season_from_date(game_date),
                "opponent_pts_allowed": prior_games["PTS"].mean(),
                "opponent_reb_rate": prior_games["REB"].mean(),
                "opponent_ast_allowed": prior_games["AST"].mean(),
                "opponent_tov_rate": prior_games["TOV"].mean(),
                "opponent_three_def": prior_games["FG3M"].mean(),
                "opponent_fg3a_allowed": prior_games["FG3A"].mean(),
                "opponent_fga_allowed": prior_games["FGA"].mean(),
                "opponent_fta_allowed": prior_games["FTA"].mean(),
                "games_used": len(prior_games),
                "defense_source": f"local_last_{len(prior_games)}",
            })

    return pd.DataFrame(rows)


def main():
    files = [
        Config.DATA_DIR / "game_results_2024-25.csv",
        Config.DATA_DIR / "game_results_2025-26.csv",
    ]

    logs = []

    for file in files:
        if file.exists():
            df = pd.read_csv(file)
            logs.append(df)
            print(f"Loaded {file.name}: {len(df):,} rows")

    if not logs:
        print("No game result files found.")
        return

    game_logs = pd.concat(logs, ignore_index=True)

    print("Building team game table...")
    team_games = build_team_game_table(game_logs)

    print(f"Team games: {len(team_games):,}")

    print("Building local rolling defense features...")
    defense_df = build_defense_features(team_games, window=10)

    output_file = Config.DATA_DIR / "team_defense_features.csv"

    defense_df.to_csv(output_file, index=False)

    print(f"Saved: {output_file}")
    print(f"Rows: {len(defense_df):,}")
    print(f"Date range: {defense_df['game_date'].min()} to {defense_df['game_date'].max()}")


if __name__ == "__main__":
    main()