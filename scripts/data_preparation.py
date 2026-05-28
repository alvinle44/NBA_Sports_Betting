import requests
import pandas as pd
import numpy as np
from datetime import datetime
import time
import warnings
import argparse
from pathlib import Path

from scripts.teammate_analyzer import TeammateImpactAnalyzer
from scripts.nba_log_scrapper import NBAResultsScraper
from scripts.config import Config
from scripts.rapidapi_injury_scrapper import RapidAPIInjuryScraper
from scripts.feature_utils import (
    extract_opponent_from_matchup,
    get_full_team_name,
    calculate_home_away_splits,
    calculate_vs_opponent_history,
    calculate_recent_trend_features,
    build_market_features,
    validate_feature_consistency,
)
from scripts.odds_merger import merge_odds_with_results, label_results

warnings.filterwarnings('ignore')


class DataPreparation:
    """
    Prepares training data with no data leakage and no preset values.
    """

    def __init__(self):
        self.scraper = NBAResultsScraper()
        self.impact_analyzer = TeammateImpactAnalyzer(self.scraper)
        self.injury_scraper = RapidAPIInjuryScraper()

    # ------------------------------------------------------------------
    # Injury helpers
    # ------------------------------------------------------------------

    def _get_injuries_from_rapidapi(self, team_code, game_date):
        """Return list of injured player names for a team on a given date."""
        if not team_code or not game_date:
            return []
        try:
            date_str = pd.to_datetime(game_date).strftime("%Y-%m-%d")
            injury_dict = self.injury_scraper.load_injuries_for_game_date(date_str)
            if not injury_dict:
                return []
            full_name = get_full_team_name(team_code)
            team_injuries = injury_dict.get(team_code) or injury_dict.get(full_name) or []
            return [p.get("name") for p in team_injuries if isinstance(p, dict) and p.get("name")]
        except Exception as e:
            print(f"Error loading injuries for {team_code}: {str(e)[:80]}")
            return []

    # ------------------------------------------------------------------
    # Feature building
    # ------------------------------------------------------------------

    def _load_player_stats_cache(self):
        """Load all cached game result CSVs into a single DataFrame."""
        frames = []
        for file_name in ["game_results_2024-25.csv", "game_results_2025-26.csv"]:
            path = Config.DATA_DIR / file_name
            if path.exists():
                df = pd.read_csv(path)
                frames.append(df)
                print(f"Loaded {len(df):,} rows from {file_name}")

        if not frames:
            print("No cached game results found")
            return pd.DataFrame(), set()

        combined = pd.concat(frames, ignore_index=True)
        combined["GAME_DATE"] = pd.to_datetime(combined["GAME_DATE"], errors="coerce")
        combined = combined.dropna(subset=["GAME_DATE"])
        return combined, set(combined["PLAYER_NAME"].dropna().unique())

    def _fetch_player_stats_from_api(self, player, season, all_stats_df, cached_players):
        """Attempt to fetch player stats from the NBA API (3 retries)."""
        for attempt in range(3):
            try:
                player_stats = self.scraper.get_player_game_stats(player, season)
                if player_stats is not None and not player_stats.empty:
                    player_stats["PLAYER_NAME"] = player
                    player_stats["GAME_DATE"] = pd.to_datetime(
                        player_stats["GAME_DATE"], errors="coerce"
                    )
                    all_stats_df = pd.concat([all_stats_df, player_stats], ignore_index=True)
                    cached_players.add(player)
                    return player_stats, all_stats_df, cached_players
            except requests.exceptions.Timeout:
                if attempt < 2:
                    time.sleep((attempt + 1) * 20)
            except Exception:
                break
        return pd.DataFrame(), all_stats_df, cached_players

    def _load_defense_lookup(self):
        defense_file = Config.DATA_DIR / "team_defense_features.csv"
        if not defense_file.exists():
            print("No defense cache found")
            return {}

        df = pd.read_csv(defense_file)
        df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce").dt.strftime("%Y-%m-%d")
        lookup = {
            (row["team_name"], row["game_date"], row["season"]): row
            for _, row in df.iterrows()
        }
        print(f"Loaded defense cache: {len(lookup):,} rows")
        return lookup

    def _resolve_opponent_stats(self, opponent, game_date, game_season, defense_lookup):
        if not opponent or not defense_lookup:
            return None
        key = (opponent, game_date.strftime("%Y-%m-%d"), game_season)
        cached = defense_lookup.get(key)
        if cached is None:
            return None
        return {
            "def_rating_weighted": cached.get("opponent_def_weighted"),
            "pace": cached.get("pace_factor"),
            "opp_pts_weighted": cached.get("opponent_pts_allowed"),
            "opp_reb_rate": cached.get("opponent_reb_rate"),
            "opp_tov_rate": cached.get("opponent_tov_rate"),
            "opp_three_def": cached.get("opponent_three_def"),
        }

    def _load_checkpoint(self):
        checkpoint_file = Config.DATA_DIR / "features_checkpoint.csv"
        indices_file = Config.DATA_DIR / "processed_indices.txt"

        if checkpoint_file.exists() and indices_file.exists():
            try:
                df = pd.read_csv(checkpoint_file)
                with open(indices_file) as f:
                    indices = {int(l.strip()) for l in f if l.strip()}
                print(f"Resuming from checkpoint: {len(df):,} rows")
                return df.to_dict("records"), indices
            except Exception as e:
                print(f"Could not load checkpoint: {e}")

        return [], set()

    def _save_checkpoint(self, features_list, processed_indices):
        checkpoint_file = Config.DATA_DIR / "features_checkpoint.csv"
        indices_file = Config.DATA_DIR / "processed_indices.txt"
        try:
            pd.DataFrame(features_list).to_csv(checkpoint_file, index=False)
            with open(indices_file, "w") as f:
                f.writelines(f"{i}\n" for i in processed_indices)
        except Exception as e:
            print(f"Checkpoint save failed: {e}")

    def build_features(self, labeled_df, season="2025-26"):
        """
        Build ML features for each prop row using only pre-game data (no leakage).
        """
        print("\nBuilding features...")

        all_stats_df, cached_players = self._load_player_stats_cache()
        if all_stats_df.empty:
            return pd.DataFrame()

        defense_lookup = self._load_defense_lookup()
        features_list, processed_indices = self._load_checkpoint()

        total = len(labeled_df)
        remaining = total - len(processed_indices)
        print(f"Total props: {total:,} | Already processed: {len(processed_indices):,} | Remaining: {remaining:,}")

        cache_hits = api_calls = api_failures = leakage_warnings = 0
        last_save_count = len(features_list)

        for idx, row in labeled_df.iterrows():
            if idx in processed_indices:
                continue

            current_count = len(features_list)

            if current_count % 100 == 0 and current_count != last_save_count:
                pct = current_count / remaining * 100 if remaining else 0
                print(
                    f"Processed {current_count:,}/{remaining:,} ({pct:.2f}%) | "
                    f"Player: {str(row.get('player_name', ''))[:20]:<20} | "
                    f"Cache hits: {cache_hits} | Leakage warnings: {leakage_warnings}"
                )

            if current_count % 1000 == 0 and current_count > last_save_count:
                self._save_checkpoint(features_list, processed_indices)
                print(f"Checkpoint saved at {current_count:,} rows")
                last_save_count = current_count

            player = row["player_name"]
            game_date = pd.to_datetime(row["game_date"], errors="coerce")
            if pd.isna(game_date):
                continue
            game_date = game_date.tz_localize(None) if game_date.tzinfo else game_date

            market = row["market"]
            line = row["line"]
            stat_col = Config.MARKET_TO_STAT.get(market)
            if not stat_col:
                continue

            # --- get player historical stats ---
            if player in cached_players:
                player_stats = all_stats_df[all_stats_df["PLAYER_NAME"] == player].copy()
                cache_hits += 1
            else:
                api_calls += 1
                player_stats, all_stats_df, cached_players = self._fetch_player_stats_from_api(
                    player, season, all_stats_df, cached_players
                )
                if player_stats.empty:
                    api_failures += 1

            if player_stats is None or player_stats.empty:
                continue

            player_stats["GAME_DATE"] = pd.to_datetime(player_stats["GAME_DATE"], errors="coerce")
            player_stats = player_stats.dropna(subset=["GAME_DATE"])
            player_stats = player_stats[player_stats["GAME_DATE"] < game_date].sort_values(
                "GAME_DATE", ascending=False
            )

            if player_stats.empty or len(player_stats) < 10:
                continue

            if player_stats.iloc[0]["GAME_DATE"] >= game_date:
                leakage_warnings += 1
                continue

            if stat_col not in player_stats.columns:
                continue

            stat_values = pd.to_numeric(player_stats[stat_col], errors="coerce")
            if stat_values.dropna().shape[0] < 10:
                continue

            # --- base stats ---
            last_5 = stat_values.head(5)
            avg_last_5 = last_5.mean()
            avg_last_10 = stat_values.head(10).mean()
            avg_season = stat_values.mean()

            matchup = player_stats.iloc[0].get("MATCHUP", "")
            opponent = extract_opponent_from_matchup(matchup)
            is_home = 0 if "@" in str(matchup) else 1

            last_game_date = player_stats.iloc[0]["GAME_DATE"]
            prev_game_date = player_stats.iloc[1]["GAME_DATE"]
            days_rest = max((last_game_date - prev_game_date).days, 0)

            avg_minutes = pd.to_numeric(player_stats["MIN"], errors="coerce").mean()
            recent_minutes = pd.to_numeric(player_stats.head(5)["MIN"], errors="coerce").mean()

            usage_rate_proxy = None
            if {"FGA", "FTA"}.issubset(player_stats.columns):
                usage_rate_proxy = (
                    pd.to_numeric(player_stats["FGA"], errors="coerce")
                    + 0.44 * pd.to_numeric(player_stats["FTA"], errors="coerce")
                ).mean()

            trend_l3 = 0.0
            if len(last_5) >= 3:
                y = last_5.tail(3).values
                trend_l3 = np.polyfit(np.arange(len(y)), y, 1)[0]

            game_season = "2024-25" if game_date < pd.Timestamp("2025-10-01") else "2025-26"
            opp_stats = self._resolve_opponent_stats(opponent, game_date, game_season, defense_lookup)

            # --- injury / teammate impact ---
            team_code = str(matchup).split()[0] if matchup else None
            injured_teammates = (
                self._get_injuries_from_rapidapi(team_code, game_date.date())
                if game_date >= pd.Timestamp("2025-10-01")
                else []
            )
            try:
                impact_features = self.impact_analyzer.calculate_impact_features(
                    player_name=player,
                    injured_players=injured_teammates,
                    market=market,
                    season=game_season,
                )
            except Exception:
                impact_features = None

            last_week_games = player_stats[
                player_stats["GAME_DATE"] >= game_date - pd.Timedelta(days=7)
            ]

            # --- assemble feature dict ---
            features = {
                "actual_value": row["actual_value"],
                "hit": row.get("hit"),
                "edge": row.get("edge"),
                "line": line,
                "market": market,
                "player_name": player,
                "game_date": game_date.date(),
                "avg_last_5": avg_last_5,
                "avg_last_10": avg_last_10,
                "last_game": stat_values.iloc[0],
                "consistency_L5": last_5.std(),
                "line_vs_L5": line - avg_last_5,
                "line_vs_season": line - avg_season,
                "is_home": is_home,
                "days_rest": days_rest,
                "back_to_back": int(days_rest <= 1),
                "avg_minutes": avg_minutes,
                "minutes_trend": recent_minutes - avg_minutes,
                "hot_hand_indicator": int((last_5.head(3) > avg_season).sum() >= 2),
                "trend_L3": trend_l3,
                "games_in_last_week": len(last_week_games),
            }

            if usage_rate_proxy is not None:
                features["usage_rate_proxy"] = usage_rate_proxy

            if opp_stats is not None:
                features["opponent_def_weighted"] = opp_stats.get("def_rating_weighted")
                features["pace_factor"] = opp_stats.get("pace")
                features["opponent_pts_allowed"] = opp_stats.get("opp_pts_weighted")
                features["opponent_reb_rate"] = opp_stats.get("opp_reb_rate")
                features["opponent_tov_rate"] = opp_stats.get("opp_tov_rate")
                if opp_stats.get("opp_three_def") is not None:
                    features["opponent_three_def"] = opp_stats["opp_three_def"]

            if impact_features and impact_features.get("missing_minutes_sum", 0) > 0:
                features.update(impact_features)

            home_away = calculate_home_away_splits(player_stats, stat_col, is_home)
            if home_away:
                features.update(home_away)

            vs_opp = calculate_vs_opponent_history(player_stats, opponent, stat_col)
            if vs_opp:
                features.update(vs_opp)

            trend = calculate_recent_trend_features(player_stats, stat_col)
            if trend:
                features.update(trend)

            features.update(build_market_features(player_stats, market, avg_minutes, opp_stats))

            features_list.append(features)
            processed_indices.add(idx)

        features_df = pd.DataFrame(features_list)
        print(f"\nBuilt features for {len(features_df):,} props")
        print(f"Cache hits: {cache_hits:,} | API calls: {api_calls:,} | "
              f"API failures: {api_failures:,} | Leakage warnings: {leakage_warnings:,}")

        if not features_df.empty:
            validate_feature_consistency(features_df)

        for path in [
            Config.DATA_DIR / "features_checkpoint.csv",
            Config.DATA_DIR / "processed_indices.txt",
        ]:
            if path.exists():
                path.unlink()

        return features_df

    # ------------------------------------------------------------------
    # Pipeline orchestration
    # ------------------------------------------------------------------

    def _load_odds(self, start_date, end_date):
        odds_file = Config.DATA_DIR / "historical_odds_combined.csv"
        if not odds_file.exists():
            print(f"Combined odds file not found: {odds_file}")
            return None

        odds_df = pd.read_csv(odds_file)
        if odds_df.empty:
            print("Combined odds file is empty")
            return None

        odds_df["game_date"] = (
            pd.to_datetime(odds_df["game_date"], errors="coerce", utc=True)
            .dt.tz_convert(None)
        )
        odds_df = odds_df.dropna(subset=["game_date"])

        if start_date:
            odds_df = odds_df[odds_df["game_date"] >= pd.to_datetime(start_date)]
        if end_date:
            odds_df = odds_df[odds_df["game_date"] <= pd.to_datetime(end_date)]

        if odds_df.empty:
            print("No odds found in the requested date range")
            return None

        return odds_df

    def _load_results(self, odds_df, end_date):
        season_2024_count = (odds_df["game_date"] < pd.Timestamp("2025-10-01")).sum()
        season_2025_count = (odds_df["game_date"] >= pd.Timestamp("2025-10-01")).sum()
        unique_players = odds_df["player_name"].dropna().unique()

        season_jobs = [
            {
                "name": "2024-25",
                "count": season_2024_count,
                "cache_file": Config.DATA_DIR / "game_results_2024-25.csv",
                "fetch_start": "2024-10-22",
                "fetch_end": "2025-06-30",
            },
            {
                "name": "2025-26",
                "count": season_2025_count,
                "cache_file": Config.DATA_DIR / "game_results_2025-26.csv",
                "fetch_start": "2025-10-21",
                "fetch_end": (
                    pd.to_datetime(end_date).strftime("%Y-%m-%d")
                    if end_date
                    else datetime.now().strftime("%Y-%m-%d")
                ),
            },
        ]

        all_results = []
        for job in season_jobs:
            if job["count"] == 0:
                continue
            print(f"\nLoading results for {job['name']}...")

            if job["cache_file"].exists():
                results = pd.read_csv(job["cache_file"])
                print(f"Loaded cache: {job['cache_file'].name} ({len(results):,} rows)")
            else:
                print(f"No cache found. Fetching {job['name']} from NBA API...")
                results = self.scraper.get_results_for_date_range(
                    job["fetch_start"], job["fetch_end"], unique_players
                )

            if results is not None and not results.empty:
                all_results.append(results)

        if not all_results:
            print("No game results available")
            return None

        return pd.concat(all_results, ignore_index=True)

    def prepare_training_data(self, start_date=None, end_date=None):
        """
        Full pipeline: load odds → fetch results → merge → label → build features → save.
        """
        print("\n" + "=" * 60)
        print("PREPARING TRAINING DATA")
        print("=" * 60)

        odds_df = self._load_odds(start_date, end_date)
        if odds_df is None:
            return None

        s24 = (odds_df["game_date"] < pd.Timestamp("2025-10-01")).sum()
        s25 = (odds_df["game_date"] >= pd.Timestamp("2025-10-01")).sum()
        print(f"Loaded historical props: {len(odds_df):,}")
        print(f"2024-25 props: {s24:,} | 2025-26 props: {s25:,}")
        print(f"Unique players: {odds_df['player_name'].dropna().nunique():,}")

        results_df = self._load_results(odds_df, end_date)
        if results_df is None:
            return None
        print(f"\nCombined result rows: {len(results_df):,}")

        merged = merge_odds_with_results(odds_df, results_df)
        if merged.empty:
            return None

        labeled = label_results(merged)
        if labeled.empty:
            return None

        features_df = self.build_features(labeled, season=Config.CURRENT_SEASON)
        if features_df is None or features_df.empty:
            print("No features were created")
            return None

        output_file = Config.DATA_DIR / "prepared_training_data.csv"
        features_df.to_csv(output_file, index=False)

        metadata_cols = {"actual_value", "hit", "edge", "line", "market", "player_name", "game_date"}
        feature_count = len([c for c in features_df.columns if c not in metadata_cols])

        print("\nTraining data saved")
        print(f"Location: {output_file}")
        print(f"Rows: {features_df.shape[0]:,} | Columns: {features_df.shape[1]:,} | Features: {feature_count:,}")

        return features_df


if __name__ == "__main__":
    from scripts.data_preparation import DataPreparation

    prep = DataPreparation()
    prep.prepare_training_data(start_date="2024-10-22", end_date="2026-04-12")
