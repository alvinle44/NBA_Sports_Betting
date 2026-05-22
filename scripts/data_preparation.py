import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pickle
import time
import warnings
import argparse
import json
from pathlib import Path
from scripts.teammate_analyzer import TeammateImpactAnalyzer
# from xgboost import XGBRegressor
# from sklearn.model_selection import TimeSeriesSplit
# from sklearn.metrics import mean_absolute_error, mean_squared_error
# from sklearn.preprocessing import StandardScaler
# from scipy.stats import norm
from scripts.nba_log_scrapper import NBAResultsScraper
from scripts.config import Config
from scripts.rapidapi_injury_scrapper import RapidAPIInjuryScraper 

# Web Scraping
try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False
    print("WARNING: beautifulsoup4 not installed. Run: pip install beautifulsoup4")

# NBA APIc
try:
    from nba_api.stats.endpoints import (
        playergamelog,
        leaguedashteamstats,
        teamgamelogs,
        commonplayerinfo
    )
    from nba_api.stats.static import players, teams
    NBA_API_AVAILABLE = True
except ImportError:
    print("WARNING: nba_api not installed. Run: pip install nba-api")
    NBA_API_AVAILABLE = False

warnings.filterwarnings('ignore')


class DataPreparation:
    """
    Prepares training data with NO DATA LEAKAGE and NO PRESET VALUES.
    
    FIXED:
    - Proper date filtering to prevent leakage
    - Removed all placeholder/preset values
    - Only uses real, calculable features
    - Validates feature consistency
    """
    
    def __init__(self):
        """Initialize with scrapers."""
        self.scraper = NBAResultsScraper()
        self.impact_analyzer = TeammateImpactAnalyzer(self.scraper)
        self.injury_scraper = RapidAPIInjuryScraper()

    def extract_opponent_from_matchup(self, matchup):
        """
        Extract opponent team code from matchup string.
        
        Examples:
            "LAL vs. PHI" → "PHI"
            "GSW @ OKC" → "OKC"
        """
        if not matchup:
            return None
        
        try:
            # Home game format: "TEAM vs. OPPONENT"
            if 'vs.' in matchup:
                opponent = matchup.split('vs.')[1].strip()
                return opponent.split()[0] if opponent else None
            
            # Away game format: "TEAM @ OPPONENT"
            elif '@' in matchup:
                opponent = matchup.split('@')[1].strip()
                return opponent.split()[0] if opponent else None
        
        except Exception:
            return None
        
        return None
    def _get_full_team_name(self, team_code):
        """
        Convert team code to full name for NBA API.
        
        Args:
            team_code: 3-letter team code (e.g., 'LAL', 'GSW')
        
        Returns:
            Full team name (e.g., 'Los Angeles Lakers')
        """
        team_mapping = {
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
        return team_mapping.get(team_code, team_code)



    def calculate_home_away_splits(self, player_stats, stat_col, is_home):
        """
        Calculate player's home vs away performance splits.
        
        Returns None if insufficient data.
        """
        # Split games by home/away (@ symbol indicates away game)
        home_games = player_stats[~player_stats['MATCHUP'].str.contains('@', na=False)]
        away_games = player_stats[player_stats['MATCHUP'].str.contains('@', na=False)]
        
        # Need at least 5 games in each context (increased from 3)
        if len(home_games) < 5 or len(away_games) < 5:
            return None
        
        home_avg = home_games[stat_col].mean()
        away_avg = away_games[stat_col].mean()
        
        return {
            'home_away_split': home_avg - away_avg,
            'context_avg': home_avg if is_home else away_avg,
            'context_consistency': (home_games[stat_col].std() if is_home 
                                else away_games[stat_col].std()),
        }

    def calculate_vs_opponent_history(self, player_stats, opponent, stat_col):

        if not opponent or player_stats.empty:
            return None

        if 'MATCHUP' not in player_stats.columns:
            return None

        if 'GAME_DATE' not in player_stats.columns:
            return None

        if stat_col not in player_stats.columns:
            return None

        player_stats = player_stats.copy()

        player_stats['GAME_DATE'] = pd.to_datetime(
            player_stats['GAME_DATE'],
            errors='coerce'
        )

        player_stats = player_stats.sort_values(
            'GAME_DATE',
            ascending=False
        )

        vs_opponent_games = player_stats[
            player_stats['MATCHUP'].str.contains(
                rf'(\bvs\.\s+{opponent}\b|\@\s+{opponent}\b)',
                regex=True,
                na=False
            )
        ].head(8)

        if len(vs_opponent_games) < 3:
            return None

        weights = np.linspace(
            1.0,
            0.6,
            len(vs_opponent_games)
        )

        values = pd.to_numeric(
            vs_opponent_games[stat_col],
            errors='coerce'
        ).fillna(0)

        vs_opp_avg = np.average(values, weights=weights)

        return {
            'vs_opponent_avg': round(float(vs_opp_avg), 3),
            'vs_opponent_games_count': len(vs_opponent_games),
        }

    def calculate_recent_trend_features(self, player_stats, stat_col):
        """
        Calculate detailed trend features.
        
        Returns None if insufficient data.
        """
        if len(player_stats) < 10:
            return None
        
        # Last 5 games vs games 6-10
        last_5 = player_stats.head(5)[stat_col].mean()
        games_6_10 = player_stats.iloc[5:10][stat_col].mean()
        
        recent_vs_older = last_5 - games_6_10
        peak_recent = player_stats.head(5)[stat_col].max()
        
        # Momentum score: weighted by recency
        weights = np.array([0.3, 0.25, 0.2, 0.15, 0.1])
        last_5_values = player_stats.head(5)[stat_col].values
        momentum_score = np.average(last_5_values, weights=weights)
        
        return {
            'recent_vs_older': recent_vs_older,
            'momentum_score': momentum_score,
            'peak_recent': peak_recent,
        }

    @staticmethod
    def merge_odds_with_results(odds_df, results_df):
        """
        Merge sportsbook odds with actual NBA game results.

        Returns:
            DataFrame containing:
                - odds information
                - actual stat result
                - hit/miss labels
        """

        print("\nMerging odds with game results...")

        odds_df = odds_df.copy()
        results_df = results_df.copy()

        # Normalize results column names
        results_df.columns = results_df.columns.str.upper()

        required_cols = ['PLAYER_NAME', 'GAME_DATE']

        missing_cols = [
            col for col in required_cols
            if col not in results_df.columns
        ]

        if missing_cols:
            print(f"Error: Missing required columns: {missing_cols}")
            print(f"Available columns: {list(results_df.columns)}")
            return pd.DataFrame()

        ## Normalize dates
        odds_df["game_date"] = pd.to_datetime(
            odds_df["game_date"],
            errors="coerce",
            utc=True
        ).dt.tz_convert(None).dt.strftime("%Y-%m-%d")

        results_df["GAME_DATE"] = pd.to_datetime(
            results_df["GAME_DATE"],
            errors="coerce"
        ).dt.strftime("%Y-%m-%d")

        # Drop invalid dates
        odds_df = odds_df.dropna(subset=["game_date"])
        results_df = results_df.dropna(subset=["GAME_DATE"])

        # Normalize names
        odds_df['normalized_player_name'] = (
            odds_df['player_name']
            .astype(str)
            .str.lower()
            .str.strip()
        )

        results_df['normalized_player_name'] = (
            results_df['PLAYER_NAME']
            .astype(str)
            .str.lower()
            .str.strip()
        )

        # Create merge keys
        odds_df['merge_key'] = (
            odds_df['normalized_player_name']
            + '_'
            + odds_df['game_date']
        )

        results_df['merge_key'] = (
            results_df['normalized_player_name']
            + '_'
            + results_df['GAME_DATE']
        )

        from scripts.config import Config

        merged_frames = []

        markets = sorted(odds_df['market'].dropna().unique())

        for market in markets:

            market_odds = odds_df[
                odds_df['market'] == market
            ].copy()

            stat_col = Config.MARKET_TO_STAT.get(market)

            if stat_col is None:
                print(f"Skipping {market}: no stat mapping found")
                continue

            stat_col = stat_col.upper()

            if stat_col not in results_df.columns:
                print(
                    f"Skipping {market}: "
                    f"'{stat_col}' not found in results"
                )
                continue

            # Keep only needed columns
            results_subset = results_df[
                ['merge_key', stat_col]
            ].copy()

            market_merged = market_odds.merge(
                results_subset,
                on='merge_key',
                how='left'
            )

            market_merged = market_merged.rename(
                columns={
                    stat_col: 'actual_value'
                }
            )

            market_merged['actual_value'] = pd.to_numeric(
                market_merged['actual_value'],
                errors='coerce'
            )

            merged_frames.append(market_merged)

        if not merged_frames:
            print("No markets were successfully merged")
            return pd.DataFrame()

        merged = pd.concat(
            merged_frames,
            ignore_index=True
        )
        dedupe_cols = [
            "player_name",
            "game_date",
            "market",
            "line",
            "bet_type",
            "bookmaker",
            "price",
            "odds",
            "american_odds",
            "commence_time",
            "last_update",
            "timestamp",
        ]

        dedupe_cols = [c for c in dedupe_cols if c in merged.columns]

        before = len(merged)

        merged = merged.drop_duplicates(
            subset=dedupe_cols,
            keep="last"
        )

        print(f"Removed exact duplicate odds rows: {before - len(merged):,}")

        # Remove rows without actual results
        before_rows = len(merged)

        merged = merged.dropna(
            subset=['actual_value']
        )

        after_rows = len(merged)

        dropped_rows = before_rows - after_rows

        drop_pct = (
            dropped_rows / before_rows * 100
            if before_rows > 0 else 0
        )

        print(f"Merged props: {after_rows:,}")
        print(
            f"Dropped props without results: "
            f"{dropped_rows:,} ({drop_pct:.1f}%)"
        )

        print("\nMerge results by market:")

        market_counts = (
            merged['market']
            .value_counts()
            .sort_values(ascending=False)
        )

        for market, count in market_counts.items():
            print(f"  {market}: {count:,}")

        return merged
        
    @staticmethod
    def label_results(merged_df):
        """
        Label sportsbook prop outcomes.

        Adds:
            - hit: 1 if bet wins, 0 if loses
            - edge: actual_value - sportsbook line

        Returns:
            DataFrame with labels.
        """

        print("\nLabeling prop results...")

        if merged_df.empty:
            print("No data to label")
            return pd.DataFrame()

        labeled = merged_df.copy()

        # Ensure numeric values
        labeled['actual_value'] = pd.to_numeric(
            labeled['actual_value'],
            errors='coerce'
        )

        labeled['line'] = pd.to_numeric(
            labeled['line'],
            errors='coerce'
        )

        # Remove rows with missing values
        before_rows = len(labeled)

        labeled = labeled.dropna(
            subset=['actual_value', 'line']
        )

        dropped_rows = before_rows - len(labeled)

        if labeled.empty:
            print("All rows removed due to missing values")
            return pd.DataFrame()

        # Normalize bet type
        labeled['bet_type'] = (
            labeled.get('bet_type', 'Over')
            .astype(str)
            .str.lower()
            .str.strip()
        )

        labeled['edge'] = (
            labeled['actual_value']
            - labeled['line']
        )

        # Create hit label
        labeled['hit'] = np.where(
            (
                (labeled['bet_type'] == 'over')
                & (labeled['actual_value'] > labeled['line'])
            )
            |
            (
                (labeled['bet_type'] == 'under')
                & (labeled['actual_value'] < labeled['line'])
            ),
            1,
            0
        )

        print(f"Labeled props: {len(labeled):,}")

        if dropped_rows > 0:
            print(f"Dropped rows with missing values: {dropped_rows:,}")

        hit_rate = labeled['hit'].mean() * 100

        print(f"Historical hit rate: {hit_rate:.2f}%")

        return labeled

    def build_features(self, labeled_df, season="2025-26"):
        """
        Build training features for player prop models.

        Uses only games before the target game date to avoid data leakage.
        """

        print("\nBuilding features...")

        all_game_stats = []

        for file_name in ["game_results_2024-25.csv", "game_results_2025-26.csv"]:
            file_path = Config.DATA_DIR / file_name

            if file_path.exists():
                df = pd.read_csv(file_path)
                all_game_stats.append(df)
                print(f"Loaded {len(df):,} rows from {file_name}")

        if not all_game_stats:
            print("No cached game results found")
            return pd.DataFrame()

        all_stats_df = pd.concat(all_game_stats, ignore_index=True)

        all_stats_df["GAME_DATE"] = pd.to_datetime(
            all_stats_df["GAME_DATE"],
            errors="coerce"
        )

        all_stats_df = all_stats_df.dropna(subset=["GAME_DATE"])
        
        cached_players = set(all_stats_df["PLAYER_NAME"].dropna().unique())
        defense_file = Config.DATA_DIR / "team_defense_features.csv"

        if defense_file.exists():
            defense_df = pd.read_csv(defense_file)

            defense_df["game_date"] = pd.to_datetime(
                defense_df["game_date"],
                errors="coerce"
            ).dt.strftime("%Y-%m-%d")

            defense_lookup = {
                (row["team_name"], row["game_date"], row["season"]): row
                for _, row in defense_df.iterrows()
            }

            print(f"Loaded defense cache: {len(defense_lookup):,} rows")
        else:
            defense_lookup = {}
            print("No defense cache found")

        checkpoint_file = Config.DATA_DIR / "features_checkpoint.csv"
        processed_indices_file = Config.DATA_DIR / "processed_indices.txt"

        features_list = []
        processed_indices = set()

        if checkpoint_file.exists() and processed_indices_file.exists():
            try:
                checkpoint_df = pd.read_csv(checkpoint_file)
                features_list = checkpoint_df.to_dict("records")

                with open(processed_indices_file, "r") as f:
                    processed_indices = {
                        int(line.strip())
                        for line in f
                        if line.strip()
                    }

                print(f"Resuming from checkpoint: {len(features_list):,} rows")

            except Exception as e:
                print(f"Could not load checkpoint: {e}")
                features_list = []
                processed_indices = set()

        total_props = len(labeled_df)
        remaining = total_props - len(processed_indices)

        print(f"Total props: {total_props:,}")
        print(f"Already processed: {len(processed_indices):,}")
        print(f"Remaining: {remaining:,}")

        cache_hits = 0
        api_calls = 0
        api_failures = 0
        leakage_warnings = 0
        last_save_count = len(features_list)
        defense_cache = {}

        for idx, row in labeled_df.iterrows():

            if idx in processed_indices:
                continue

            current_count = len(features_list)

            if current_count % 100 == 0 and current_count != last_save_count:
                pct = (current_count / remaining * 100) if remaining else 0
                player_preview = str(row.get("player_name", ""))[:20]

                print(
                    f"Processed {current_count:,}/{remaining:,} "
                    f"({pct:.2f}%) | Player: {player_preview:<20} | "
                    f"Cache hits: {cache_hits} | Leakage warnings: {leakage_warnings}"
                )

            if current_count % 1000 == 0 and current_count > last_save_count:
                try:
                    pd.DataFrame(features_list).to_csv(checkpoint_file, index=False)

                    with open(processed_indices_file, "w") as f:
                        for proc_idx in processed_indices:
                            f.write(f"{proc_idx}\n")

                    print(f"Checkpoint saved at {current_count:,} rows")
                    last_save_count = current_count

                except Exception as e:
                    print(f"Checkpoint save failed: {e}")

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

            player_stats = pd.DataFrame()

            if player in cached_players:
                player_stats = all_stats_df[
                    all_stats_df["PLAYER_NAME"] == player
                ].copy()

                cache_hits += 1

            else:
                api_calls += 1

                for attempt in range(3):
                    try:
                        player_stats = self.scraper.get_player_game_stats(
                            player,
                            season
                        )

                        if player_stats is not None and not player_stats.empty:
                            player_stats["PLAYER_NAME"] = player
                            player_stats["GAME_DATE"] = pd.to_datetime(
                                player_stats["GAME_DATE"],
                                errors="coerce"
                            )

                            all_stats_df = pd.concat(
                                [all_stats_df, player_stats],
                                ignore_index=True
                            )

                            cached_players.add(player)
                            break

                    except requests.exceptions.Timeout:
                        if attempt < 2:
                            time.sleep((attempt + 1) * 20)
                        else:
                            api_failures += 1

                    except Exception:
                        api_failures += 1
                        break

            if player_stats is None or player_stats.empty:
                continue

            player_stats["GAME_DATE"] = pd.to_datetime(
                player_stats["GAME_DATE"],
                errors="coerce"
            )

            player_stats = player_stats.dropna(subset=["GAME_DATE"])

            player_stats = player_stats[
                player_stats["GAME_DATE"] < game_date
            ].sort_values("GAME_DATE", ascending=False)

            if player_stats.empty or len(player_stats) < 10:
                continue

            if player_stats.iloc[0]["GAME_DATE"] >= game_date:
                leakage_warnings += 1
                continue

            if stat_col not in player_stats.columns:
                continue

            stat_values = pd.to_numeric(
                player_stats[stat_col],
                errors="coerce"
            )

            if stat_values.dropna().shape[0] < 10:
                continue

            last_5 = stat_values.head(5)
            last_10 = stat_values.head(10)

            avg_last_5 = last_5.mean()
            avg_last_10 = last_10.mean()
            avg_season = stat_values.mean()

            matchup = player_stats.iloc[0].get("MATCHUP", "")
            opponent = self.extract_opponent_from_matchup(matchup)
            is_home = 0 if "@" in str(matchup) else 1

            last_game_date = player_stats.iloc[0]["GAME_DATE"]
            prev_game_date = player_stats.iloc[1]["GAME_DATE"]
            days_rest = max((last_game_date - prev_game_date).days, 0)
            back_to_back = int(days_rest <= 1)

            avg_minutes = pd.to_numeric(
                player_stats["MIN"],
                errors="coerce"
            ).mean()

            recent_minutes = pd.to_numeric(
                player_stats.head(5)["MIN"],
                errors="coerce"
            ).mean()

            minutes_trend = recent_minutes - avg_minutes

            usage_rate_proxy = None

            if {"FGA", "FTA"}.issubset(player_stats.columns):
                fga = pd.to_numeric(player_stats["FGA"], errors="coerce")
                fta = pd.to_numeric(player_stats["FTA"], errors="coerce")
                usage_rate_proxy = (fga + 0.44 * fta).mean()

            game_season = "2024-25" if game_date < pd.Timestamp("2025-10-01") else "2025-26"

            opp_stats = None

            if opponent and defense_lookup:
                date_key = game_date.strftime("%Y-%m-%d")
                cache_key = (opponent, date_key, game_season)

                cached_defense = defense_lookup.get(cache_key)

                if cached_defense is not None:
                    opp_stats = {
                        "def_rating_weighted": cached_defense.get("opponent_def_weighted"),
                        "pace": cached_defense.get("pace_factor"),
                        "opp_pts_weighted": cached_defense.get("opponent_pts_allowed"),
                        "opp_reb_rate": cached_defense.get("opponent_reb_rate"),
                        "opp_tov_rate": cached_defense.get("opponent_tov_rate"),
                        "opp_three_def": cached_defense.get("opponent_three_def"),
                    }

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
                    season=game_season
                )
            except Exception:
                impact_features = None

            trend_l3 = 0.0

            if len(last_5) >= 3:
                y = last_5.tail(3).values
                x = np.arange(len(y))
                trend_l3 = np.polyfit(x, y, 1)[0]

            last_week_games = player_stats[
                player_stats["GAME_DATE"] >= game_date - pd.Timedelta(days=7)
            ]

            home_away_features = self.calculate_home_away_splits(
                player_stats,
                stat_col,
                is_home
            )

            vs_opp_features = self.calculate_vs_opponent_history(
                player_stats,
                opponent,
                stat_col
            )

            trend_features = self.calculate_recent_trend_features(
                player_stats,
                stat_col
            )

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
                "back_to_back": back_to_back,
                "avg_minutes": avg_minutes,
                "minutes_trend": minutes_trend,
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
                    features["opponent_three_def"] = opp_stats.get("opp_three_def")

            if impact_features and impact_features.get("missing_minutes_sum", 0) > 0:
                features.update(impact_features)

            if home_away_features:
                features.update(home_away_features)

            if vs_opp_features:
                features.update(vs_opp_features)

            if trend_features:
                features.update(trend_features)

            if market == "player_points":
                if "FG_PCT" in player_stats.columns:
                    fg_recent = pd.to_numeric(
                        player_stats.head(5)["FG_PCT"],
                        errors="coerce"
                    )

                    fg_old = pd.to_numeric(
                        player_stats.iloc[5:10]["FG_PCT"],
                        errors="coerce"
                    )

                    features["fg_pct_L5"] = fg_recent.mean()
                    features["fg_pct_trend"] = fg_recent.mean() - fg_old.mean()

                if "FTA" in player_stats.columns:
                    features["fta_rate_L5"] = pd.to_numeric(
                        player_stats.head(5)["FTA"],
                        errors="coerce"
                    ).mean()

                if "FGA" in player_stats.columns and "PTS" in player_stats.columns:
                    fga_l5 = pd.to_numeric(
                        player_stats.head(5)["FGA"],
                        errors="coerce"
                    ).mean()

                    if fga_l5 > 0:
                        pts_l5 = pd.to_numeric(
                            player_stats.head(5)["PTS"],
                            errors="coerce"
                        ).mean()

                        features["efficiency_last_5"] = pts_l5 / fga_l5

                    features["pts_per_minute"] = (
                        pd.to_numeric(player_stats["PTS"], errors="coerce").mean()
                        / (avg_minutes + 0.1)
                    )

            elif market == "player_assists":
                if {"AST", "TOV"}.issubset(player_stats.columns):
                    ast_mean = pd.to_numeric(
                        player_stats["AST"],
                        errors="coerce"
                    ).mean()

                    tov_mean = pd.to_numeric(
                        player_stats["TOV"],
                        errors="coerce"
                    ).mean()

                    if tov_mean > 0.1:
                        features["ast_to_tov_ratio"] = ast_mean / tov_mean

                    features["ast_per_minute"] = ast_mean / (avg_minutes + 0.1)

                if opp_stats is not None and opp_stats.get("pace") is not None:
                    features["pace_factor_deviation"] = opp_stats["pace"] - 100.0

            elif market == "player_rebounds":
                if "REB" in player_stats.columns:
                    reb_mean = pd.to_numeric(
                        player_stats["REB"],
                        errors="coerce"
                    ).mean()

                    features["reb_per_minute"] = reb_mean / (avg_minutes + 0.1)

                    if {"OREB", "DREB"}.issubset(player_stats.columns) and reb_mean > 0.1:
                        features["oreb_rate"] = pd.to_numeric(
                            player_stats["OREB"],
                            errors="coerce"
                        ).mean() / reb_mean

                        features["dreb_rate"] = pd.to_numeric(
                            player_stats["DREB"],
                            errors="coerce"
                        ).mean() / reb_mean

                        features["oreb_L5"] = pd.to_numeric(
                            player_stats.head(5)["OREB"],
                            errors="coerce"
                        ).mean()

                        features["dreb_L5"] = pd.to_numeric(
                            player_stats.head(5)["DREB"],
                            errors="coerce"
                        ).mean()

            elif market == "player_threes":
                if {"FG3A", "FG3M"}.issubset(player_stats.columns):
                    fg3a_l5 = pd.to_numeric(
                        player_stats.head(5)["FG3A"],
                        errors="coerce"
                    ).mean()

                    fg3m_l5 = pd.to_numeric(
                        player_stats.head(5)["FG3M"],
                        errors="coerce"
                    ).mean()

                    if fg3a_l5 > 0:
                        features["three_point_pct_L5"] = fg3m_l5 / fg3a_l5
                        features["three_point_attempts_L5"] = fg3a_l5

            features_list.append(features)
            processed_indices.add(idx)

        features_df = pd.DataFrame(features_list)

        print(f"\nBuilt features for {len(features_df):,} props")
        print(f"Cache hits: {cache_hits:,}")
        print(f"API calls: {api_calls:,}")
        print(f"API failures: {api_failures:,}")
        print(f"Leakage warnings: {leakage_warnings:,}")

        if not features_df.empty:
            self.validate_feature_consistency(features_df)

        for path in [checkpoint_file, processed_indices_file]:
            if path.exists():
                path.unlink()

        return features_df
    
    def validate_feature_consistency(self, features_df):
        """
        Validate feature quality.

        Checks for:
            - constant features
            - sparse features
            - low variance features
            - excessive missing values
        """

        print("\n" + "=" * 60)
        print("FEATURE VALIDATION")
        print("=" * 60)

        if features_df.empty:
            print("Feature dataframe is empty")
            return

        metadata_cols = {
            'actual_value',
            'hit',
            'edge',
            'line',
            'market',
            'player_name',
            'game_date'
        }

        feature_cols = [
            col for col in features_df.columns
            if col not in metadata_cols
        ]

        issues = []

        for col in feature_cols:

            series = pd.to_numeric(
                features_df[col],
                errors='coerce'
            )

            total_rows = len(series)

            if total_rows == 0:
                continue

            non_null = series.dropna()

            if non_null.empty:
                issues.append(
                    f"{col}: all values missing"
                )
                continue

            unique_vals = non_null.nunique()

            if unique_vals <= 1:
                issues.append(
                    f"{col}: constant feature"
                )
                continue

            missing_pct = (
                series.isna().sum() / total_rows * 100
            )

            if missing_pct > 50:
                issues.append(
                    f"{col}: {missing_pct:.1f}% missing"
                )

            zero_pct = (
                (non_null == 0).sum() / len(non_null) * 100
            )

            if zero_pct > 85:
                issues.append(
                    f"{col}: {zero_pct:.1f}% zeros"
                )

            variance = non_null.var()

            if variance < 1e-4:
                issues.append(
                    f"{col}: extremely low variance ({variance:.6f})"
                )

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
            print(
                f"Markets: "
                f"{features_df['market'].nunique()}"
            )

        print("=" * 60)
    
    def _get_injuries_from_rapidapi(self, team_code, game_date):
        """
        Load injured players for a team on a specific date.

        Args:
            team_code: Team abbreviation, e.g. "GSW"
            game_date: date or datetime object

        Returns:
            List of injured player names.
        """

        if not team_code or not game_date:
            return []

        try:
            date_str = pd.to_datetime(game_date).strftime("%Y-%m-%d")
            injury_dict = self.injury_scraper.load_injuries_for_game_date(date_str)

            if not injury_dict:
                return []

            full_team_name = self._get_full_team_name(team_code)

            team_injuries = (
                injury_dict.get(team_code)
                or injury_dict.get(full_team_name)
                or []
            )

            return [
                player.get("name")
                for player in team_injuries
                if isinstance(player, dict) and player.get("name")
            ]

        except Exception as e:
            print(f"Error loading injuries for {team_code}: {str(e)[:80]}")
            return []
    
    def prepare_training_data(self, start_date, end_date):
        """
        Prepare training data from the combined historical odds file.

        Pipeline:
            1. Load historical odds
            2. Load or fetch actual NBA results
            3. Merge odds with results
            4. Label hit/miss outcomes
            5. Build ML features
            6. Save prepared training data
        """

        print("\n" + "=" * 60)
        print("PREPARING TRAINING DATA")
        print("=" * 60)

        odds_file = Config.DATA_DIR / "historical_odds_combined.csv"

        if not odds_file.exists():
            print(f"Combined odds file not found: {odds_file}")
            return None

        odds_df = pd.read_csv(odds_file)

        if odds_df.empty:
            print("Combined odds file is empty")
            return None

        odds_df["game_date"] = pd.to_datetime(
            odds_df["game_date"],
            errors="coerce"
        )

        odds_df["game_date"] = pd.to_datetime(
            odds_df["game_date"],
            errors="coerce",
            utc=True
        ).dt.tz_convert(None)

        odds_df = odds_df.dropna(subset=["game_date"])

        if start_date:
            start_date = pd.to_datetime(start_date)
            odds_df = odds_df[odds_df["game_date"] >= start_date]

        if end_date:
            end_date = pd.to_datetime(end_date)
            odds_df = odds_df[odds_df["game_date"] <= end_date]

        if odds_df.empty:
            print("No odds found in the requested date range")
            return None

        print(f"Loaded historical props: {len(odds_df):,}")
        season_2024_25_count = (
            odds_df["game_date"] < pd.Timestamp("2025-10-01")
        ).sum()

        season_2025_26_count = (
            odds_df["game_date"] >= pd.Timestamp("2025-10-01")
        ).sum()

        print(f"2024-25 props: {season_2024_25_count:,}")
        print(f"2025-26 props: {season_2025_26_count:,}")

        unique_players = (
            odds_df["player_name"]
            .dropna()
            .unique()
        )

        print(f"Unique players: {len(unique_players):,}")

        all_results = []

        season_jobs = [
            {
                "name": "2024-25",
                "count": season_2024_25_count,
                "cache_file": Config.DATA_DIR / "game_results_2024-25.csv",
                "fetch_start": "2024-10-22",
                "fetch_end": "2025-06-30",
            },
            {
                "name": "2025-26",
                "count": season_2025_26_count,
                "cache_file": Config.DATA_DIR / "game_results_2025-26.csv",
                "fetch_start": "2025-10-21",
                "fetch_end": end_date.strftime("%Y-%m-%d") if end_date else None,
            },
        ]

        for job in season_jobs:

            if job["count"] == 0:
                continue

            print(f"\nLoading results for {job['name']}...")

            if job["cache_file"].exists():
                results = pd.read_csv(job["cache_file"])
                print(f"Loaded cache: {job['cache_file'].name} ({len(results):,} rows)")

            else:
                fetch_end = job["fetch_end"]

                if fetch_end is None:
                    fetch_end = datetime.now().strftime("%Y-%m-%d")

                print(f"No cache found. Fetching {job['name']} from NBA API...")

                results = self.scraper.get_results_for_date_range(
                    job["fetch_start"],
                    fetch_end,
                    unique_players
                )

            if results is not None and not results.empty:
                all_results.append(results)
                print(f"Loaded {len(results):,} result rows for {job['name']}")

        if not all_results:
            print("No game results available")
            return None

        results_df = pd.concat(
            all_results,
            ignore_index=True
        )

        print(f"\nCombined result rows: {len(results_df):,}")

        merged = self.merge_odds_with_results(
            odds_df,
            results_df
        )

        if merged.empty:
            print("No merged odds/results rows")
            return None

        labeled = self.label_results(merged)

        if labeled.empty:
            print("No labeled rows")
            return None

        features_df = self.build_features(
            labeled,
            season=Config.CURRENT_SEASON
        )

        if features_df is None or features_df.empty:
            print("No features were created")
            return None

        output_file = Config.DATA_DIR / "prepared_training_data.csv"

        features_df.to_csv(
            output_file,
            index=False
        )

        metadata_cols = {
            "actual_value",
            "hit",
            "edge",
            "line",
            "market",
            "player_name",
            "game_date"
        }

        feature_count = len([
            col for col in features_df.columns
            if col not in metadata_cols
        ])

        print("\nTraining data saved")
        print(f"Location: {output_file}")
        print(f"Rows: {features_df.shape[0]:,}")
        print(f"Columns: {features_df.shape[1]:,}")
        print(f"Feature count: {feature_count:,}")

        return features_df

if __name__ == "__main__":

    from scripts.data_preparation import DataPreparation

    prep = DataPreparation()

    prep.prepare_training_data(
        start_date="2024-10-22",
        end_date="2026-04-12"
    )