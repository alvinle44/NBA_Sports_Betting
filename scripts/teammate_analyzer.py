"""
Teammate Impact Analyzer

New implementation:
Creates injury/opportunity features for NBA prop models.

Instead of memorizing exact "with/without teammate" splits, this estimates
how much opportunity is removed from the lineup when teammates are injured.

Best for XGBoost features like:
- missing_usage_proxy_sum
- missing_fga_sum
- missing_minutes_sum
- missing_points_sum
- missing_rebounds_sum
- missing_assists_sum
"""

import pandas as pd
import numpy as np
from scripts.config import Config


class TeammateImpactAnalyzer:
    """
    Builds teammate injury features based on missing opportunity.

    This is better than hardcoding:
        curry_out = 1

    Instead it creates general features like:
        missing_fga_sum = 22.5
        missing_usage_proxy_sum = 34.2
        missing_minutes_sum = 35.1

    XGBoost can learn how those affect points, assists, rebounds, etc.
    """

    def __init__(self, nba_scraper):
        self.scraper = nba_scraper
        self.player_stats_cache = {}

    def _get_player_stats(self, player_name, season):
        """
        Get and cache player game logs.
        """
        cache_key = f"{player_name}_{season}"

        if cache_key in self.player_stats_cache:
            return self.player_stats_cache[cache_key]

        try:
            stats = self.scraper.get_player_game_stats(player_name, season)

            if stats is None or stats.empty:
                stats = pd.DataFrame()

            self.player_stats_cache[cache_key] = stats
            return stats

        except Exception:
            self.player_stats_cache[cache_key] = pd.DataFrame()
            return pd.DataFrame()

    def _safe_mean(self, df, col):
        """
        Safely calculate mean for a column.
        """
        if df is None or df.empty or col not in df.columns:
            return 0.0

        return float(pd.to_numeric(df[col], errors="coerce").fillna(0).mean())

    def _calculate_usage_proxy(self, stats_df):
        """
        Estimate offensive usage/opportunity using box score stats.

        True usage rate needs team possessions, but this proxy is useful:
            FGA + 0.44*FTA + TOV + 0.5*AST

        For points props, FGA/FTA/TOV matter most.
        For assists, AST gives ball-handling signal.
        """
        if stats_df is None or stats_df.empty:
            return 0.0

        fga = self._safe_mean(stats_df, "FGA")
        fta = self._safe_mean(stats_df, "FTA")
        tov = self._safe_mean(stats_df, "TOV")
        ast = self._safe_mean(stats_df, "AST")

        return float(fga + 0.44 * fta + tov + 0.5 * ast)

    def _get_teammate_profile(self, teammate_name, season):
        """
        Build the injured teammate's normal role profile.

        This represents the opportunity removed from the lineup.
        """
        stats = self._get_player_stats(teammate_name, season)

        if stats.empty:
            return None

        games_played = len(stats)

        # Require some sample size so random players do not pollute features
        min_games = getattr(Config, "MIN_TEAMMATE_GAMES", 5)
        if games_played < min_games:
            return None

        profile = {
            "name": teammate_name,
            "games_played": games_played,

            # Main opportunity removed
            "minutes": self._safe_mean(stats, "MIN"),
            "points": self._safe_mean(stats, "PTS"),
            "fga": self._safe_mean(stats, "FGA"),
            "fta": self._safe_mean(stats, "FTA"),
            "fg3a": self._safe_mean(stats, "FG3A"),
            "ast": self._safe_mean(stats, "AST"),
            "reb": self._safe_mean(stats, "REB"),
            "oreb": self._safe_mean(stats, "OREB"),
            "dreb": self._safe_mean(stats, "DREB"),
            "tov": self._safe_mean(stats, "TOV"),

            # General offensive role proxy
            "usage_proxy": self._calculate_usage_proxy(stats),
        }

        return profile

    def calculate_impact_features(
        self,
        player_name,
        injured_players,
        market="player_points",
        season="2025-26"
    ):
        """
        Calculate injury features for one player-game row.

        Args:
            player_name: Player being predicted
            injured_players: Injured teammates for that game
            market: player_points, player_assists, player_rebounds, etc.
            season: NBA season

        Returns:
            Dict of ML features.
        """

        features = {
            # Count features
            "key_teammates_injured": 0,
            "num_injured_teammates": 0,

            # Missing opportunity features
            "missing_minutes_sum": 0.0,
            "missing_points_sum": 0.0,
            "missing_fga_sum": 0.0,
            "missing_fta_sum": 0.0,
            "missing_fg3a_sum": 0.0,
            "missing_ast_sum": 0.0,
            "missing_reb_sum": 0.0,
            "missing_oreb_sum": 0.0,
            "missing_dreb_sum": 0.0,
            "missing_tov_sum": 0.0,
            "missing_usage_proxy_sum": 0.0,

            # Top teammate flags
            "top_usage_teammate_out": 0,
            "top_minutes_teammate_out": 0,
            "top_scorer_out": 0,
            "top_rebounder_out": 0,
            "top_assister_out": 0,

            # Market-specific missing opportunity
            "missing_relevant_opportunity": 0.0,

            # Concentration features
            "max_missing_usage_proxy": 0.0,
            "max_missing_minutes": 0.0,
            "max_missing_points": 0.0,
        }

        if not injured_players:
            return features

        injured_players = list(set(injured_players))
        features["num_injured_teammates"] = len(injured_players)

        min_minutes = getattr(Config, "MIN_TEAMMATE_MINUTES", 15)

        teammate_profiles = []

        for teammate in injured_players:
            # Avoid counting the target player as their own injured teammate
            if teammate.lower().strip() == player_name.lower().strip():
                continue

            profile = self._get_teammate_profile(teammate, season)

            if profile is None:
                continue

            # Ignore low-rotation players
            if profile["minutes"] < min_minutes:
                continue

            teammate_profiles.append(profile)

        if not teammate_profiles:
            return features

        features["key_teammates_injured"] = len(teammate_profiles)

        # Sum missing opportunity
        for p in teammate_profiles:
            features["missing_minutes_sum"] += p["minutes"]
            features["missing_points_sum"] += p["points"]
            features["missing_fga_sum"] += p["fga"]
            features["missing_fta_sum"] += p["fta"]
            features["missing_fg3a_sum"] += p["fg3a"]
            features["missing_ast_sum"] += p["ast"]
            features["missing_reb_sum"] += p["reb"]
            features["missing_oreb_sum"] += p["oreb"]
            features["missing_dreb_sum"] += p["dreb"]
            features["missing_tov_sum"] += p["tov"]
            features["missing_usage_proxy_sum"] += p["usage_proxy"]

        # Max/concentration features
        features["max_missing_usage_proxy"] = max(p["usage_proxy"] for p in teammate_profiles)
        features["max_missing_minutes"] = max(p["minutes"] for p in teammate_profiles)
        features["max_missing_points"] = max(p["points"] for p in teammate_profiles)

        # Determine top injured player by category
        top_usage = max(teammate_profiles, key=lambda x: x["usage_proxy"])
        top_minutes = max(teammate_profiles, key=lambda x: x["minutes"])
        top_scorer = max(teammate_profiles, key=lambda x: x["points"])
        top_rebounder = max(teammate_profiles, key=lambda x: x["reb"])
        top_assister = max(teammate_profiles, key=lambda x: x["ast"])

        # These are flags saying a meaningful top-role teammate is out
        if top_usage["usage_proxy"] >= 18:
            features["top_usage_teammate_out"] = 1

        if top_minutes["minutes"] >= 28:
            features["top_minutes_teammate_out"] = 1

        if top_scorer["points"] >= 15:
            features["top_scorer_out"] = 1

        if top_rebounder["reb"] >= 6:
            features["top_rebounder_out"] = 1

        if top_assister["ast"] >= 4:
            features["top_assister_out"] = 1

        # Market-specific opportunity signal
        if market == "player_points":
            # Points mostly depend on shots, free throws, usage, and scoring removed
            features["missing_relevant_opportunity"] = (
                features["missing_fga_sum"]
                + 0.44 * features["missing_fta_sum"]
                + 0.25 * features["missing_points_sum"]
            )

        elif market == "player_assists":
            # Assists depend on ball-handling removed
            features["missing_relevant_opportunity"] = (
                features["missing_ast_sum"]
                + 0.5 * features["missing_usage_proxy_sum"]
            )

        elif market == "player_rebounds":
            # Rebounds depend on missing rebounding/minutes from bigs
            features["missing_relevant_opportunity"] = (
                features["missing_reb_sum"]
                + 0.2 * features["missing_minutes_sum"]
            )

        elif market == "player_threes":
            # Threes depend heavily on missing 3PA
            features["missing_relevant_opportunity"] = (
                features["missing_fg3a_sum"]
                + 0.25 * features["missing_fga_sum"]
            )

        else:
            features["missing_relevant_opportunity"] = features["missing_usage_proxy_sum"]

        # Round everything for cleaner CSV output
        for key, value in features.items():
            if isinstance(value, float):
                features[key] = round(value, 3)

        return features