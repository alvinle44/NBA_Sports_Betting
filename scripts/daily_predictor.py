import pickle
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from scripts.config import Config
from scripts.nba_log_scrapper import NBAResultsScraper
from scripts.teammate_analyzer import TeammateImpactAnalyzer
from scripts.rapidapi_injury_scrapper import RapidAPIInjuryScraper
from scripts.feature_utils import (
    extract_opponent_from_matchup,
    get_full_team_name,
    calculate_home_away_splits,
    calculate_vs_opponent_history,
    calculate_recent_trend_features,
    build_market_features,
)

warnings.filterwarnings('ignore')

BANKROLL_FILE = Path(__file__).parent.parent / "data" / "bankroll.json"


def load_bankroll_amount() -> float:
    """Return current bankroll from bankroll.json, defaulting to 1000."""
    if BANKROLL_FILE.exists():
        import json
        with open(BANKROLL_FILE) as f:
            return float(json.load(f).get("current", 1000))
    return 1000.0


def recommended_units(confidence: float) -> float:
    """Map confidence % to a unit recommendation capped at 3u."""
    if confidence >= 67:   return 3.0
    elif confidence >= 64: return 2.0
    elif confidence >= 61: return 1.5
    elif confidence >= 58: return 1.0
    elif confidence >= 55: return 0.5
    elif confidence >= 52: return 0.25
    return 0.0


def calculate_bet_size(
    bankroll: float,
    confidence: float,
    american_odds: float,
    unit_percent: float = 0.01,
    kelly_fraction: float = 0.25,
) -> dict:
    """
    Quarter-Kelly bet sizing capped at 5% of bankroll.

    Returns bet_amount, units, kelly_percent.
    """
    p = confidence / 100.0
    q = 1.0 - p

    if american_odds > 0:
        decimal_odds = american_odds / 100 + 1
    else:
        decimal_odds = 100 / abs(american_odds) + 1

    b = decimal_odds - 1
    kelly = (b * p - q) / b if b > 0 else 0.0
    kelly = max(kelly, 0.0)
    kelly *= kelly_fraction
    kelly = min(kelly, 0.05)

    bet_amount = bankroll * kelly
    unit_size = bankroll * unit_percent
    units = bet_amount / unit_size if unit_size > 0 else 0.0

    return {
        "bet_amount": round(bet_amount, 2),
        "units": round(units, 2),
        "kelly_percent": round(kelly * 100, 2),
    }


# Columns that must never be passed to the model
_PREDICT_EXCLUDE = {
    'actual_value', 'hit', 'edge', 'market', 'player_name', 'game_date',
    'bookmaker', 'bet_type', 'odds', 'odds_over', 'odds_under', 'american_odds',
    'price', 'timestamp', 'last_update', 'commence_time',
    'normalized_player_name', 'merge_key', 'implied_prob',
}


class DailyPredictor:

    def __init__(self):
        self.scraper = NBAResultsScraper()
        self.impact_analyzer = TeammateImpactAnalyzer(self.scraper)
        self.injury_scraper = RapidAPIInjuryScraper()
        self.models = self._load_models()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_models(self):
        models = {}
        print("\nLoading models...")

        for market in Config.MARKETS:
            model_file = Config.MODELS_DIR / f"{market}_model.pkl"
            if model_file.exists():
                try:
                    with open(model_file, 'rb') as f:
                        models[market] = pickle.load(f)
                    feat_count = len(models[market].get('features', []))
                    print(f"  Loaded {market} ({feat_count} features)")
                except Exception as e:
                    print(f"  Error loading {market}: {e}")
            else:
                print(f"  Not found: {model_file}")

        if not models:
            raise ValueError("No models loaded. Run model_trainer.py first.")

        print(f"Loaded {len(models)} models\n")
        return models

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

    def run_daily_predictions(self, save_to_ongoing=True):
        print("\n" + "=" * 70)
        print("NBA PROPS DAILY PREDICTIONS")
        print("=" * 70)
        print(f"Date: {datetime.now().strftime('%Y-%m-%d %I:%M %p')}")

        # 1. Fetch today's props
        print("\nStep 1: Fetching today's props...")
        props_df = self._fetch_todays_props()
        if props_df.empty:
            print("No props found for today")
            return pd.DataFrame()
        print(f"Found {len(props_df)} props")

        # 2. Fetch current injuries
        print("\nStep 2: Fetching current injuries...")
        current_injuries = self._get_current_injuries()

        # 3. Build prediction features (mirrors data_preparation.py)
        print("\nStep 3: Building prediction features...")
        features_df = self._build_prediction_features(props_df, current_injuries)
        if features_df.empty:
            print("No prediction features could be built")
            return pd.DataFrame()
        print(f"Built features for {len(features_df)} props")

        # 4. Make predictions
        print("\nStep 4: Making predictions...")
        predictions = self._make_predictions(features_df)
        if predictions.empty:
            print("No predictions generated")
            return pd.DataFrame()

        # 5. Calculate EV and Kelly sizing
        print("\nStep 5: Calculating EV and Kelly bet sizes...")
        predictions = self._calculate_ev(predictions)
        predictions = self._add_kelly_sizing(predictions)
        predictions['recommendation'] = predictions.apply(self._make_recommendation, axis=1)

        positive_ev = predictions[predictions['expected_value'] > 0]
        print(f"\nPREDICTION SUMMARY")
        print(f"  Total props analyzed: {len(predictions)}")
        print(f"  Positive EV bets: {len(positive_ev)}")
        if len(positive_ev):
            print(f"  Avg edge: {positive_ev['edge_vs_line'].mean():.2f}")
            print(f"  Avg confidence: {positive_ev['confidence'].mean():.1f}%")

        # 6. Save and display
        if save_to_ongoing and not props_df.empty:
            self._save_to_ongoing_collection(props_df)

        self._display_top_picks(predictions[predictions['expected_value'] > 0])

        return predictions

    # ------------------------------------------------------------------
    # Props fetching
    # ------------------------------------------------------------------

    def _fetch_todays_props(self):
        from scripts.live_odds_scraper import LiveOddsScraper
        try:
            scraper = LiveOddsScraper()
            props_df = scraper.get_todays_props()
            if props_df.empty:
                print("  No props available yet")
            return props_df
        except Exception as e:
            print(f"  Error fetching props: {e}")
            return pd.DataFrame()

    def _get_current_injuries(self):
        try:
            injuries = self.injury_scraper.get_current_injuries()
            if injuries:
                print(f"  Found injuries for {len(injuries)} teams")
            else:
                print("  No injuries reported today")
            return injuries or {}
        except Exception as e:
            print(f"  Could not fetch injuries: {e}")
            return {}

    # ------------------------------------------------------------------
    # Defense data (local CSV only — no live API calls)
    # ------------------------------------------------------------------

    def _load_defense_lookup(self):
        """Load team_defense_features.csv; return most-recent row per (team, season)."""
        defense_file = Config.DATA_DIR / "team_defense_features.csv"
        if not defense_file.exists():
            print("  No local defense data found (team_defense_features.csv)")
            return {}

        df = pd.read_csv(defense_file)
        df['game_date'] = pd.to_datetime(df['game_date'], errors='coerce')
        df = df.dropna(subset=['game_date']).sort_values('game_date', ascending=False)

        lookup = {}
        for _, row in df.iterrows():
            key = (str(row.get('team_abbrev', '')), str(row.get('season', '')))
            if key not in lookup:
                lookup[key] = row

        print(f"  Loaded defense data for {len(lookup)} team-seasons")
        return lookup

    def _resolve_defense_stats(self, defense_lookup, opponent, season):
        """Return the most recent defense stats dict for an opponent, or None."""
        if not opponent or not defense_lookup:
            return None

        row = defense_lookup.get((opponent, season))
        if row is None:
            return None

        # Support both old and new column names for reb/tov
        reb_rate = row.get('opponent_reb_rate') if row.get('opponent_reb_rate') is not None \
            else row.get('opponent_rebounds_allowed')
        tov_rate = row.get('opponent_tov_rate') if row.get('opponent_tov_rate') is not None \
            else row.get('opponent_turnovers_forced')

        return {
            'opp_pts_weighted': row.get('opponent_pts_allowed'),
            'opp_reb_rate': reb_rate,
            'opp_tov_rate': tov_rate,
            'opp_three_def': row.get('opponent_three_def'),
        }

    # ------------------------------------------------------------------
    # Feature building — mirrors data_preparation.py::build_features
    # ------------------------------------------------------------------

    def _fetch_player_game_logs(self, players):
        """
        Return dict: player_name → game log DataFrame.
        Uses a daily cache file to avoid redundant API calls.
        """
        today = datetime.now().strftime('%Y-%m-%d')
        cache_path = Config.DATA_DIR / f"game_logs_cache_{today}.pkl"

        cached = {}
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    cached = pickle.load(f)
            except Exception:
                cached = {}

        missing = [p for p in players if p not in cached]
        if missing:
            print(f"  Fetching game logs for {len(missing)} players...")

        for player in missing:
            try:
                logs = self.scraper.get_player_game_stats(player, Config.CURRENT_SEASON)
                cached[player] = logs if (logs is not None and not logs.empty) else pd.DataFrame()
            except Exception:
                cached[player] = pd.DataFrame()

        if missing:
            try:
                with open(cache_path, 'wb') as f:
                    pickle.dump(cached, f)
            except Exception:
                pass

        return cached

    def _build_prediction_features(self, props_df, current_injuries):
        """
        Build one feature row per prop, using only historical data (no future leakage).

        The feature set matches what data_preparation.py::build_features produces so
        the saved imputer and feature list align exactly.
        """
        today = pd.Timestamp(datetime.now().date())
        season = Config.CURRENT_SEASON
        game_season = (
            "2024-25" if today < pd.Timestamp("2025-10-01") else "2025-26"
        )

        defense_lookup = self._load_defense_lookup()

        unique_players = props_df['player_name'].dropna().unique().tolist()
        game_logs = self._fetch_player_game_logs(unique_players)

        rows = []
        skipped = 0

        for _, prop_row in props_df.iterrows():
            player = prop_row['player_name']
            market = prop_row['market']
            line = prop_row['line']
            stat_col = Config.MARKET_TO_STAT.get(market)

            if not stat_col:
                skipped += 1
                continue

            player_stats = game_logs.get(player, pd.DataFrame())
            if player_stats.empty:
                skipped += 1
                continue

            player_stats = player_stats.copy()
            player_stats['GAME_DATE'] = pd.to_datetime(
                player_stats['GAME_DATE'], errors='coerce'
            )
            player_stats = player_stats.dropna(subset=['GAME_DATE'])
            player_stats = player_stats[
                player_stats['GAME_DATE'] < today
            ].sort_values('GAME_DATE', ascending=False)

            if len(player_stats) < 5 or stat_col not in player_stats.columns:
                skipped += 1
                continue

            stat_values = pd.to_numeric(player_stats[stat_col], errors='coerce')
            if stat_values.dropna().shape[0] < 5:
                skipped += 1
                continue

            last_5 = stat_values.head(5)
            avg_last_5 = last_5.mean()
            avg_last_10 = stat_values.head(10).mean() if len(stat_values) >= 10 else avg_last_5
            avg_season = stat_values.mean()

            matchup = player_stats.iloc[0].get('MATCHUP', '')
            opponent = extract_opponent_from_matchup(matchup)
            is_home = 0 if '@' in str(matchup) else 1

            last_game_date = player_stats.iloc[0]['GAME_DATE']
            days_rest = None
            back_to_back = None
            if len(player_stats) >= 2:
                prev_game_date = player_stats.iloc[1]['GAME_DATE']
                days_rest = max((last_game_date - prev_game_date).days, 0)
                back_to_back = int(days_rest <= 1)

            avg_minutes = pd.to_numeric(player_stats['MIN'], errors='coerce').mean()
            recent_minutes = pd.to_numeric(
                player_stats.head(5)['MIN'], errors='coerce'
            ).mean()
            minutes_trend = (
                recent_minutes - avg_minutes
                if pd.notna(recent_minutes) and pd.notna(avg_minutes)
                else None
            )

            usage_rate_proxy = None
            if {'FGA', 'FTA'}.issubset(player_stats.columns):
                usage_rate_proxy = (
                    pd.to_numeric(player_stats['FGA'], errors='coerce')
                    + 0.44 * pd.to_numeric(player_stats['FTA'], errors='coerce')
                ).mean()

            trend_l3 = None
            if len(last_5) >= 3:
                y = last_5.tail(3).values
                trend_l3 = float(np.polyfit(np.arange(len(y)), y, 1)[0])

            last_week_games = player_stats[
                player_stats['GAME_DATE'] >= today - pd.Timedelta(days=7)
            ]

            opp_stats = self._resolve_defense_stats(defense_lookup, opponent, game_season)

            # Injury / teammate impact
            team_code = str(matchup).split()[0] if matchup else None
            injured_teammates = []
            if team_code and isinstance(current_injuries, dict):
                raw = current_injuries.get(team_code) or []
                injured_teammates = [
                    p['name'] for p in raw
                    if isinstance(p, dict) and p.get('name')
                ]

            impact_features = {}
            if injured_teammates:
                try:
                    impact = self.impact_analyzer.calculate_impact_features(
                        player_name=player,
                        injured_players=injured_teammates,
                        market=market,
                        season=game_season,
                    )
                    if impact and impact.get('missing_minutes_sum', 0) > 0:
                        impact_features = impact
                except Exception:
                    pass

            # --- assemble feature row ---
            features = {
                # Core features (match training exactly)
                'line': line,
                'avg_last_5': avg_last_5,
                'avg_last_10': avg_last_10,
                'last_game': stat_values.iloc[0] if not stat_values.empty else None,
                'consistency_L5': last_5.std(),
                'line_vs_L5': line - avg_last_5,
                'line_vs_season': line - avg_season,
                'is_home': is_home,
                'days_rest': days_rest,
                'back_to_back': back_to_back,
                'avg_minutes': avg_minutes,
                'minutes_trend': minutes_trend,
                'hot_hand_indicator': int((last_5.head(3) > avg_season).sum() >= 2),
                'trend_L3': trend_l3,
                'games_in_last_week': len(last_week_games),
            }

            if usage_rate_proxy is not None:
                features['usage_rate_proxy'] = usage_rate_proxy

            if opp_stats is not None:
                features['opponent_pts_allowed'] = opp_stats.get('opp_pts_weighted')
                features['opponent_reb_rate'] = opp_stats.get('opp_reb_rate')
                features['opponent_tov_rate'] = opp_stats.get('opp_tov_rate')
                if opp_stats.get('opp_three_def') is not None:
                    features['opponent_three_def'] = opp_stats['opp_three_def']

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

            # Metadata carried through to output (prefixed to avoid collision with features)
            features['_player_name'] = player
            features['_market'] = market
            features['_line'] = float(line)
            features['_avg_last_5'] = float(avg_last_5)
            features['_odds_over'] = prop_row.get('odds_over', np.nan)
            features['_odds_under'] = prop_row.get('odds_under', np.nan)
            features['_bookmaker'] = prop_row.get('bookmaker', '')

            rows.append(features)

        if skipped:
            print(f"  Skipped {skipped} props (missing stats or unknown market)")

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def _make_predictions(self, features_df):
        """
        For each market:
          1. Reindex to the model's exact feature list (NaN for any missing column)
          2. Apply the saved imputer
          3. Predict
          4. Compute confidence from residual_std via normal CDF
        """
        results = []

        for market in features_df['_market'].unique():
            if market not in self.models:
                print(f"  No model for {market}, skipping")
                continue

            market_rows = features_df[features_df['_market'] == market].copy()
            model_data = self.models[market]

            model = model_data['model']
            imputer = model_data['imputer']
            feature_list = model_data['features']
            residual_std = float(model_data.get('residual_std') or 3.0)

            # Widen uncertainty band during playoffs — models are trained on
            # regular season data and have no concept of playoff intensity.
            if Config.is_playoff_period():
                residual_std *= Config.PLAYOFF_SIGMA_MULTIPLIER

            # Reindex to training feature list — unknown columns become NaN
            X = market_rows.reindex(columns=feature_list).apply(pd.to_numeric, errors="coerce")
            X_imputed = pd.DataFrame(
                imputer.transform(X),
                columns=feature_list,
                index=market_rows.index,
            )

            y_pred = model.predict(X_imputed)
            lines = market_rows['_line'].values

            # P(actual > line) via normal CDF centred on prediction
            sigma = max(residual_std, 0.5)
            prob_over = norm.cdf((y_pred - lines) / sigma)

            bet_direction = np.where(y_pred >= lines, 'OVER', 'UNDER')
            # Confidence = probability that the recommended side wins
            confidence = np.where(bet_direction == 'OVER', prob_over, 1.0 - prob_over) * 100.0

            result = pd.DataFrame({
                'player_name': market_rows['_player_name'].values,
                'market': market,
                'line': lines,
                'predicted_value': y_pred,
                'edge_vs_line': y_pred - lines,
                'bet_direction': bet_direction,
                'prob_over': prob_over,
                'confidence': confidence,
                'residual_std': residual_std,
                'is_playoff': Config.is_playoff_period(),
                'odds_over': market_rows['_odds_over'].values,
                'odds_under': market_rows['_odds_under'].values,
                'bookmaker': market_rows['_bookmaker'].values,
                # carry for Kelly sizing sanity check
                '_avg_last_5': market_rows['_avg_last_5'].values,
            })

            print(f"  {market}: {len(result)} predictions")
            results.append(result)

        if not results:
            return pd.DataFrame()

        return pd.concat(results, ignore_index=True)

    # ------------------------------------------------------------------
    # EV, Kelly, and recommendation
    # ------------------------------------------------------------------

    def _calculate_ev(self, df):
        """
        EV% = win_prob * (decimal_odds - 1) - loss_prob
        Uses prob_over and the relevant side's odds.
        """
        def _ev_row(row):
            is_over = row['bet_direction'] == 'OVER'
            win_prob = row['prob_over'] if is_over else (1.0 - row['prob_over'])
            odds = row['odds_over'] if is_over else row['odds_under']

            if pd.isna(odds):
                odds = -110

            decimal = (odds / 100 + 1) if odds > 0 else (100 / abs(odds) + 1)
            ev = win_prob * (decimal - 1) - (1 - win_prob)
            return round(ev * 100, 3)

        df = df.copy()
        df['expected_value'] = df.apply(_ev_row, axis=1)
        df['odds'] = df.apply(
            lambda r: r['odds_over'] if r['bet_direction'] == 'OVER' else r['odds_under'],
            axis=1,
        )
        return df

    def _add_kelly_sizing(self, df, bankroll=None):
        if bankroll is None:
            bankroll = load_bankroll_amount()

        def _kelly_row(row):
            is_over = row['bet_direction'] == 'OVER'
            odds = row['odds_over'] if is_over else row['odds_under']
            if pd.isna(odds):
                odds = -110
            sizing = calculate_bet_size(bankroll, float(row['confidence']), float(odds))
            return sizing

        df = df.copy()
        sizing_col = df.apply(_kelly_row, axis=1)
        df['kelly_bet_size'] = sizing_col.apply(lambda x: x['bet_amount'])
        df['units']          = sizing_col.apply(lambda x: x['units'])
        df['kelly_percent']  = sizing_col.apply(lambda x: x['kelly_percent'])
        df['rec_units']      = df['confidence'].apply(recommended_units)
        return df

    def _make_recommendation(self, row):
        if (
            row['expected_value'] > 0
            and row['confidence'] > 55
            and row['kelly_bet_size'] >= 5
        ):
            return f"BET {row['bet_direction']}  ${row['kelly_bet_size']:.0f}  ({row['rec_units']}u)"
        return "SKIP"

    # ------------------------------------------------------------------
    # Display and persistence
    # ------------------------------------------------------------------

    def _display_top_picks(self, predictions):
        if predictions.empty:
            print("\nNo positive EV bets found today")
            return

        top = predictions.nlargest(10, 'expected_value')
        print("\n" + "=" * 70)
        print("TOP BETTING RECOMMENDATIONS")
        print("=" * 70)

        for _, row in top.iterrows():
            print(f"\n{row['player_name']} — {row['market']}")
            print(f"  Line:           {row['line']:.1f}")
            print(f"  Prediction:     {row['predicted_value']:.2f}  (±{row['residual_std']:.2f} std)")
            print(f"  Bet:            {row['bet_direction']}")
            print(f"  P(over):        {row['prob_over'] * 100:.1f}%")
            print(f"  Confidence:     {row['confidence']:.1f}%")
            print(f"  Edge vs line:   {row['edge_vs_line']:+.2f}")
            print(f"  Odds:           {row['odds']}")
            print(f"  EV:             {row['expected_value']:+.2f}%")
            print(f"  Kelly bet:      ${row['kelly_bet_size']:.2f}")
            print(f"  Bookmaker:      {row['bookmaker']}")

    def _save_to_ongoing_collection(self, props_df):
        """Save today's props to the ongoing collection in long format.

        Regular season data goes to ongoing_odds_collection.csv.
        Playoff data goes to ongoing_odds_playoffs.csv so the two periods
        remain separate and can be used for independent analysis.
        """
        try:
            from scripts.live_odds_scraper import LiveOddsScraper

            is_playoff = Config.is_playoff_period()
            filename = "ongoing_odds_playoffs.csv" if is_playoff else "ongoing_odds_collection.csv"
            ongoing_file = Config.DATA_DIR / filename

            scraper = LiveOddsScraper()
            long_df = scraper.to_long_format(props_df)
            long_df['collection_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            long_df['is_playoff'] = is_playoff

            if ongoing_file.exists():
                existing = pd.read_csv(ongoing_file)
                combined = pd.concat([existing, long_df], ignore_index=True)
                dedup_cols = [c for c in [
                    "player_name", "market", "line", "bet_type", "odds", "bookmaker", "timestamp"
                ] if c in combined.columns]
                combined = combined.drop_duplicates(subset=dedup_cols)
                combined.to_csv(ongoing_file, index=False)
            else:
                long_df.to_csv(ongoing_file, index=False)

            label = "PLAYOFF" if is_playoff else "regular season"
            print(f"\nSaved {len(long_df)} rows ({label}) -> {filename}")

        except Exception as e:
            print(f"Error saving to ongoing collection: {e}")


if __name__ == "__main__":
    predictor = DailyPredictor()
    predictions = predictor.run_daily_predictions()

    if not predictions.empty:
        output_file = Config.DATA_DIR / f"predictions_{datetime.now().strftime('%Y-%m-%d')}.csv"
        predictions.to_csv(output_file, index=False)
        print(f"\nPredictions saved to: {output_file}")
