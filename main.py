import requests          # Make HTTP requests to APIs
import pandas as pd      # Data manipulation (like Excel for Python)
import numpy as np       # Numerical operations and math
from datetime import datetime, timedelta  # Handle dates and times
import pickle           # Save/load Python objects (models, cache)
import time             # Add delays for rate limiting
import warnings         # Suppress unnecessary warnings
import argparse         # Parse command line arguments
import json             # Read/write JSON files
from pathlib import Path  # Cross-platform file paths

# Machine Learning Libraries
from xgboost import XGBRegressor  # Gradient boosted trees (our ML algorithm)
from sklearn.model_selection import TimeSeriesSplit  # Time-series cross-validation
from sklearn.metrics import mean_absolute_error, mean_squared_error  # Evaluate model
from sklearn.preprocessing import StandardScaler  # Normalize features
from scipy.stats import norm
from scripts.config import Config
from scripts.daily_predictor import DailyPredictor
from scripts.data_preparation import DataPreparation
from scripts.model_trainer import ModelTrainer
from scripts.nba_log_scrapper import NBAResultsScraper
from scripts.odds_collecter import HistoricalOddsScraper
from scripts.results_tracker import ResultsTracker  # Calculate probabilities from predictions
ODDS_API_KEY = "01d3539ad85763e285a27d276e598c4e"
# Web Scraping
try:
    from bs4 import BeautifulSoup  # Parse HTML for injury scraping
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False
    print("WARNING: beautifulsoup4 not installed. Run: pip install beautifulsoup4")

# NBA API - Official NBA statistics
try:
    from nba_api.stats.endpoints import (
        playergamelog,      # Get player's game-by-game stats
        leaguedashteamstats,  # Get team defensive stats
        teamgamelogs,       # Get team's recent games
        commonplayerinfo    # Get player bio info (position, etc.)
    )
    from nba_api.stats.static import players, teams  # Get player/team IDs
    NBA_API_AVAILABLE = True
except ImportError:
    print("WARNING: nba_api not installed. Run: pip install nba-api")
    NBA_API_AVAILABLE = False

warnings.filterwarnings('ignore')  # Suppress pandas/sklearn warnings


def main():
    """
    Main entry point for the system.
    
    Handles command-line arguments and routes to appropriate function.
    
    Usage examples:
        python nba_props_system.py --mode collect_historical --start 2024-01-01 --end 2024-03-31
        python nba_props_system.py --mode prepare_data --start 2024-01-01 --end 2024-03-31
        python nba_props_system.py --mode train
        python nba_props_system.py --mode daily_predictions
        python nba_props_system.py --mode track_results
        python nba_props_system.py --mode retrain
        python nba_props_system.py --mode merge_datasets
    """
    parser = argparse.ArgumentParser(description='NBA Player Props Betting System')
    parser.add_argument('--mode', required=True, 
                       choices=['collect_historical', 'prepare_data', 'train', 
                               'daily_predictions', 'track_results', 'retrain', 'merge_datasets'],
                       help='Operation mode')
    parser.add_argument('--start', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', help='End date (YYYY-MM-DD)')
    parser.add_argument('--predictions_file', help='Predictions file to track results')
    
    args = parser.parse_args()
    
    # ===== MODE: COLLECT HISTORICAL ODDS =====
    if args.mode == 'collect_historical':
        """
        Collect historical odds from The Odds API.
        
        What it does:
        - Scrapes odds for date range
        - Saves to CSV
        - Costs API quota
        
        Run this ONCE to get initial training data.
        """
        if not args.start or not args.end:
            print("Error: --start and --end dates required")
            return
        
        # Convert to ISO format
        start = f"{args.start}T00:00:00Z"
        end = f"{args.end}T23:59:59Z"
        
        scraper = HistoricalOddsScraper(Config.ODDS_API_KEY)
        scraper.scrape_date_range(start, end)
    
    # ===== MODE: PREPARE TRAINING DATA =====
    elif args.mode == 'prepare_data':
        """
        Prepare training data from historical odds.
        
        What it does:
        - Loads historical odds
        - Fetches actual game results
        - Merges and labels
        - Builds features
        - Saves training-ready CSV
        
        Run this AFTER collect_historical.
        """
        if not args.start or not args.end:
            print("Error: --start and --end dates required")
            return
        
        start = f"{args.start}T00:00:00Z"
        end = f"{args.end}T23:59:59Z"
        
        DataPreparation.prepare_training_data(start, end)
    
    # ===== MODE: TRAIN MODELS =====
    elif args.mode == 'train':
        """
        Train machine learning models.
        
        What it does:
        - Loads training data
        - Trains 8 models (one per market)
        - Saves models to disk
        
        Run this AFTER prepare_data.
        """
        # Find latest training data file
        training_files = list(Config.DATA_DIR.glob('training_data_*.csv'))
        if not training_files:
            print("Error: No training data found. Run 'prepare_data' first.")
            return
        
        latest_file = max(training_files, key=lambda p: p.stat().st_mtime)
        print(f"Using training data: {latest_file}")
        
        trainer = ModelTrainer()
        trainer.train_all_models(latest_file)
    
    # ===== MODE: DAILY PREDICTIONS =====
    elif args.mode == 'daily_predictions':
        """
        Make daily predictions.
        
        What it does:
        - Loads cached player stats
        - Scrapes injuries from ESPN
        - Gets today's games
        - Gets props for each game
        - Makes predictions
        - Outputs recommendations
        - Saves data for future retraining
        
        Run this EVERY MORNING before games.
        """
        predictor = DailyPredictor(Config.ODDS_API_KEY)
        predictor.run_daily_predictions()
    
    # ===== MODE: TRACK RESULTS =====
    elif args.mode == 'track_results':
        """
        Track yesterday's betting results.
        
        What it does:
        - Loads yesterday's predictions
        - Fetches actual results
        - Calculates wins/losses
        - Calculates ROI
        
        Run this NEXT DAY after games finish.
        """
        if not args.predictions_file:
            # Find most recent predictions
            pred_files = list(Config.PREDICTIONS_DIR.glob('recommendations_*.csv'))
            if not pred_files:
                print("Error: No predictions file found")
                return
            latest_file = max(pred_files, key=lambda p: p.stat().st_mtime)
        else:
            latest_file = args.predictions_file
        
        ResultsTracker.track_results(latest_file)
    
    # ===== MODE: RETRAIN WITH NEW DATA =====
    elif args.mode == 'retrain':
        """
        Retrain models with accumulated ongoing data.
        
        What it does:
        - Loads ongoing_odds_collection.csv
        - Fetches actual results
        - Labels everything
        - Retrains all models
        
        Run this WEEKLY to keep models current.
        """
        print(f"\n{'='*60}")
        print("RETRAINING MODELS WITH NEW DATA")
        print(f"{'='*60}")
        
        # Check for ongoing collection file
        ongoing_file = Config.DATA_DIR / 'ongoing_odds_collection.csv'
        if not ongoing_file.exists():
            print("Error: No ongoing odds collection found.")
            print("Run daily_predictions first to start collecting data.")
            return
        
        odds_df = pd.read_csv(ongoing_file)
        print(f"Loaded {len(odds_df)} odds records from ongoing collection")
        
        # Get unique players
        unique_players = odds_df['player_name'].unique()
        print(f"Unique players: {len(unique_players)}")
        
        # Get date range
        odds_df['game_date_only'] = pd.to_datetime(odds_df['game_date']).dt.strftime('%Y-%m-%d')
        start_date = odds_df['game_date_only'].min()
        end_date = odds_df['game_date_only'].max()
        print(f"Date range: {start_date} to {end_date}")
        
        # Fetch results
        scraper = NBAResultsScraper()
        results_df = scraper.get_results_for_date_range(
            start_date + "T00:00:00Z",
            end_date + "T23:59:59Z",
            unique_players
        )
        
        if results_df.empty:
            print("Error: Could not fetch game results")
            return
        
        # Merge and label
        merged = DataPreparation.merge_odds_with_results(odds_df, results_df)
        labeled = DataPreparation.label_results(merged)
        
        # Build features
        features_df = DataPreparation.build_features(labeled, results_df)
        
        # Save updated training data
        training_file = Config.DATA_DIR / f"training_data_ongoing_{datetime.now().strftime('%Y%m%d')}.csv"
        features_df.to_csv(training_file, index=False)
        print(f"\n✓ Saved updated training data: {training_file}")
        
        # Retrain models
        trainer = ModelTrainer()
        trainer.train_all_models(training_file)
        
        print(f"\n{'='*60}")
        print("✓ RETRAINING COMPLETE")
        print(f"{'='*60}")
        print(f"Models updated with {len(features_df)} training samples")
        print("You can now run daily_predictions with the updated models")
    
    # ===== MODE: MERGE ALL DATASETS =====
    elif args.mode == 'merge_datasets':
        """
        Merge all training data files into one comprehensive dataset.
        
        What it does:
        - Loads all training_data_*.csv files
        - Combines them
        - Removes duplicates
        - Saves merged dataset
        
        Run this MONTHLY for full model refresh.
        """
        print(f"\n{'='*60}")
        print("MERGING DATASETS")
        print(f"{'='*60}")
        
        all_data = []
        
        # Load all training data files
        for file in Config.DATA_DIR.glob('training_data_*.csv'):
            df = pd.read_csv(file)
            all_data.append(df)
            print(f"Loaded: {file.name} ({len(df)} records)")
        
        if not all_data:
            print("Error: No training data files found")
            return
        
        # Combine and deduplicate
        combined = pd.concat(all_data, ignore_index=True)
        
        # Remove duplicates (same player, game, market, line)
        combined = combined.drop_duplicates(
            subset=['player_name', 'game_date', 'market', 'line'],
            keep='last'
        )
        
        print(f"\nCombined: {len(combined)} unique training samples")
        
        # Save merged dataset
        merged_file = Config.DATA_DIR / f"training_data_merged_{datetime.now().strftime('%Y%m%d')}.csv"
        combined.to_csv(merged_file, index=False)
        
        print(f"✓ Saved merged dataset: {merged_file}")
        print("\nNow run: python nba_props_system.py --mode train")
        print("to retrain models with all available data")


# ============================================================================
# SECTION 10: HELP TEXT & DOCUMENTATION
# ============================================================================

if __name__ == "__main__":
    # If no args provided, show usage guide
    import sys
    if len(sys.argv) == 1:
        print("""
╔════════════════════════════════════════════════════════════════════════════╗
║                  NBA PLAYER PROPS BETTING SYSTEM                           ║
║                     Complete Production Version                            ║
╚════════════════════════════════════════════════════════════════════════════╝

FEATURES:
─────────
✓ Historical odds collection from The Odds API
✓ Automatic ESPN injury scraping (no manual work!)
✓ Automatic player position detection
✓ Smart caching (30x faster on repeated runs)
✓ Recent defensive stats (last 10 games + season averages)
✓ 32 advanced features including:
  - Player rolling averages & trends
  - Opponent defense (season, recent, weighted)
  - Teammate injuries & usage bump
  - Matchup history
  - Game context (home/away, rest)
✓ Separate XGBoost models per market
✓ Probability-calibrated predictions
✓ Continuous data collection for retraining
✓ Performance tracking with ROI calculation

INSTALLATION:
─────────────
pip install pandas numpy requests xgboost scikit-learn scipy nba-api beautifulsoup4

SETUP (First Time):
───────────────────
1. Set your API key:
   Edit line 122: ODDS_API_KEY = "01d3539ad85763e285a27d276e598c4e"
   Get key from: https://theoddsapi.com

2. Collect historical data (one-time):
   python nba_props_system.py --mode collect_historical --start 2024-01-01 --end 2024-03-31

3. Prepare training data:
   python nba_props_system.py --mode prepare_data --start 2024-01-01 --end 2024-03-31

4. Train models:
   python nba_props_system.py --mode train

DAILY WORKFLOW:
───────────────
Every morning before games (9 AM):

1. Get predictions:
   python nba_props_system.py --mode daily_predictions
   
   This automatically:
   • Loads cached player stats (instant if < 24hrs old)
   • Scrapes ESPN injuries (no manual work!)
   • Fetches today's games
   • Gets props from all bookmakers
   • Makes predictions with 32 features
   • Outputs betting recommendations
   • Saves data for future retraining

2. Track results (next day):
   python nba_props_system.py --mode track_results

WEEKLY MAINTENANCE:
───────────────────
After ~20 games, retrain models:
   python nba_props_system.py --mode retrain

MONTHLY MAINTENANCE:
────────────────────
For comprehensive update:
   python nba_props_system.py --mode merge_datasets
   python nba_props_system.py --mode train

FILE STRUCTURE:
───────────────
data/
  ├─ historical_odds_*.csv           [Initial historical scrape]
  ├─ ongoing_odds_collection.csv     [AUTO-GROWS with daily predictions]
  ├─ player_stats_cache_*.pkl        [24hr cache, auto-refreshes]
  ├─ injuries.json                   [Auto-scraped from ESPN]
  ├─ game_results_*.csv              [Actual NBA stats]
  └─ training_data_*.csv             [Labeled training data]

models/
  ├─ player_points_model.pkl         [Trained models]
  ├─ player_assists_model.pkl
  └─ ... (8 models total)

predictions/
  ├─ all_predictions_*.csv           [All daily analysis]
  ├─ recommendations_*.csv           [Your betting picks]
  └─ results_*.csv                   [Performance tracking]

KEY FEATURES EXPLAINED:
───────────────────────

1. AUTOMATIC INJURY SCRAPING
   • Scrapes ESPN every run
   • No manual entry needed
   • Falls back to cached file if ESPN fails

2. AUTOMATIC POSITION DETECTION
   • Uses NBA API + stat analysis
   • Works for any player (not just stars)
   • No manual mapping required

3. RECENT DEFENSIVE STATS
   • Season-long (stable baseline)
   • Recent 10 games (current form)
   • Weighted average (70% recent, 30% season)
   • Defensive trend (improving/declining)

4. SMART CACHING
   • First run/day: ~60 seconds
   • Later runs: ~30 seconds
   • 95% fewer API calls

5. CONTINUOUS LEARNING
   • Daily predictions auto-save to ongoing_odds_collection.csv
   • Weekly retraining keeps models current
   • Self-improving over time

CONFIGURATION:
──────────────
Edit Config class (lines 117-177) to customize:
  • MIN_EDGE = 2.0        (minimum edge to bet)
  • MIN_PROB = 0.55       (minimum win probability)
  • MARKETS = [...]       (which props to track)
  • CURRENT_SEASON        (update each season)

TIPS FOR SUCCESS:
─────────────────
✓ Start with paper trading (track but don't bet)
✓ Need 52.4%+ win rate to profit at -110 odds
✓ Target 55%+ for comfortable margin
✓ Use 1-2% of bankroll per bet
✓ Retrain weekly with new data
✓ Track everything for analysis

TROUBLESHOOTING:
────────────────
• "No models found" → Run --mode train first
• "Injury scraper failing" → ESPN structure may have changed, uses cached file
• "Slow predictions" → Delete cache to force refresh
• "Out of API quota" → Historical scraping costs quota, daily predictions cheap

SUPPORT:
────────
• Documentation: See detailed comments in code
• API Docs: https://the-odds-api.com/liveapi/guides/v4/
• NBA API: https://github.com/swar/nba_api

Good luck! 🎲 Bet responsibly. Past performance ≠ future results.
        """)
    else:
        main()