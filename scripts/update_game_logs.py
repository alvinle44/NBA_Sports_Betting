from scripts.nba_log_scrapper import NBAResultsScraper
from scripts.config import Config
# from config import Config
from datetime import datetime, timedelta
import pandas as pd

scraper = NBAResultsScraper()

# Check existing data
existing_file = Config.DATA_DIR / "game_results_2025-26.csv"

if existing_file.exists():
    existing = pd.read_csv(existing_file)
    existing['GAME_DATE'] = pd.to_datetime(existing['GAME_DATE'])
    last_date = existing['GAME_DATE'].max()
    start_date = (last_date + timedelta(days=1)).strftime('%Y-%m-%d')
    print(f"Updating from: {start_date}")
else:
    start_date = '2025-10-21'
    print(f"Starting fresh from: {start_date}")

# Get players from odds file
odds_file = Config.DATA_DIR / "historical_odds_combined.csv"
odds_df = pd.read_csv(odds_file)
players_list = odds_df['player_name'].unique().tolist()

# Fetch new games
today = datetime.now().strftime('%Y-%m-%d')
new_games = scraper.get_results_for_date_range(
    start_date=start_date,
    end_date=today,
    players_list=players_list,
    season='2025-26'
)

print(f"✓ Updated game logs")