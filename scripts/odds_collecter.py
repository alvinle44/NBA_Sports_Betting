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
from scripts.config import Config
# Machine Learning Libraries
from xgboost import XGBRegressor  # Gradient boosted trees (our ML algorithm)
from sklearn.model_selection import TimeSeriesSplit  # Time-series cross-validation
from sklearn.metrics import mean_absolute_error, mean_squared_error  # Evaluate model
from sklearn.preprocessing import StandardScaler  # Normalize features
from scipy.stats import norm  # Calculate probabilities from predictions

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



class HistoricalOddsScraper:
    """
    Scrapes historical odds data from The Odds API.
    
    NEW FEATURES:
    - Automatic duplicate date detection
    - Collection logging (tracks what you've collected)
    - Smart skip logic (avoids re-collecting same dates)
    - Progress tracking
    """
    
    def __init__(self, api_key):
        """
        Initialize the scraper.
        
        Args:
            api_key: Your The Odds API key
        """
        self.api_key = api_key
        self.base_url = "https://api.the-odds-api.com/v4"
        self.quota_used = 0
    
    def get_historical_events(self, date):
        """
        Get all NBA games that occurred on a specific date.
        NOW WITH DEBUG OUTPUT
        
        Args:
            date: Date string in ISO format (e.g., "2024-01-15T00:00:00Z")
        
        Returns:
            List of game objects with IDs, teams, and start times
        """
        url = f"{self.base_url}/historical/sports/basketball_nba/events"
        params = {
            'apiKey': self.api_key,
            'date': date
        }
        
        # ===== DEBUG OUTPUT =====
        print(f"    📡 API Request:")
        print(f"       URL: {url}")
        print(f"       Date: {date}")
        
        response = requests.get(url, params=params)
        
        print(f"    📥 Response:")
        print(f"       Status: {response.status_code}")
        
        if response.status_code == 200:
            self.update_quota(response)
            result = response.json()
            
            # Show what we got back
            print(f"       Keys: {list(result.keys())}")
            
            events = result.get('data', [])
            print(f"       Events: {len(events)}")
            
            if not events and 'data' in result:
                print(f"       ⚠️  API returned 'data' key but it's empty")
                print(f"       Full response: {result}")
            
            return events
        else:
            print(f"    ❌ ERROR Response:")
            print(f"       {response.text[:500]}")
            
        return []
    
    def get_event_odds(self, event_id, event_date):
        """
        Get player props odds for a specific game.
        FILTERS TO ONLY: DraftKings, FanDuel, BetMGM
        
        Args:
            event_id: Unique game identifier
            event_date: When the game started
        
        Returns:
            Dict with odds from 3 bookmakers only
        """
        close_time = datetime.fromisoformat(event_date.replace('Z', '+00:00'))
        close_time -= timedelta(hours=1)
        date_param = close_time.isoformat().replace('+00:00', 'Z')
        
        url = f"{self.base_url}/historical/sports/basketball_nba/events/{event_id}/odds"
        params = {
            'apiKey': self.api_key,
            'date': date_param,
            'regions': 'us',
            'markets': ','.join(Config.MARKETS),
            'bookmakers': ','.join(Config.BOOKMAKERS),  # ← FILTERS TO 3 BOOKMAKERS
            'oddsFormat': 'american'
        }
        
        response = requests.get(url, params=params)
        if response.status_code == 200:
            self.update_quota(response)
            return response.json()
        return None
    
    def update_quota(self, response):
        """Track API quota usage from response headers."""
        used = response.headers.get('x-requests-used')
        if used:
            self.quota_used = int(used)
    
    def parse_props(self, odds_data):
        """
        Convert API response into flat pandas DataFrame.
        
        Args:
            odds_data: Raw JSON response from API
        
        Returns:
            DataFrame with one row per prop line
        """
        if not odds_data or 'data' not in odds_data:
            return pd.DataFrame()
        
        data = odds_data['data']
        rows = []
        
        for bookmaker in data.get('bookmakers', []):
            for market in bookmaker.get('markets', []):
                for outcome in market.get('outcomes', []):
                    rows.append({
                        'game_id': data['id'],
                        'game_date': data['commence_time'],
                        'home_team': data['home_team'],
                        'away_team': data['away_team'],
                        'bookmaker': bookmaker['key'],
                        'market': market['key'],
                        'player_name': outcome.get('description', ''),
                        'bet_type': outcome['name'],
                        'line': outcome.get('point'),
                        'odds': outcome.get('price'),
                        'timestamp': odds_data.get('timestamp', ''),
                    })
        
        return pd.DataFrame(rows)
    
    # ========== NEW METHODS FOR DUPLICATE PREVENTION ==========
    
    def check_existing_dates(self):
        """
        Check which dates already have data collected.
        
        Returns:
            dict: {date: {'file': filename, 'rows': count}}
        """
        existing_dates = {}
        
        print("\n🔍 Checking for existing data...")
        
        # Check all historical odds files
        for file in Config.DATA_DIR.glob('historical_odds_*.csv'):
            try:
                df = pd.read_csv(file)
                
                if 'game_date' not in df.columns or len(df) == 0:
                    continue
                
                df['game_date'] = pd.to_datetime(df['game_date'])
                
                # Get unique dates in this file
                dates = df['game_date'].dt.date.unique()
                
                for date in dates:
                    if date not in existing_dates:
                        existing_dates[date] = {
                            'files': [],
                            'total_rows': 0
                        }
                    
                    existing_dates[date]['files'].append(file.name)
                    existing_dates[date]['total_rows'] += len(df[df['game_date'].dt.date == date])
                
            except Exception as e:
                print(f"⚠️  Warning: Could not read {file.name}: {e}")
        
        if existing_dates:
            print(f"✓ Found existing data for {len(existing_dates)} dates")
        else:
            print("✓ No existing data found (clean slate)")
        
        return existing_dates
    
    def log_collection(self, start_date, end_date, rows, dates_collected, quota_used, filename):
        """
        Log collection details to a tracking file.
        
        Args:
            start_date: Start date of collection
            end_date: End date of collection
            rows: Number of rows collected
            dates_collected: Number of unique dates
            quota_used: API requests used
            filename: Output filename
        """
        log_file = Config.DATA_DIR / 'collection_log.json'
        
        # Load existing log
        if log_file.exists():
            try:
                with open(log_file, 'r') as f:
                    log = json.load(f)
            except:
                log = {}
        else:
            log = {}
        
        # Add new entry
        entry_key = f"{start_date[:10]}_{end_date[:10]}"
        log[entry_key] = {
            'collected_on': datetime.now().isoformat(),
            'start_date': start_date[:10],
            'end_date': end_date[:10],
            'rows': rows,
            'dates_collected': dates_collected,
            'quota_used': quota_used,
            'file': filename
        }
        
        # Save log
        with open(log_file, 'w') as f:
            json.dump(log, f, indent=2)
        
        print(f"\n✓ Logged collection to {log_file.name}")
    
    def get_dates_in_range(self, start_date, end_date):
        """
        Get list of all dates in range.
        
        Args:
            start_date: Start date string
            end_date: End date string
        
        Returns:
            list of datetime.date objects
        """
        start = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
        end = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
        
        dates = []
        current = start
        while current <= end:
            dates.append(current.date())
            current += timedelta(days=1)
        
        return dates
    
    def scrape_date_range(self, start_date, end_date, max_quota=199500):
        """
        Main scraping function with automatic duplicate prevention.
        NOW WITH INCREMENTAL SAVING AND QUOTA PROTECTION!
        
        Process:
        1. Check what dates already exist
        2. Show user what will be skipped
        3. Collect only NEW dates
        4. SAVE AFTER EACH DAY (new!)
        5. Stop if quota limit approaching (new!)
        6. Log everything
        
        Args:
            start_date: Start date (ISO format)
            end_date: End date (ISO format)
            max_quota: Stop when reaching this quota (default 19,800)
        
        Returns:
            DataFrame with all collected odds
        """
        print(f"\n{'='*60}")
        print(f"NBA Historical Odds Collection")
        print(f"{'='*60}")
        print(f"Date Range: {start_date[:10]} to {end_date[:10]}")
        print(f"Quota Safety Limit: {max_quota:,} / 100,000")
        
        # ===== STEP 1: CHECK EXISTING DATA =====
        existing_dates = self.check_existing_dates()
        
        # Get all dates in requested range
        requested_dates = self.get_dates_in_range(start_date, end_date)
        
        # Determine which dates are NEW vs EXISTING
        new_dates = []
        duplicate_dates = []
        
        for date in requested_dates:
            if date in existing_dates:
                duplicate_dates.append(date)
            else:
                new_dates.append(date)
        
        # ===== STEP 2: SHOW SUMMARY =====
        print(f"\n📊 Collection Plan:")
        print(f"  Total dates requested: {len(requested_dates)}")
        print(f"  New dates to collect: {len(new_dates)}")
        print(f"  Dates already collected: {len(duplicate_dates)}")
        
        if duplicate_dates:
            print(f"\n⚠️  These dates will be SKIPPED (already collected):")
            for date in duplicate_dates[:10]:  # Show first 10
                files = existing_dates[date]['files']
                rows = existing_dates[date]['total_rows']
                print(f"    {date} - {rows} props in {files[0]}")
            if len(duplicate_dates) > 10:
                print(f"    ... and {len(duplicate_dates) - 10} more dates")
        
        if not new_dates:
            print("\n✓ All dates already collected! No new data to fetch.")
            print("  To re-collect, delete or rename existing CSV files.")
            return pd.DataFrame()
        
        # ===== STEP 3: ESTIMATE QUOTA =====
        estimated_requests = len(new_dates) * 13  # ~13 per day average
        
        print(f"\n💰 Quota Estimate:")
        print(f"   Current used: {self.quota_used:,}")
        print(f"   Estimated for collection: {estimated_requests:,}")
        print(f"   Total after collection: {self.quota_used + estimated_requests:,}")
        
        if self.quota_used + estimated_requests > max_quota:
            print(f"\n⚠️  WARNING: This would exceed your safety limit of {max_quota:,}!")
            print(f"   Consider collecting fewer days or increasing max_quota")
        
        # ===== STEP 4: CONFIRM WITH USER =====
        response = input(f"\nProceed with collecting {len(new_dates)} new dates? (y/n): ")
        if response.lower() != 'y':
            print("Collection cancelled.")
            return pd.DataFrame()
        
        # ===== STEP 5: PREPARE OUTPUT FILE =====
        output_file = Config.DATA_DIR / f"historical_odds_{start_date[:10]}_{end_date[:10]}.csv"
        
        print(f"\n{'='*60}")
        print("Starting Collection...")
        print(f"{'='*60}")
        print(f"📁 Output file: {output_file.name}")
        print(f"💾 Saving incrementally after each day")
        
        all_props = []
        dates_collected = 0
        dates_skipped = 0
        
        current = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
        end_dt = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
        
        while current <= end_dt:
            # ===== QUOTA SAFETY CHECK =====
            if self.quota_used >= max_quota:
                print(f"\n🛑 QUOTA SAFETY LIMIT REACHED!")
                print(f"   Used: {self.quota_used:,} / {max_quota:,}")
                print(f"   Stopping collection to prevent overage")
                print(f"   Collected data has been saved incrementally")
                break
            
            date_str = current.isoformat().replace('+00:00', 'Z')
            current_date = current.date()
            
            # Skip if already exists
            if current_date in existing_dates:
                dates_skipped += 1
                current += timedelta(days=1)
                continue
            
            print(f"\n[{dates_collected + 1}/{len(new_dates)}] {current.strftime('%Y-%m-%d')}")
            
            # Get games for this day
            events = self.get_historical_events(date_str)
            
            if not events:
                print(f"  No games found")
                current += timedelta(days=1)
                continue
            
            print(f"  Found {len(events)} games")
            
            # ===== COLLECT PROPS FOR THIS DAY =====
            day_props = []
            
            for event in events:
                # Check quota before each game
                if self.quota_used >= max_quota:
                    print(f"    🛑 Quota limit reached, stopping day collection")
                    break
                
                odds_data = self.get_event_odds(event['id'], event['commence_time'])
                if odds_data:
                    props_df = self.parse_props(odds_data)
                    if not props_df.empty:
                        day_props.append(props_df)
                        print(f"    {event['away_team']} @ {event['home_team']}: {len(props_df)} props")
                time.sleep(1)  # Rate limiting
            
            # ===== SAVE THIS DAY'S DATA IMMEDIATELY =====
            if day_props:
                day_df = pd.concat(day_props, ignore_index=True)
                
                # Append to file (or create if first day)
                if output_file.exists():
                    # Read existing data
                    existing_df = pd.read_csv(output_file)
                    # Combine with new day's data
                    combined_df = pd.concat([existing_df, day_df], ignore_index=True)
                    # Save combined data
                    combined_df.to_csv(output_file, index=False)
                    print(f"  💾 Appended {len(day_df)} props → Total: {len(combined_df):,} props in file")
                else:
                    # Create new file with first day's data
                    day_df.to_csv(output_file, index=False)
                    print(f"  💾 Created file with {len(day_df)} props")
                
                # Also keep in memory for final report
                all_props.append(day_df)
            else:
                print(f"  ⚠️  No props collected for this day")
            
            dates_collected += 1
            current += timedelta(days=1)
            
            print(f"  📊 Quota used: {self.quota_used:,} / 20,000")
            
            # Show quota warning if getting close
            remaining = 20000 - self.quota_used
            if remaining < 500:
                print(f"WARNING: Only {remaining:,} requests remaining!")
        
        # ===== STEP 6: FINAL SUMMARY =====
        print(f"\n{'='*60}")
        print("Collection Complete!")
        print(f"{'='*60}")
        print(f"  Dates collected: {dates_collected}")
        print(f"  Dates skipped: {dates_skipped}")
        print(f"  Total API requests: {self.quota_used:,}")
        
        if all_props:
            final_df = pd.concat(all_props, ignore_index=True)
            
            print(f"\n✓ Final file saved: {output_file}")
            print(f"  Total props collected: {len(final_df):,}")
            print(f"  Unique dates: {len(all_props)}")
            print(f"  Average props per day: {len(final_df) / len(all_props):.0f}")
            
            # Verify file exists and matches
            if output_file.exists():
                saved_df = pd.read_csv(output_file)
                print(f"  ✓ File verified: {len(saved_df):,} rows on disk")
            
            # Log the collection
            self.log_collection(
                start_date,
                end_date,
                len(final_df),
                dates_collected,
                self.quota_used,
                output_file.name
            )
            
            return final_df
        else:
            print("\n⚠️  No new data collected")
            return pd.DataFrame()