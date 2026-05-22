"""
NBA Official Injury Report Scraper
Scrapes daily injury data from official.nba.com
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import json
import time
from scripts.config import Config

class NBAInjuryReportScraper:
    """
    Scrapes official NBA injury reports.
    
    Data source: https://official.nba.com/nba-injury-report-2025-26-season/
    
    Features:
    - Daily automated scraping
    - JSON and CSV export
    - Historical injury tracking
    - Team and player filtering
    """
    
    def __init__(self):
        self.base_url = "https://official.nba.com/nba-injury-report-2025-26-season/"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        self.injuries_file = Config.DATA_DIR / "current_injuries.json"
        self.history_file = Config.DATA_DIR / "injury_history.csv"
    
    def scrape_current_injuries(self):
        """
        Scrape current injury report from official NBA site.
        
        Returns:
            DataFrame with current injuries
        """
        print(f"\n🏥 Scraping NBA Official Injury Report...")
        print(f"   URL: {self.base_url}")
        
        try:
            # Fetch page
            response = requests.get(self.base_url, headers=self.headers, timeout=30)
            response.raise_for_status()
            
            print(f"   ✓ Page loaded (status: {response.status_code})")
            
            # Parse HTML
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find injury report table
            # The page structure may vary, so we'll try multiple selectors
            injuries = []
            
            # Method 1: Look for table with class 'injury-report'
            table = soup.find('table', class_='injury-report')
            
            if not table:
                # Method 2: Look for any table with 'Game Date' header
                tables = soup.find_all('table')
                for t in tables:
                    headers = [th.get_text(strip=True) for th in t.find_all('th')]
                    if 'Game Date' in headers or 'Player' in headers:
                        table = t
                        break
            
            if not table:
                print("   ⚠️  Could not find injury table on page")
                print("   Page structure may have changed")
                return pd.DataFrame()
            
            # Parse table
            rows = table.find_all('tr')
            
            current_game_date = None
            current_game_time = None
            current_matchup = None
            
            for row in rows[1:]:  # Skip header
                cells = row.find_all(['td', 'th'])
                
                if len(cells) < 3:
                    continue
                
                # Check if this is a game header row
                cell_text = cells[0].get_text(strip=True)
                
                if 'vs' in cell_text or '@' in cell_text:
                    # This is a game matchup row
                    current_matchup = cell_text
                    if len(cells) >= 2:
                        current_game_date = cells[1].get_text(strip=True) if len(cells) > 1 else None
                        current_game_time = cells[2].get_text(strip=True) if len(cells) > 2 else None
                    continue
                
                # This is a player injury row
                if len(cells) >= 4:
                    player_name = cells[0].get_text(strip=True)
                    team = cells[1].get_text(strip=True)
                    injury_status = cells[2].get_text(strip=True)
                    reason = cells[3].get_text(strip=True) if len(cells) > 3 else ''
                    
                    if player_name and player_name != 'Player':
                        injuries.append({
                            'date_scraped': datetime.now().strftime('%Y-%m-%d'),
                            'game_date': current_game_date,
                            'game_time': current_game_time,
                            'matchup': current_matchup,
                            'player_name': player_name,
                            'team': team,
                            'status': injury_status,
                            'reason': reason
                        })
            
            if not injuries:
                print("   No injuries found in table")
                return pd.DataFrame()
            
            injuries_df = pd.DataFrame(injuries)
            print(f"   ✓ Found {len(injuries_df)} injury reports")
            
            # Show summary
            print(f"\n   Summary:")
            print(f"   - Teams affected: {injuries_df['team'].nunique()}")
            print(f"   - Players injured: {injuries_df['player_name'].nunique()}")
            print(f"   - Status breakdown:")
            for status, count in injuries_df['status'].value_counts().items():
                print(f"     • {status}: {count}")
            
            return injuries_df
        
        except requests.RequestException as e:
            print(f"   Error fetching injury report: {e}")
            return pd.DataFrame()
        
        except Exception as e:
            print(f"    Error parsing injury report: {e}")
            return pd.DataFrame()
    
    def save_current_injuries(self, injuries_df):
        """
        Save current injuries to JSON file.
        
        Args:
            injuries_df: DataFrame with injuries
        """
        if injuries_df.empty:
            print("   ⚠️  No injuries to save")
            return
        
        # Convert to dict for JSON
        injuries_dict = {
            'last_updated': datetime.now().isoformat(),
            'count': len(injuries_df),
            'injuries': injuries_df.to_dict('records')
        }
        
        # Save as JSON
        with open(self.injuries_file, 'w') as f:
            json.dump(injuries_dict, f, indent=2)
        
        print(f"    Saved current injuries to: {self.injuries_file}")
    
    def update_injury_history(self, injuries_df):
        """
        Append current injuries to historical log.
        
        Args:
            injuries_df: DataFrame with current injuries
        """
        if injuries_df.empty:
            return
        
        # Load existing history
        if self.history_file.exists():
            history_df = pd.read_csv(self.history_file)
        else:
            history_df = pd.DataFrame()
        
        # Append new injuries
        updated_history = pd.concat([history_df, injuries_df], ignore_index=True)
        
        # Remove duplicates (same player, same date, same status)
        updated_history = updated_history.drop_duplicates(
            subset=['date_scraped', 'player_name', 'status'],
            keep='last'
        )
        
        # Sort by date
        updated_history = updated_history.sort_values('date_scraped', ascending=False)
        
        # Save
        updated_history.to_csv(self.history_file, index=False)
        
        print(f"   ✓ Updated injury history: {len(updated_history):,} total records")
    
    def load_current_injuries(self):
        """
        Load current injuries from cache.
        
        Returns:
            DataFrame with current injuries or None
        """
        if not self.injuries_file.exists():
            return None
        
        try:
            with open(self.injuries_file, 'r') as f:
                data = json.load(f)
            
            # Check if cache is recent (< 6 hours old)
            last_updated = datetime.fromisoformat(data['last_updated'])
            age_hours = (datetime.now() - last_updated).total_seconds() / 3600
            
            if age_hours > 6:
                print(f"   ⚠️  Injury cache is {age_hours:.1f} hours old (stale)")
                return None
            
            print(f"   ✓ Loaded {data['count']} injuries from cache ({age_hours:.1f}h old)")
            
            return pd.DataFrame(data['injuries'])
        
        except Exception as e:
            print(f"   ⚠️  Could not load injury cache: {e}")
            return None
    
    def get_team_injuries(self, team_code):
        """
        Get injuries for a specific team.
        
        Args:
            team_code: Team code (e.g., 'LAL', 'GSW')
        
        Returns:
            List of injured player names
        """
        injuries_df = self.load_current_injuries()
        
        if injuries_df is None or injuries_df.empty:
            return []
        
        # Filter for team
        team_injuries = injuries_df[injuries_df['team'] == team_code]
        
        # Only include OUT and DOUBTFUL (not QUESTIONABLE/PROBABLE)
        significant_statuses = ['Out', 'Doubtful']
        team_injuries = team_injuries[team_injuries['status'].isin(significant_statuses)]
        
        return team_injuries['player_name'].tolist()
    
    def get_player_injury_status(self, player_name):
        """
        Get injury status for a specific player.
        
        Args:
            player_name: Player's full name
        
        Returns:
            Dict with status info or None
        """
        injuries_df = self.load_current_injuries()
        
        if injuries_df is None or injuries_df.empty:
            return None
        
        player_injury = injuries_df[injuries_df['player_name'] == player_name]
        
        if player_injury.empty:
            return None
        
        return player_injury.iloc[0].to_dict()
    
    def run_daily_update(self):
        """
        Run daily injury report update.
        
        This should be called each morning before making predictions.
        """
        print(f"\n{'='*70}")
        print("DAILY INJURY REPORT UPDATE")
        print(f"{'='*70}")
        print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Scrape current injuries
        injuries_df = self.scrape_current_injuries()
        
        if injuries_df.empty:
            print("\n⚠️  Failed to scrape injuries. Using cached data if available.")
            return
        
        # Save current injuries
        self.save_current_injuries(injuries_df)
        
        # Update historical log
        self.update_injury_history(injuries_df)
        
        print(f"\n{'='*70}")
        print("✓ INJURY REPORT UPDATE COMPLETE")
        print(f"{'='*70}")
        
        return injuries_df
    
    def print_injury_report(self):
        """
        Print formatted injury report.
        """
        injuries_df = self.load_current_injuries()
        
        if injuries_df is None or injuries_df.empty:
            print("No current injury data available")
            return
        
        print(f"\n{'='*70}")
        print("CURRENT NBA INJURY REPORT")
        print(f"{'='*70}")
        
        # Group by team
        for team in sorted(injuries_df['team'].unique()):
            team_injuries = injuries_df[injuries_df['team'] == team]
            print(f"\n{team}:")
            
            for _, injury in team_injuries.iterrows():
                status_emoji = {
                    'Out': '🔴',
                    'Doubtful': '🟠',
                    'Questionable': '🟡',
                    'Probable': '🟢'
                }.get(injury['status'], '⚪')
                
                print(f"  {status_emoji} {injury['player_name']:25s} {injury['status']:12s} {injury['reason']}")