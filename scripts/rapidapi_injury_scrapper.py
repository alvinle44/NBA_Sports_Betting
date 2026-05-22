import requests
import json
from datetime import datetime, timedelta
from scripts.config import Config
import time

class RapidAPIInjuryScraper:
    """
    Fetches NBA injuries from RapidAPI for current season (2025-26).
    Replaces ESPN scraping with reliable API calls.
    """
    
    def __init__(self):
        self.api_key = Config.RAPIDAPI_KEY  # Add to config.py
        self.base_url = "https://nba-injuries-reports.p.rapidapi.com/injuries/nba"
        self.headers = {
            "X-RapidAPI-Key": self.api_key,
            "X-RapidAPI-Host": "nba-injuries-reports.p.rapidapi.com"
        }
        
        # Cache directory for historical injuries
        self.cache_dir = Config.DATA_DIR / "injury_cache"
        self.cache_dir.mkdir(exist_ok=True)
    
    def get_injuries_for_date(self, date_str):
        """
        Fetch injuries for a specific date from RapidAPI and cache the result.
        
        Args:
            date_str (str): Date in 'YYYY-MM-DD' format.
        
        Returns:
            List[dict]: Each injury with keys {player, team, status, reason, date}.
        """
        print(f"Fetching injuries for {date_str}...")
        
        # --- 1️⃣ Check cache first ---
        cache_file = self.cache_dir / f"injuries_{date_str}.json"
        if cache_file.exists():
            print(f"  ✓ Loading from cache: {cache_file}")
            with open(cache_file, 'r') as f:
                return json.load(f)

        # --- 2️⃣ Make API call ---
        url = f"{self.base_url}/{date_str}"
        try:
            response = requests.get(url, headers=self.headers, timeout=10)

            # Handle rate limits and errors
            if response.status_code == 429:
                print("  ⚠️ Rate limited, waiting 60 seconds...")
                time.sleep(60)
                return self.get_injuries_for_date(date_str)
            elif response.status_code != 200:
                print(f"  ❌ API error: {response.status_code}")
                return []

            data = response.json()

            # --- 3️⃣ Normalize data structure ---
            # Some APIs return a list, others wrap it in a dict
            if isinstance(data, list):
                injury_data = data
            elif isinstance(data, dict):
                # Look for common keys used by RapidAPI endpoints
                injury_data = data.get('body') or data.get('data') or data.get('injuries') or []
            else:
                print("  ⚠️ Unexpected data format")
                return []

            # --- 4️⃣ Parse and clean data ---
            parsed = []
            for injury in injury_data:
                # Safely extract fields with fallbacks
                player = injury.get('playerName') or injury.get('player') or injury.get('name', '')
                team = injury.get('team') or injury.get('teamName', '')
                status = injury.get('status', 'OUT')
                reason = injury.get('injury') or injury.get('description', 'Unknown')

                parsed.append({
                    "player": player,
                    "team": team,
                    "status": status,
                    "reason": reason,
                    "date": date_str
                })

            # --- 5️⃣ Save to cache ---
            with open(cache_file, 'w') as f:
                json.dump(parsed, f, indent=2)

            print(f"  ✓ Found {len(parsed)} injuries")
            time.sleep(1.5)  # polite delay for rate limiting
            return parsed

        except Exception as e:
            print(f"  ❌ Error fetching injuries: {e}")
            return []
    
    def backfill_historical_injuries(self, start_date, end_date):
        """
        Backfill injuries for date range (for current 25-26 season).
        
        Args:
            start_date: 'YYYY-MM-DD' (e.g., '2025-10-21')
            end_date: 'YYYY-MM-DD' (e.g., '2025-11-15')
        
        Example:
            scraper.backfill_historical_injuries('2025-10-21', '2025-11-15')
        """
        print(f"\n🏥 BACKFILLING INJURIES: {start_date} to {end_date}")
        
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        
        current = start
        total_injuries = 0
        
        while current <= end:
            date_str = current.strftime('%Y-%m-%d')
            injuries = self.get_injuries_for_date(date_str)
            total_injuries += len(injuries)
            
            current += timedelta(days=1)
        
        print(f"\n✅ BACKFILL COMPLETE")
        print(f"   Cached {total_injuries} injury records")
        print(f"   Location: {self.cache_dir}")
    
    def get_current_injuries(self):
        """
        Get today's injuries for live predictions.
        
        Returns:
            Dict mapping team names to injured players
        """
        today = datetime.now().strftime('%Y-%m-%d')
        injuries = self.get_injuries_for_date(today)
        
        # Convert to team -> players dict
        injury_dict = {}
        for injury in injuries:
            team = injury['team']
            if team not in injury_dict:
                injury_dict[team] = []
            
            # Only include OUT and DOUBTFUL
            if injury['status'] in ['OUT', 'DOUBTFUL']:
                injury_dict[team].append({
                    'name': injury['player'],
                    'status': injury['status'],
                    'reason': injury['reason']
                })
        
        return injury_dict
    
    def load_injuries_for_game_date(self, game_date):
        """
        Load cached injuries for a specific game date.
        Used during data preparation for historical games.
        
        Args:
            game_date: 'YYYY-MM-DD'
        
        Returns:
            Dict of {team: [players]} or empty dict if not found
        """
        cache_file = self.cache_dir / f"injuries_{game_date}.json"
        
        if not cache_file.exists():
            return {}
        
        with open(cache_file, 'r') as f:
            injuries = json.load(f)
        
        # Convert to team -> players format
        injury_dict = {}
        for injury in injuries:
            team = injury['team']
            if team not in injury_dict:
                injury_dict[team] = []
            
            if injury['status'] in ['OUT', 'DOUBTFUL']:
                injury_dict[team].append({
                    'name': injury['player'],
                    'status': injury['status'],
                    'reason': injury['reason']
                })
        
        return injury_dict


if __name__ == "__main__":
    """
    Usage:
    
    # Backfill current season (2025-26):
    python scripts/rapidapi_injury_scraper.py 2025-10-21 2025-11-15
    
    # Get today's injuries:
    python scripts/rapidapi_injury_scraper.py
    """
    import sys
    
    scraper = RapidAPIInjuryScraper()
    
    if len(sys.argv) == 3:
        # Backfill mode
        start = sys.argv[1]
        end = sys.argv[2]
        scraper.backfill_historical_injuries(start, end)
    else:
        # Current injuries
        injuries = scraper.get_current_injuries()
        print("\n📋 TODAY'S INJURIES:")
        for team, players in injuries.items():
            print(f"\n{team}:")
            for player in players:
                print(f"  - {player['name']} ({player['status']}) - {player['reason']}")