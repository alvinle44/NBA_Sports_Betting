# scripts/live_odds_scraper.py

import time
import requests
import pandas as pd
from scripts.config import Config

class LiveOddsScraper:
    """
    Fetch LIVE (current) props from The Odds API for today's games.
    """
    
    def __init__(self, api_key=None):
        """Initialize scraper with API key."""
        self.api_key = api_key or Config.ODDS_API_KEY
        self.base_url = "https://api.the-odds-api.com/v4"
    
    def get_todays_props(self):
        """
        Fetch player props for upcoming games.
        Uses the correct event-specific endpoint.
        """
        print("\n📡 Fetching props from The Odds API...")
        
        try:
            # Step 1: Get upcoming games
            events_url = f"{self.base_url}/sports/basketball_nba/events"
            
            events_response = requests.get(
                events_url, 
                params={'apiKey': self.api_key},
                timeout=30
            )
            
            if events_response.status_code != 200:
                print(f"   ❌ API Error getting events: {events_response.status_code}")
                return pd.DataFrame()
            
            events = events_response.json()
            
            if not events:
                print("   ⚠️  No upcoming games")
                return pd.DataFrame()
            
            # Filter to games in next 48 hours
            from datetime import datetime
            now = datetime.now()
            upcoming_games = []
            
            for event in events:
                game_time = datetime.fromisoformat(event['commence_time'].replace('Z', '+00:00'))
                hours_until = (game_time - now.replace(tzinfo=game_time.tzinfo)).total_seconds() / 3600
                
                if 0 < hours_until < 48:
                    upcoming_games.append(event)
            
            print(f"   ✓ Found {len(upcoming_games)} upcoming games")
            
            # Step 2: Fetch props for each game
            all_props = []
            
            # Include combo markets
            all_markets = Config.MARKETS + [
                'player_points_rebounds_assists',
                'player_points_rebounds',
                'player_points_assists',
                'player_rebounds_assists'
            ]
            
            for game in upcoming_games:
                event_id = game['id']
                
                # Fetch props for this specific game
                odds_url = f"{self.base_url}/sports/basketball_nba/events/{event_id}/odds"
                
                odds_params = {
                    'apiKey': self.api_key,
                    'regions': 'us',
                    'markets': ','.join(all_markets),
                    'bookmakers': ','.join(Config.BOOKMAKERS),
                    'oddsFormat': 'american'
                }
                
                odds_response = requests.get(odds_url, params=odds_params, timeout=30)
                
                if odds_response.status_code != 200:
                    continue  # Props not available for this game yet
                
                odds_data = odds_response.json()
                
                # Parse props for this game
                game_props = self._parse_single_game_props(odds_data, game)
                all_props.extend(game_props)
                
                time.sleep(0.3)  # Rate limiting
            
            if not all_props:
                print("   ⚠️  No props available yet (typically released 2-4 hours before tip-off)")
                return pd.DataFrame()
            
            # Show quota after all requests
            quota_used = odds_response.headers.get('x-requests-used', 'Unknown')
            quota_remaining = odds_response.headers.get('x-requests-remaining', 'Unknown')
            print(f"   📊 API Quota: {quota_used} used, {quota_remaining} remaining")
            
            props_df = pd.DataFrame(all_props)
            print(f"   ✅ Fetched {len(props_df)} props")
            
            return props_df
        
        except Exception as e:
            print(f"   ❌ Error fetching props: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()
    
    def _parse_single_game_props(self, odds_data, game):
        """Parse props from a single game's odds data."""
        props = []
        
        for bookmaker in odds_data.get('bookmakers', []):
            bookmaker_key = bookmaker.get('key', '')
            
            if bookmaker_key not in Config.BOOKMAKERS:
                continue
            
            for market in bookmaker.get('markets', []):
                market_key = market.get('key', '')
                
                # Group outcomes by player
                outcomes_by_player = {}
                
                for outcome in market.get('outcomes', []):
                    player_name = outcome.get('description', '')
                    line = outcome.get('point')
                    bet_type = outcome.get('name', '')
                    odds = outcome.get('price', -110)
                    
                    if player_name not in outcomes_by_player:
                        outcomes_by_player[player_name] = {
                            'line': line,
                            'odds_over': None,
                            'odds_under': None
                        }
                    
                    if bet_type == 'Over':
                        outcomes_by_player[player_name]['odds_over'] = odds
                    elif bet_type == 'Under':
                        outcomes_by_player[player_name]['odds_under'] = odds

                # Create prop entries
                for player_name, data in outcomes_by_player.items():
                    props.append({
                        'game_id': game['id'],
                        'game_date': game['commence_time'][:10],
                        'home_team': game['home_team'],
                        'away_team': game['away_team'],
                        'bookmaker': bookmaker_key,
                        'market': market_key,
                        'player_name': player_name,
                        'line': data['line'],
                        'odds_over': data['odds_over'] if data['odds_over'] else -110,
                        'odds_under': data['odds_under'] if data['odds_under'] else -110,
                    })

        return props

    def to_long_format(self, wide_df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert wide props DataFrame (odds_over / odds_under columns) to the
        same long format used by historical_odds_combined.csv:
          game_id, game_date, home_team, away_team, bookmaker, market,
          player_name, bet_type, line, odds, timestamp
        """
        if wide_df.empty:
            return pd.DataFrame()

        ts = pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        rows = []
        for _, r in wide_df.iterrows():
            base = {
                "game_id":    r.get("game_id", ""),
                "game_date":  r.get("game_date", ""),
                "home_team":  r.get("home_team", ""),
                "away_team":  r.get("away_team", ""),
                "bookmaker":  r.get("bookmaker", ""),
                "market":     r.get("market", ""),
                "player_name": r.get("player_name", ""),
                "line":       r.get("line"),
                "timestamp":  ts,
            }
            if pd.notna(r.get("odds_over")):
                rows.append({**base, "bet_type": "Over",  "odds": r["odds_over"]})
            if pd.notna(r.get("odds_under")):
                rows.append({**base, "bet_type": "Under", "odds": r["odds_under"]})
        return pd.DataFrame(rows)