import requests          # Make HTTP requests to APIs
import pandas as pd      # Data manipulation (like Excel for Python)
import numpy as np       # Numerical operations and math
from datetime import datetime, timedelta  # Handle dates and times
import pickle           # Save/load Python objects (models, cache)
import time             # Add delays for rate limiting
import warnings         # Suppress unnecessary warnings
import json             # Read/write JSON files
from scripts.config import Config
from scripts.rapidapi_injury_scrapper import RapidAPIInjuryScraper
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

class NBAResultsScraper:
    """
    Scrapes actual NBA game results and player stats.
    
    This does three main things:
    1. Gets player game logs (actual stats from games)
    2. Gets team defensive stats (both season and recent)
    3. Scrapes injury reports from ESPN automatically
    
    Uses smart caching to avoid redundant API calls.
    """
    
    def __init__(self):
        """Initialize with empty caches for speed."""
        if not NBA_API_AVAILABLE:
            raise ImportError("nba_api required. Install: pip install nba-api")
        
        # Caches - store data to avoid re-fetching
        self.player_cache = {}        # player_name → player_id
        self.team_stats_cache = {}    # team defensive stats
        self.injury_cache = {}        # current injuries
        self.player_stats_cache = {}  # player game logs
        self.injury_scraper = RapidAPIInjuryScraper()  # NEW!
    
    def load_cached_player_stats(self, season="2025-26"):
        """
        Load player stats from disk cache if less than 24 hours old.
        
        Saves ~30 seconds on repeated runs same day.
        
        Returns:
            True if cache loaded, False if needs fresh data
        """
        cache_file = Config.DATA_DIR / f'player_stats_cache_{season}.pkl'
        
        if cache_file.exists():
            # Check if cache is recent (< 24 hours old)
            file_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
            if file_age.total_seconds() < 86400:  # 86400 seconds = 24 hours
                try:
                    with open(cache_file, 'rb') as f:
                        self.player_stats_cache = pickle.load(f)
                    print(f"Loaded cached stats for {len(self.player_stats_cache)} players")
                    return True
                except:
                    pass
        return False
    
    def save_player_stats_cache(self, season="2025-26"):
        """Save player stats cache to disk for next run."""
        cache_file = Config.DATA_DIR / f'player_stats_cache_{season}.pkl'
        with open(cache_file, 'wb') as f:
            pickle.dump(self.player_stats_cache, f)
        print(f"Saved stats cache for {len(self.player_stats_cache)} players")
    
    def get_player_id(self, player_name):
        """
        Convert player name to NBA API player ID.
        
        NBA API requires numeric IDs (e.g., LeBron = 2544).
        This function looks up the ID and caches it.
        
        Args:
            player_name: Full name like "LeBron James"
        
        Returns:
            Integer player ID or None if not found
        """
        # Check cache first
        if player_name in self.player_cache:
            return self.player_cache[player_name]
        
        # Get all NBA players
        all_players = players.get_players()
        
        # Find matching player (case-insensitive)
        player = [p for p in all_players if p['full_name'].lower() == player_name.lower()]
        
        if player:
            player_id = player[0]['id']
            self.player_cache[player_name] = player_id  # Cache it
            return player_id
        return None
    
    def get_team_id(self, team_name):
        """
        Convert team name to NBA API team ID.
        
        Similar to get_player_id but for teams.
        """
        all_teams = teams.get_teams()
        team = [t for t in all_teams if t['full_name'] == team_name or t['nickname'] == team_name]
        if team:
            return team[0]['id']
        return None
    
    def get_player_position(self, player_name):
        """
        AUTOMATICALLY detect player's position from NBA API + stats.
        
        No manual mapping needed! Works for any player.
        
        Process:
        1. Try NBA API commonplayerinfo (has position field)
        2. If "Guard", use assists to determine PG vs SG
        3. If "Forward", use rebounds to determine SF vs PF
        4. If "Center", return C
        5. Fallback to stats-based inference if API fails
        
        Returns:
            Position string: PG, SG, SF, PF, or C
        """
        try:
            player_id = self.get_player_id(player_name)
            if not player_id:
                return 'SF'  # Default if player not found
            
            # Get player info from NBA API
            player_info = commonplayerinfo.CommonPlayerInfo(player_id=player_id)
            info_df = player_info.get_data_frames()[0]
            
            if not info_df.empty and 'POSITION' in info_df.columns:
                position = info_df['POSITION'].iloc[0]
                position_upper = position.upper()
                
                # NBA API returns "Guard", "Forward", "Center" (not specific enough)
                # Use stats to determine specific position
                
                if 'GUARD' in position_upper:
                    # Determine PG vs SG based on assists
                    stats_df = self.get_player_game_stats(player_name, Config.CURRENT_SEASON)
                    if not stats_df.empty and len(stats_df) > 5:
                        avg_assists = stats_df.head(10)['AST'].mean()
                        avg_points = stats_df.head(10)['PTS'].mean()
                        
                        # Point guards have higher assist rates
                        # If assists > 6 OR assist/point ratio > 0.3 → PG
                        if avg_assists > 6 or (avg_assists / avg_points > 0.3 if avg_points > 0 else False):
                            return 'PG'
                        else:
                            return 'SG'
                    return 'SG'  # Default guard
                    
                elif 'FORWARD' in position_upper:
                    # Determine SF vs PF based on rebounds
                    stats_df = self.get_player_game_stats(player_name, Config.CURRENT_SEASON)
                    if not stats_df.empty and len(stats_df) > 5:
                        avg_rebounds = stats_df.head(10)['REB'].mean()
                        
                        # Power forwards get more rebounds (play closer to basket)
                        if avg_rebounds > 8:
                            return 'PF'
                        else:
                            return 'SF'
                    return 'SF'  # Default forward
                    
                elif 'CENTER' in position_upper:
                    return 'C'
            
            # FALLBACK: Infer position from stats if API doesn't have it
            stats_df = self.get_player_game_stats(player_name, Config.CURRENT_SEASON)
            if stats_df.empty or len(stats_df) < 5:
                return 'SF'  # Default
            
            recent = stats_df.head(10)
            avg_reb = recent['REB'].mean()
            avg_ast = recent['AST'].mean()
            avg_pts = recent['PTS'].mean()
            
            # Stat-based heuristics
            if avg_reb > 9:  # High rebounds = big man
                return 'PF' if avg_pts > 20 else 'C'
            elif avg_ast > 6:  # High assists = guard
                return 'PG' if avg_ast > 8 else 'SG'
            elif avg_reb > 6:  # Moderate rebounds = forward
                return 'SF'
            else:
                return 'SG'  # Default
            
        except Exception as e:
            print(f"Could not auto-detect position for {player_name}: {e}")
            # Last resort: check manual map for stars only
            manual_map = {
                'Stephen Curry': 'PG', 'Damian Lillard': 'PG',
                'Devin Booker': 'SG', 'Donovan Mitchell': 'SG',
                'LeBron James': 'SF', 'Kevin Durant': 'SF',
                'Giannis Antetokounmpo': 'PF', 'Anthony Davis': 'PF',
                'Nikola Jokic': 'C', 'Joel Embiid': 'C',
            }
            return manual_map.get(player_name, 'SF')
    
    def get_player_game_stats(self, player_name, season='2025-26', max_retries=3):
        """
        Fetch player game stats with retry logic and longer timeout.
        
        Args:
            player_name: Player's full name
            season: NBA season (e.g., '2024-25')
            max_retries: Number of retry attempts on timeout
        
        Returns:
            DataFrame with player's game stats
        """
        import time
        NAME_FIXES = {
        'Isaiah Stewart II': 'Isaiah Stewart',
        'Moe Wagner': 'Moritz Wagner',
        'K.J. Martin': 'KJ Martin',
        'Nicolas Claxton': 'Nic Claxton',
        'Herb Jones': 'Herbert Jones',
        'C.J. McCollum': 'CJ McCollum',
        'R.J. Barrett': 'RJ Barrett',
        'B.J. Boston Jr': 'BJ Boston',
        'A.J. Green': 'AJ Green',
        'Paul Reed Jr': 'Paul Reed',
        'Bruce Brown Jr': 'Bruce Brown',
        'G.G. Jackson': 'GG Jackson',
        'Kenyon Martin Jr.': 'Kenyon Martin Jr',
        'Xavier Tillman, Sr.': 'Xavier Tillman',
        'Carlton Carrington': 'Bub Carrington',
        'Mohamed Bamba':'Mo Bamba',
        'James Huff':'Jay Huff'

        # Add more as you find them
        }
        if player_name in NAME_FIXES:
            print(f"    🔧 Name fix: {player_name} → {NAME_FIXES[player_name]}")
            player_name = NAME_FIXES[player_name]
            
        for attempt in range(max_retries):
            try:
                # Add delay to avoid rate limiting
                if attempt > 0:
                    wait_time = attempt * 15  # 15s, 30s, 45s
                    print(f"    ⏱️  Retry {attempt + 1}/{max_retries} for {player_name} in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    # Small delay between all requests
                    time.sleep(0.6)
                
                # Get player ID
                from nba_api.stats.static import players
                player_dict = players.find_players_by_full_name(player_name)
                
                if not player_dict:
                    if attempt == 0:  # Only print once
                        print(f"    ⚠️  Player not found: {player_name}")
                    return pd.DataFrame()
                
                player_id = player_dict[0]['id']
                
                # Fetch game log with LONGER TIMEOUT
                from nba_api.stats.endpoints import playergamelog
                
                gamelog = playergamelog.PlayerGameLog(
                    player_id=player_id,
                    season=season,
                    timeout=90  # Increased from 30 to 90 seconds
                )
                
                df = gamelog.get_data_frames()[0]
                
                if not df.empty:
                    # Success!
                    if attempt > 0:
                        print(f"    ✓ Successfully fetched {player_name} on attempt {attempt + 1}")
                    return df
                
                return pd.DataFrame()
                
            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    print(f"    ⏱️  Timeout for {player_name} (attempt {attempt + 1}/{max_retries})")
                    continue  # Try again
                else:
                    print(f"    ❌ Failed to fetch {player_name} after {max_retries} attempts (timeout)")
                    return pd.DataFrame()
            
            except Exception as e:
                if attempt == 0:  # Only print once
                    print(f"    ❌ Error fetching {player_name}: {str(e)[:100]}")
                if attempt < max_retries - 1:
                    time.sleep(5)
                    continue
                return pd.DataFrame()
        
        return pd.DataFrame()
    
    def get_team_defensive_stats(self, team_name, season="2025-26", last_n_games=10, as_of_date=None):
        """
        Get team's defensive statistics with as_of_date support.
        
        ✅ FIXED: Uses correct NBA API columns
        """
        team_id = self.get_team_id(team_name)
        if not team_id:
            return None
        
        try:
            # Get team game logs
            from nba_api.stats.endpoints import teamgamelogs
            
            game_logs = teamgamelogs.TeamGameLogs(
                season_nullable=season,
                team_id_nullable=team_id,
                season_type_nullable='Regular Season'
            )
            
            games_df = game_logs.get_data_frames()[0]
            
            if games_df.empty:
                return None
            
            # ✅ Filter to games BEFORE as_of_date
            if as_of_date:
                games_df['GAME_DATE'] = pd.to_datetime(games_df['GAME_DATE'])
                as_of_date_dt = pd.to_datetime(as_of_date)
                games_df = games_df[games_df['GAME_DATE'] < as_of_date_dt]
                
                if games_df.empty:
                    return None
            
            # Sort by date (most recent first)
            games_df = games_df.sort_values('GAME_DATE', ascending=False)
            
            # ✅ FIX: Calculate opponent points from game logs
            # PTS_DIFF = Team points - Opponent points
            # So: Opponent points = Team points - PTS_DIFF
            
            if 'PTS' in games_df.columns and 'PLUS_MINUS' in games_df.columns:
                # opponent_pts = team_pts - plus_minus
                games_df['OPP_PTS'] = games_df['PTS'] - games_df['PLUS_MINUS']
            elif 'PTS' in games_df.columns and 'W_PCT' in games_df.columns:
                # Estimate opponent points (rough approximation)
                games_df['OPP_PTS'] = games_df['PTS'] * 0.95  # Placeholder
            else:
                # Can't calculate, use defaults
                return {
                    'def_rating_weighted': 110.0,
                    'def_rating_season': 110.0,
                    'def_rating_last_10': 110.0,
                    'pace': 100.0,
                    'pace_season': 100.0,
                    'opp_pts_weighted': 110.0,
                    'opp_pts_per_game_season': 110.0,
                    'opp_pts_last_10': 110.0,
                    'games_used': len(games_df),
                    'recent_games_used': min(len(games_df), last_n_games)
                }
            
            # Calculate season averages
            season_opp_pts = games_df['OPP_PTS'].mean()
            
            # Calculate recent averages (last N games)
            if len(games_df) >= last_n_games:
                recent_games = games_df.head(last_n_games)
                recent_opp_pts = recent_games['OPP_PTS'].mean()
                
                # Defensive trend (negative = improving)
                if len(recent_games) >= 10:
                    def_trend = (
                        recent_games.head(5)['OPP_PTS'].mean() - 
                        recent_games.tail(5)['OPP_PTS'].mean()
                    )
                else:
                    def_trend = 0
            else:
                recent_opp_pts = season_opp_pts
                def_trend = 0
            
            # Estimate defensive rating (points per 100 possessions)
            # Simple estimate: (opp_pts / 100) * 100 = opp_pts as rating
            def_rating_season = season_opp_pts
            def_rating_recent = recent_opp_pts
            
            # Weighted average (70% recent, 30% season)
            def_rating_weighted = (0.7 * def_rating_recent) + (0.3 * def_rating_season)
            opp_pts_weighted = (0.7 * recent_opp_pts) + (0.3 * season_opp_pts)
            
            # Estimate pace (possessions per game)
            # Average NBA pace is ~100
            if 'FGA' in games_df.columns and 'FTA' in games_df.columns:
                # Pace estimate: FGA + 0.44 * FTA + TOV
                pace = 100.0  # Simplified for now
            else:
                pace = 100.0
            
            stats = {
                'def_rating_weighted': def_rating_weighted,
                'def_rating_season': def_rating_season,
                'def_rating_last_10': def_rating_recent,
                'pace': pace,
                'pace_season': pace,
                'opp_pts_weighted': opp_pts_weighted,
                'opp_pts_per_game_season': season_opp_pts,
                'opp_pts_last_10': recent_opp_pts,
                'def_trend': def_trend,
                'games_used': len(games_df),
                'recent_games_used': min(len(games_df), last_n_games)
            }
            
            if as_of_date:
                stats['as_of_date'] = str(as_of_date)
            
            return stats
            
        except Exception as e:
            print(f"Error fetching defensive stats for {team_name}: {e}")
            return None
    

    def get_player_stats_before_date(self, player_name, target_date, season='2025-26', last_n_games=10):
        """
        Get player's stats from games BEFORE a specific date.
        
        ✅ Ensures no data leakage - only uses prior games!
        
        Args:
            player_name: Player's full name
            target_date: Date to filter before (e.g., '2025-11-18')
            season: NBA season
            last_n_games: How many recent games to include
        
        Returns:
            Dict with player stats calculated from games BEFORE target_date
        
        Example:
            # For LeBron's Nov 18 game, get his stats from before Nov 18:
            stats = get_player_stats_before_date(
                player_name='LeBron James',
                target_date='2025-11-18',  # ✅ Only uses games before Nov 18
                season='2025-26',
                last_n_games=10
            )
        """
        # Get all game logs for the season
        all_games = self.get_player_game_stats(player_name, season)
        
        if all_games.empty:
            return None
        
        # Convert dates
        all_games['GAME_DATE'] = pd.to_datetime(all_games['GAME_DATE'])
        target_date_dt = pd.to_datetime(target_date)
        
        # ✅ Filter to games BEFORE target date
        prior_games = all_games[all_games['GAME_DATE'] < target_date_dt]
        
        if prior_games.empty:
            print(f"  ⚠️  No prior games found for {player_name} before {target_date}")
            return None
        
        # Sort by date (most recent first)
        prior_games = prior_games.sort_values('GAME_DATE', ascending=False)
        
        # Take last N games
        recent_games = prior_games.head(last_n_games)
        
        # Calculate stats
        stats = {
            'player_name': player_name,
            'as_of_date': str(target_date),
            'games_used': len(recent_games),
            'avg_pts': recent_games['PTS'].mean() if 'PTS' in recent_games else 0,
            'avg_ast': recent_games['AST'].mean() if 'AST' in recent_games else 0,
            'avg_reb': recent_games['REB'].mean() if 'REB' in recent_games else 0,
            'avg_min': recent_games['MIN'].mean() if 'MIN' in recent_games else 0,
            'last_game_pts': recent_games.iloc[0]['PTS'] if 'PTS' in recent_games else 0,
            'last_game_date': str(recent_games.iloc[0]['GAME_DATE'].date()),
        }
        
        return stats
    
    def scrape_espn_injuries(self):
        """
        AUTOMATICALLY scrape current injury report from ESPN.
        
        No manual work needed! Runs every time you make predictions.
        
        Process:
        1. Requests ESPN injury page
        2. Parses HTML tables with BeautifulSoup
        3. Finds players marked "OUT" or "DOUBTFUL"
        4. Returns dict: {team_name: [injured_players]}
        5. Saves to JSON file as backup
        
        Returns:
            Dict mapping team names to list of injured players
        """
        if not BS4_AVAILABLE:
            print("BeautifulSoup not available, loading from file")
            return self.load_injuries_from_file()
        
        print("\nScraping ESPN injury report...")
        
        url = "https://www.espn.com/nba/injuries"
        
        try:
            # Add headers to appear like a browser
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code != 200:
                print(f"Failed to fetch ESPN injuries: {response.status_code}")
                return self.load_injuries_from_file()  # Fallback to cached file
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            injuries = {}
            
            # ESPN uses tables wrapped in ResponsiveTable divs
            tables = soup.find_all('div', class_='ResponsiveTable')
            
            if not tables:
                # Try alternative structure
                tables = soup.find_all('table')
            
            for table in tables:
                # Find team name (appears before injury table)
                team_header = table.find_previous('div', class_='Table__Title')
                if not team_header:
                    team_header = table.find_previous('h2')
                
                if not team_header:
                    continue
                
                team_name = team_header.get_text(strip=True)
                # Clean team name (remove "Injuries" suffix if present)
                team_name = team_name.split('Injuries')[0].strip()
                
                injured_players = []
                
                # Find all rows in injury table
                rows = table.find_all('tr')
                
                for row in rows[1:]:  # Skip header row
                    cols = row.find_all('td')
                    
                    if len(cols) < 3:
                        continue
                    
                    player_name = cols[0].get_text(strip=True)
                    status = cols[2].get_text(strip=True) if len(cols) > 2 else ''
                    
                    # Only include if OUT or DOUBTFUL (miss game)
                    if 'OUT' in status.upper() or 'DOUBTFUL' in status.upper():
                        # Clean player name (remove position info if present)
                        player_name = player_name.split('\n')[0].strip()
                        injured_players.append(player_name)
                
                if injured_players:
                    injuries[team_name] = injured_players
            
            if not injuries:
                print("No injuries parsed from ESPN (site structure may have changed)")
                print("Falling back to cached injury file...")
                return self.load_injuries_from_file()
            
            # Save to file for backup
            self.save_injuries_to_file(injuries)
            
            print(f"✓ Found injuries for {len(injuries)} teams:")
            for team, players in injuries.items():
                print(f"  {team}: {', '.join(players)}")
            
            return injuries
            
        except Exception as e:
            print(f"Error scraping ESPN injuries: {e}")
            print("Falling back to cached injury file...")
            return self.load_injuries_from_file()
    
    def save_injuries_to_file(self, injuries):
        """
        Save scraped injuries to JSON file.
        
        Keeps historical record and provides fallback if ESPN fails.
        """
        injury_file = Config.DATA_DIR / 'injuries.json'
        today = datetime.now().strftime('%Y-%m-%d')
        
        # Load existing injuries
        if injury_file.exists():
            try:
                with open(injury_file, 'r') as f:
                    all_injuries = json.load(f)
            except:
                all_injuries = {}
        else:
            all_injuries = {}
        
        # Update with today's data
        all_injuries[today] = injuries
        
        # Save
        with open(injury_file, 'w') as f:
            json.dump(all_injuries, f, indent=2)
        
        print(f"✓ Saved injuries to {injury_file}")
    
    def load_injuries_from_file(self, game_date=None):
        """
        Load injuries from cached JSON file.
        
        Fallback if ESPN scraping fails.
        """
        injury_file = Config.DATA_DIR / 'injuries.json'
        
        if not injury_file.exists():
            print("No injury file found. Creating empty one...")
            return {}
        
        try:
            with open(injury_file, 'r') as f:
                all_injuries = json.load(f)
            
            # Get today's date or specified date
            if game_date:
                date_str = game_date[:10] if 'T' in game_date else game_date
            else:
                date_str = datetime.now().strftime('%Y-%m-%d')
            
            if date_str in all_injuries:
                return all_injuries[date_str]
            
            # If no data for today, get most recent
            dates = sorted(all_injuries.keys(), reverse=True)
            if dates:
                print(f"Using injury data from {dates[0]}")
                return all_injuries[dates[0]]
            
            return {}
            
        except Exception as e:
            print(f"Error loading injury file: {e}")
            return {}
    
    def get_team_injury_report(self, team_name, game_date):
        """
        Get injury report for specific team on specific date.
        
        Uses cached data from scrape_espn_injuries().
        
        Returns:
            List of injured player names
        """
        if not self.injury_cache:
            # Load from file if not in memory
            self.injury_cache = self.load_injuries_from_file(game_date)
        
        # Try to match team name (handles variations)
        for cached_team, injured in self.injury_cache.items():
            if team_name in cached_team or cached_team in team_name:
                return injured
        
        return []
    
    def calculate_teammate_impact(self, player_name, team_name, game_date, season="2024-25"):
        """
        Calculate impact of missing teammates on player's usage/opportunity.
        
        When star players are out, others get more shots/touches.
        
        Process:
        1. Check injury report for this team
        2. Count key players out
        3. Estimate usage bump (5% per key player out)
        4. Return opportunity multiplier
        
        Returns:
            Dict with:
            - key_players_out: Number of important teammates injured
            - usage_bump_expected: % increase in usage (e.g., 0.05 = 5%)
            - opportunity_score: Multiplier (1.0 = normal, 1.10 = 10% more opportunity)
        """
        try:
            # Get injury report
            injured_players = self.get_team_injury_report(team_name, game_date)
            
            # Simple calculation: More injuries = more opportunity
            # In production, you'd weight by player importance (stars vs role players)
            impact = {
                'key_players_out': len(injured_players),
                'injured_players': injured_players,
                'expected_usage_bump': len(injured_players) * 0.05,  # 5% per injury
                'opportunity_score': 1.0 + (len(injured_players) * 0.1),  # 10% boost per injury
            }
            
            return impact
            
        except Exception as e:
            print(f"Error calculating teammate impact: {e}")
            return {
                'key_players_out': 0,
                'injured_players': [],
                'expected_usage_bump': 0,
                'opportunity_score': 1.0
            }
    
    def get_results_for_date_range(self, start_date, end_date, players_list, season=None):
        """
        Fetch and save player game results for all players in the date range.
        Saves each player's data immediately after fetching so progress isn't lost.

        Args:
            start_date (str): Start date 'YYYY-MM-DD'
            end_date (str): End date 'YYYY-MM-DD'
            players_list (list[str]): List of player names
            season (str): Optional override season (e.g., '2025-26')

        Returns:
            pd.DataFrame: Combined DataFrame of all player stats.
        """
        import time, os, pandas as pd
        from pathlib import Path
        from datetime import datetime

        print(f"\n{'='*70}")
        print(f"Fetching NBA Game Results ({start_date} → {end_date})")
        print(f"{'='*70}")

        # --- 1️⃣ Determine season if not provided ---
        if season is None:
            start = datetime.strptime(start_date[:10], '%Y-%m-%d')
            if start.month >= 10:
                season = f"{start.year}-{str(start.year + 1)[-2:]}"
            else:
                season = f"{start.year - 1}-{str(start.year)[-2:]}"
        print(f"Season detected: {season}")

        # --- 2️⃣ Prepare save paths ---
        data_dir = Path(Config.DATA_DIR)
        player_dir = data_dir / "player_game_logs"
        player_dir.mkdir(parents=True, exist_ok=True)

        master_file = data_dir / f"game_results_{season}.csv"
        print(f"Data will be saved under: {data_dir.resolve()}")

        all_stats = []

        # --- 3️⃣ Loop through players ---
        for i, player in enumerate(players_list, 1):
            print(f"\n[{i}/{len(players_list)}] Fetching stats for {player}")

            try:
                stats = self.get_player_game_stats(player, season)
            except Exception as e:
                print(f"    ❌ Error fetching {player}: {e}")
                continue

            if not stats.empty:
                stats['PLAYER_NAME'] = player
                stats['GAME_DATE'] = pd.to_datetime(stats['GAME_DATE']).dt.strftime('%Y-%m-%d')

                # --- 4️⃣ Save this player's data immediately ---
                player_file = player_dir / f"{player.replace(' ', '_')}_{season}.csv"
                stats.to_csv(player_file, index=False)
                print(f"    💾 Saved {len(stats)} games for {player} → {player_file.name}")

                # --- 5️⃣ Append to master file incrementally ---
                if master_file.exists():
                    existing = pd.read_csv(master_file)
                    combined = pd.concat([existing, stats], ignore_index=True)
                else:
                    combined = stats.copy()

                combined.to_csv(master_file, index=False)
                print(f"    📈 Master file updated ({len(combined)} total rows).")

                all_stats.append(stats)

            else:
                print(f"    ⚠️ No data for {player}")

            time.sleep(1.2)  # Prevent hitting rate limits

        # --- 6️⃣ Combine everything in memory (optional return) ---
        if all_stats:
            full_df = pd.concat(all_stats, ignore_index=True)
            print(f"\n✓ Completed {len(players_list)} players total")
            print(f"✓ Master file saved at {master_file}")
            return full_df

        print("\n⚠️ No data fetched for this date range.")
        return pd.DataFrame()
    
    
    def get_team_injuries_on_date(self, team_code, game_date):
        """
        Get list of injured players for a team on a specific date.
        
        Args:
            team_code: Team code (e.g., 'LAL', 'GSW')
            game_date: Date as string 'YYYY-MM-DD' or datetime
        
        Returns:
            List of injured player names
        """
        if self.injury_df is None:
            return []
        
        # Convert game_date to datetime
        if isinstance(game_date, str):
            game_date = pd.to_datetime(game_date)
        
        # Filter for this team and date
        # Note: You'll need to match team codes properly
        # This is simplified - adjust based on your CSV format
        
        team_injuries = self.injury_df[
            (self.injury_df['Date'] <= game_date)
        ]
        
        # Extract player names
        # Adjust this based on your CSV column names
        if 'Relinquished' in team_injuries.columns:
            # Parse player name from "Player Name (TEAM)" format
            injured_players = []
            for _, row in team_injuries.iterrows():
                player_str = row['Relinquished']
                if isinstance(player_str, str) and team_code in player_str:
                    # Extract name before parentheses
                    player_name = player_str.split('(')[0].strip()
                    injured_players.append(player_name)
            
            return list(set(injured_players))  # Remove duplicates
        
        return []