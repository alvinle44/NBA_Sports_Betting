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
from config import Config
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
    
    def load_cached_player_stats(self, season="2024-25"):
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
    
    def save_player_stats_cache(self, season="2024-25"):
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
    
    def get_player_game_stats(self, player_name, season="2024-25"):
        """
        Get player's game-by-game statistics for a season.
        
        Returns DataFrame with columns like:
        - GAME_DATE: When game was played
        - PTS, AST, REB: Stats from that game
        - MIN: Minutes played
        - Plus many more stats
        
        Uses cache to avoid redundant API calls.
        """
        # Check cache first
        cache_key = f"{player_name}_{season}"
        if cache_key in self.player_stats_cache:
            return self.player_stats_cache[cache_key]
        
        player_id = self.get_player_id(player_name)
        if not player_id:
            return pd.DataFrame()
        
        try:
            # Fetch from NBA API
            gamelog = playergamelog.PlayerGameLog(
                player_id=player_id,
                season=season
            )
            df = gamelog.get_data_frames()[0]
            df['player_name'] = player_name
            
            # Cache it for next time
            self.player_stats_cache[cache_key] = df
            
            time.sleep(0.6)  # Rate limit - NBA API allows ~100/minute
            return df
        except Exception as e:
            print(f"Error fetching {player_name}: {e}")
            return pd.DataFrame()
    
    def get_team_defensive_stats(self, team_name, season="2024-25", last_n_games=10):
        """
        Get team's defensive statistics - BOTH season-long AND recent.
        
        This is KEY for accurate predictions! Returns:
        1. Season-long stats (stable baseline)
        2. Recent 10 games stats (current form)
        3. Weighted average (70% recent, 30% season)
        4. Defensive trend (improving or declining?)
        
        Why both?
        - Season = reliable but may be outdated
        - Recent = current form but can be noisy
        - Weighted = best of both worlds
        
        Args:
            team_name: NBA team name
            season: Season year
            last_n_games: How many recent games to analyze (default 10)
        
        Returns:
            Dict with 10+ defensive metrics
        """
        team_id = self.get_team_id(team_name)
        if not team_id:
            return None
        
        try:
            # ===== GET SEASON-LONG STATS =====
            team_stats = leaguedashteamstats.LeagueDashTeamStats(
                season=season,
                per_mode_detailed="PerGame",
                measure_type_detailed_defense="Defense"
            )
            
            df = team_stats.get_data_frames()[0]
            team_row = df[df['TEAM_ID'] == team_id]
            
            if team_row.empty:
                return None
            
            season_stats = {
                'def_rating_season': team_row['DEF_RATING'].values[0],
                'opp_pts_per_game_season': team_row['OPP_PTS'].values[0] if 'OPP_PTS' in team_row else 0,
                'pace_season': team_row['PACE'].values[0] if 'PACE' in team_row else 100,
            }
            
            # ===== GET RECENT STATS (Last N Games) =====
            game_logs = teamgamelogs.TeamGameLogs(
                season_nullable=season,
                team_id_nullable=team_id,
                season_type_nullable='Regular Season'
            )
            
            games_df = game_logs.get_data_frames()[0]
            
            if not games_df.empty and len(games_df) >= last_n_games:
                recent_games = games_df.head(last_n_games)
                
                # Calculate recent defensive performance
                recent_stats = {
                    'opp_pts_last_10': recent_games['OPP_PTS'].mean() if 'OPP_PTS' in recent_games else 0,
                    'opp_fg_pct_last_10': recent_games['OPP_FG_PCT'].mean() if 'OPP_FG_PCT' in recent_games else 0.45,
                    'opp_fg3_pct_last_10': recent_games['OPP_FG3_PCT'].mean() if 'OPP_FG3_PCT' in recent_games else 0.35,
                    
                    # DEFENSIVE TREND: Are they improving or declining?
                    # Compare last 5 games to previous 5 games
                    'def_trend': (
                        recent_games.head(5)['OPP_PTS'].mean() - 
                        recent_games.tail(5)['OPP_PTS'].mean()
                    ) if len(recent_games) >= 10 else 0,
                    # Negative trend = improving (allowing fewer points)
                    # Positive trend = declining (allowing more points)
                }
                
                # Calculate recent defensive rating estimate
                if 'POSS' in recent_games.columns:
                    recent_stats['def_rating_last_10'] = (
                        recent_games['OPP_PTS'].sum() / 
                        recent_games['POSS'].sum() * 100
                    )
                else:
                    # Estimate if possessions not available
                    pace = season_stats['pace_season']
                    recent_stats['def_rating_last_10'] = (
                        recent_stats['opp_pts_last_10'] / pace * 100
                    )
            else:
                # Not enough recent games, use season stats
                recent_stats = {
                    'opp_pts_last_10': season_stats['opp_pts_per_game_season'],
                    'opp_fg_pct_last_10': 0.45,
                    'opp_fg3_pct_last_10': 0.35,
                    'def_rating_last_10': season_stats['def_rating_season'],
                    'def_trend': 0,
                }
            
            # ===== COMBINE & CALCULATE WEIGHTED =====
            stats = {**season_stats, **recent_stats}
            
            # WEIGHTED AVERAGE (70% recent, 30% season)
            # This is the PRIMARY stat used for predictions!
            stats['def_rating_weighted'] = (
                0.7 * stats['def_rating_last_10'] + 
                0.3 * stats['def_rating_season']
            )
            
            stats['opp_pts_weighted'] = (
                0.7 * stats['opp_pts_last_10'] + 
                0.3 * stats['opp_pts_per_game_season']
            )
            
            # ===== POSITION-SPECIFIC DEFENSE =====
            # How well they defend guards vs forwards vs centers
            try:
                from nba_api.stats.endpoints import teamdashptshot
                position_def = teamdashptshot.TeamDashPtShot(
                    team_id=team_id,
                    season=season
                )
                pos_df = position_def.get_data_frames()[0]
                
                stats['def_vs_guards'] = pos_df[pos_df['PLAYER_POSITION'].str.contains('Guard', na=False)]['FG_PCT'].mean() if not pos_df.empty else 0.45
                stats['def_vs_forwards'] = pos_df[pos_df['PLAYER_POSITION'].str.contains('Forward', na=False)]['FG_PCT'].mean() if not pos_df.empty else 0.47
                stats['def_vs_centers'] = pos_df[pos_df['PLAYER_POSITION'].str.contains('Center', na=False)]['FG_PCT'].mean() if not pos_df.empty else 0.50
            except:
                # Fallback values if endpoint fails
                stats['def_vs_guards'] = 0.45
                stats['def_vs_forwards'] = 0.47
                stats['def_vs_centers'] = 0.50
            
            return stats
            
        except Exception as e:
            print(f"Error fetching defensive stats for {team_name}: {e}")
            return None
    
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
    
    def get_results_for_date_range(self, start_date, end_date, players_list):
        """
        Fetch actual game results for all players in date range.
        
        Used when preparing training data or tracking results.
        
        Args:
            start_date: Start date
            end_date: End date
            players_list: List of player names to fetch
        
        Returns:
            DataFrame with all game results
        """
        print(f"\n{'='*60}")
        print(f"Fetching NBA Game Results")
        print(f"{'='*60}")
        
        # Determine season from date
        start = datetime.strptime(start_date[:10], '%Y-%m-%d')
        if start.month >= 10:
            season = f"{start.year}-{str(start.year + 1)[-2:]}"
        else:
            season = f"{start.year - 1}-{str(start.year)[-2:]}"
        
        print(f"Season: {season}")
        print(f"Fetching stats for {len(players_list)} unique players...")
        
        all_stats = []
        for i, player in enumerate(players_list, 1):
            print(f"[{i}/{len(players_list)}] {player}")
            stats = self.get_player_game_stats(player, season)
            if not stats.empty:
                all_stats.append(stats)
        
        if all_stats:
            combined = pd.concat(all_stats, ignore_index=True)
            
            # Convert game date to match odds format
            combined['GAME_DATE'] = pd.to_datetime(combined['GAME_DATE']).dt.strftime('%Y-%m-%d')
            
            output_file = Config.DATA_DIR / f"game_results_{season}.csv"
            combined.to_csv(output_file, index=False)
            print(f"\n✓ Saved {len(combined)} game results to {output_file}")
            return combined
        
        return pd.DataFrame()