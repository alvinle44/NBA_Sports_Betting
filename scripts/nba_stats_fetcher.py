"""
NBA Stats Fetcher - Improved Version
Fetches player statistics from the official NBA API with robust error handling.

FIXES:
- Handles different column name formats from NBA API
- Better error messages and debugging
- Validates season format
"""

import pandas as pd
import numpy as np
from datetime import datetime
import time
import requests
from pathlib import Path

from scripts.config import Config


class NBAStatsFetcher:
    """
    Fetches player statistics from NBA API.
    
    Features:
    - Robust column name handling
    - Last 5 game averages for each stat
    - Consistency metrics
    - Rate limiting built-in
    """
    
    def __init__(self):
        """Initialize the NBA stats fetcher."""
        self.base_url = "https://stats.nba.com/stats"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
            'Accept': 'application/json',
            'Referer': 'https://stats.nba.com/',
            'Origin': 'https://stats.nba.com',
            'x-nba-stats-origin': 'stats',
            'x-nba-stats-token': 'true'
        }
        
        # Map market names to NBA stat columns
        self.stat_mapping = {
            'player_points': 'PTS',
            'player_assists': 'AST',
            'player_rebounds': 'REB',
            'player_threes': 'FG3M',
            'player_steals': 'STL',
            'player_blocks': 'BLK',
            'player_turnovers': 'TOV',
        }
    
    def fetch_player_stats_for_list(self, player_names, season='2025-26', last_n_games=10):
        """
        Fetch stats for a specific list of players.
        
        Args:
            player_names: List of player names
            season: NBA season (e.g., '2024-25' for 2024-2025 season)
            last_n_games: Number of recent games to analyze
        
        Returns:
            DataFrame with player stats and calculated features
        """
        print(f"\n📊 Fetching stats for {len(player_names)} specific players...")
        print(f"   Season: {season}")
        print(f"   Analyzing last {last_n_games} games")
        
        try:
            # Validate season format
            if not self._validate_season(season):
                print(f"   ⚠️  Invalid season format: {season}")
                print(f"   💡 Use format '2024-25' for 2024-2025 season")
                return pd.DataFrame()
            
            # Get all players
            all_players = self._get_all_players(season)
            
            if all_players.empty:
                print("   ❌ Could not fetch player list from NBA API")
                return pd.DataFrame()
            
            print(f"   ✓ Fetched {len(all_players)} players from NBA API")
            
            # Find player name and ID columns
            name_col, id_col = self._find_key_columns(all_players)
            
            if not name_col or not id_col:
                print(f"   ❌ Could not identify player columns")
                print(f"   📋 Available columns: {list(all_players.columns)}")
                return pd.DataFrame()
            
            print(f"   ✓ Using columns: name='{name_col}', id='{id_col}'")
            
            # Standardize names for matching
            all_players['name_clean'] = all_players[name_col].str.upper().str.strip()
            target_names = [name.upper().strip() for name in player_names]
            
            # Filter to only requested players
            players_to_fetch = all_players[all_players['name_clean'].isin(target_names)]
            
            if players_to_fetch.empty:
                print("   ❌ None of the requested players found in NBA database")
                print(f"   💡 Sample NBA names: {all_players[name_col].head(3).tolist()}")
                return pd.DataFrame()
            
            match_count = len(players_to_fetch)
            total_count = len(player_names)
            print(f"   ✓ Found {match_count}/{total_count} matching players ({match_count/total_count*100:.1f}%)")
            
            # Show unmatched players
            if match_count < total_count:
                matched_names = set(players_to_fetch['name_clean'])
                unmatched = [name for name in target_names if name not in matched_names]
                print(f"   ⚠️  Unmatched players (showing first 5): {unmatched[:5]}")
            
            # Fetch game logs for each player
            all_stats = []
            
            for idx, player in players_to_fetch.iterrows():
                player_id = player[id_col]
                player_name = player[name_col]
                
                try:
                    stats = self._fetch_player_game_logs(
                        player_id=player_id,
                        player_name=player_name,
                        season=season,
                        last_n_games=last_n_games
                    )
                    
                    if stats is not None:
                        all_stats.append(stats)
                        print(f"   ✓ {player_name}")
                    else:
                        print(f"   ⚠️  No recent games for {player_name}")
                    
                    # Rate limiting
                    time.sleep(0.6)
                
                except Exception as e:
                    print(f"   ⚠️  Error fetching {player_name}: {str(e)[:50]}")
                    continue
            
            if not all_stats:
                print("   ❌ No stats fetched for any players")
                return pd.DataFrame()
            
            # Combine all player stats
            stats_df = pd.concat(all_stats, ignore_index=True)
            
            print(f"   ✅ Successfully fetched stats for {len(stats_df)} players")
            
            return stats_df
        
        except Exception as e:
            print(f"   ❌ Error fetching player stats: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()
    
    def fetch_all_player_stats(self, season='2025-26', last_n_games=10):
        """Fetch stats for all active NBA players."""
        print(f"\n📊 Fetching stats for all active players...")
        
        all_players = self._get_all_players(season)
        
        if all_players.empty:
            return pd.DataFrame()
        
        name_col, _ = self._find_key_columns(all_players)
        
        if not name_col:
            return pd.DataFrame()
        
        player_names = all_players[name_col].tolist()
        
        return self.fetch_player_stats_for_list(
            player_names=player_names,
            season=season,
            last_n_games=last_n_games
        )
    
    def _validate_season(self, season):
        """Validate season format (should be 'YYYY-YY')."""
        try:
            parts = season.split('-')
            if len(parts) != 2:
                return False
            
            year1 = int(parts[0])
            year2 = int(parts[1])
            
            # Check reasonable year range and sequential
            if year1 < 2000 or year1 > 2100:
                return False
            if year2 != (year1 + 1) % 100:
                return False
            
            return True
        except:
            return False
    
    def _find_key_columns(self, df):
        """
        Find player name and ID columns in DataFrame.
        NBA API uses different column names in different endpoints.
        """
        # Possible name columns (in order of preference)
        name_candidates = ['PLAYER_NAME', 'DISPLAY_FIRST_LAST', 'PLAYER', 'NAME']
        id_candidates = ['PLAYER_ID', 'PERSON_ID', 'ID']
        
        name_col = None
        id_col = None
        
        for col in name_candidates:
            if col in df.columns:
                name_col = col
                break
        
        for col in id_candidates:
            if col in df.columns:
                id_col = col
                break
        
        return name_col, id_col
    
    def _get_all_players(self, season='2025-26'):
        """Get list of all active NBA players."""
        url = f"{self.base_url}/leaguedashplayerstats"
        
        params = {
            'College': '',
            'Conference': '',
            'Country': '',
            'DateFrom': '',
            'DateTo': '',
            'Division': '',
            'DraftPick': '',
            'DraftYear': '',
            'GameScope': '',
            'GameSegment': '',
            'Height': '',
            'LastNGames': '0',
            'LeagueID': '00',
            'Location': '',
            'MeasureType': 'Base',
            'Month': '0',
            'OpponentTeamID': '0',
            'Outcome': '',
            'PORound': '0',
            'PaceAdjust': 'N',
            'PerMode': 'PerGame',
            'Period': '0',
            'PlayerExperience': '',
            'PlayerPosition': '',
            'PlusMinus': 'N',
            'Rank': 'N',
            'Season': season,
            'SeasonSegment': '',
            'SeasonType': 'Regular Season',
            'ShotClockRange': '',
            'StarterBench': '',
            'TeamID': '0',
            'TwoWay': '0',
            'VsConference': '',
            'VsDivision': '',
            'Weight': ''
        }
        
        try:
            response = requests.get(url, headers=self.headers, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if 'resultSets' not in data or len(data['resultSets']) == 0:
                return pd.DataFrame()
            
            headers = data['resultSets'][0]['headers']
            rows = data['resultSets'][0]['rowSet']
            
            df = pd.DataFrame(rows, columns=headers)
            
            # Filter to players with at least 1 game played
            if 'GP' in df.columns:
                df = df[df['GP'] > 0]
            
            return df
        
        except Exception as e:
            print(f"   ⚠️  Error fetching player list: {e}")
            return pd.DataFrame()
    
    def _fetch_player_game_logs(self, player_id, player_name, season='2025-26', last_n_games=10):
        """Fetch game logs for a specific player and calculate features."""
        url = f"{self.base_url}/playergamelog"
        
        params = {
            'PlayerID': player_id,
            'Season': season,
            'SeasonType': 'Regular Season',
            'LeagueID': '00'
        }
        
        try:
            response = requests.get(url, headers=self.headers, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if 'resultSets' not in data or len(data['resultSets']) == 0:
                return None
            
            headers = data['resultSets'][0]['headers']
            rows = data['resultSets'][0]['rowSet']
            
            if not rows:
                return None
            
            game_logs = pd.DataFrame(rows, columns=headers)
            
            # Sort by game date (most recent first)
            if 'GAME_DATE' in game_logs.columns:
                game_logs = game_logs.sort_values('GAME_DATE', ascending=False)
            
            # Take last N games
            recent_games = game_logs.head(last_n_games)
            
            if len(recent_games) < 3:  # Need at least 3 games
                return None
            
            # Calculate comprehensive features
            features = {
                'PLAYER_ID': player_id,
                'PLAYER_NAME': player_name,
                'games_in_last_week': len(recent_games),
            }
            
            # Different time windows
            last_5 = recent_games.head(5)
            last_10 = recent_games.head(10)
            last_3 = recent_games.head(3)
            older_games = recent_games.iloc[5:10] if len(recent_games) > 5 else last_5
            
            # Calculate stats for each market type
            for market, stat_col in self.stat_mapping.items():
                if stat_col in recent_games.columns:
                    # Basic averages
                    features[f'avg_last_5_{market}'] = last_5[stat_col].mean()
                    features[f'avg_last_10_{market}'] = last_10[stat_col].mean() if len(last_10) >= 5 else last_5[stat_col].mean()
                    features[f'last_game_{market}'] = recent_games.iloc[0][stat_col]
                    
                    # Consistency
                    avg = last_5[stat_col].mean()
                    std = last_5[stat_col].std()
                    if avg > 0:
                        cv = std / avg
                        consistency = max(0, min(1, 1 - cv))
                    else:
                        consistency = 0.5
                    features[f'consistency_L5_{market}'] = consistency
                    
                    # Trends
                    recent_avg = last_3[stat_col].mean()
                    older_avg = older_games[stat_col].mean()
                    features[f'trend_L3_{market}'] = recent_avg - older_avg if older_avg > 0 else 0
                    features[f'recent_vs_older_{market}'] = recent_avg / older_avg if older_avg > 0 else 1.0
                    
                    # Peak/momentum
                    features[f'peak_recent_{market}'] = last_5[stat_col].max()
                    features[f'momentum_score_{market}'] = (recent_avg - avg) / std if std > 0 else 0
            
            # Minutes-based features
            if 'MIN' in recent_games.columns:
                features['avg_minutes'] = last_5['MIN'].mean()
                features['MIN_L5'] = last_5['MIN'].mean()
                features['minutes_trend'] = last_3['MIN'].mean() - older_games['MIN'].mean() if len(older_games) > 0 else 0
                
                # Per-minute stats
                total_min = last_5['MIN'].sum()
                if total_min > 0:
                    if 'PTS' in last_5.columns:
                        features['pts_per_minute'] = last_5['PTS'].sum() / total_min
                    if 'AST' in last_5.columns:
                        features['ast_per_minute'] = last_5['AST'].sum() / total_min
                    if 'REB' in last_5.columns:
                        features['reb_per_minute'] = last_5['REB'].sum() / total_min
            else:
                features['avg_minutes'] = 30.0
                features['MIN_L5'] = 30.0
                features['minutes_trend'] = 0
                features['pts_per_minute'] = 0.6
                features['ast_per_minute'] = 0.2
                features['reb_per_minute'] = 0.3
            
            # Shooting efficiency
            if 'FG_PCT' in last_5.columns:
                features['fg_pct_L5'] = last_5['FG_PCT'].mean()
                features['fg_pct_trend'] = last_3['FG_PCT'].mean() - older_games['FG_PCT'].mean() if len(older_games) > 0 else 0
            else:
                features['fg_pct_L5'] = 0.45
                features['fg_pct_trend'] = 0
            
            if 'FG3_PCT' in last_5.columns:
                features['FG3_PCT_L5'] = last_5['FG3_PCT'].mean() * 100
                features['three_point_pct_L5'] = last_5['FG3_PCT'].mean()
            else:
                features['FG3_PCT_L5'] = 35.0
                features['three_point_pct_L5'] = 0.35
            
            if 'FG3A' in last_5.columns:
                features['FG3A_L5'] = last_5['FG3A'].mean()
                features['three_point_attempts_L5'] = last_5['FG3A'].mean()
            else:
                features['FG3A_L5'] = 5.0
                features['three_point_attempts_L5'] = 5.0
            
            if 'FTA' in last_5.columns and 'FGA' in last_5.columns:
                total_fga = last_5['FGA'].sum()
                features['fta_rate_L5'] = last_5['FTA'].sum() / total_fga if total_fga > 0 else 0.2
            else:
                features['fta_rate_L5'] = 0.2
            
            # Rebounding breakdown
            if 'OREB' in last_5.columns:
                features['oreb_L5'] = last_5['OREB'].mean()
                features['oreb_rate'] = last_5['OREB'].mean() / last_5['REB'].mean() if 'REB' in last_5.columns and last_5['REB'].mean() > 0 else 0.25
            else:
                features['oreb_L5'] = 1.0
                features['oreb_rate'] = 0.25
            
            if 'DREB' in last_5.columns:
                features['dreb_L5'] = last_5['DREB'].mean()
                features['dreb_rate'] = last_5['DREB'].mean() / last_5['REB'].mean() if 'REB' in last_5.columns and last_5['REB'].mean() > 0 else 0.75
            else:
                features['dreb_L5'] = 3.0
                features['dreb_rate'] = 0.75
            
            # Assist-to-turnover ratio
            if 'AST' in last_5.columns and 'TOV' in last_5.columns:
                total_tov = last_5['TOV'].sum()
                features['ast_to_tov_ratio'] = last_5['AST'].sum() / total_tov if total_tov > 0 else 2.0
            else:
                features['ast_to_tov_ratio'] = 2.0
            
            # Home/away split (placeholder - would need actual game location data)
            features['home_away_split'] = 1.0
            
            # Context features (placeholder - would need opponent/situation data)
            features['context_avg'] = features[f'avg_last_5_{list(self.stat_mapping.keys())[0]}']
            features['context_consistency'] = 0.8
            features['vs_opponent_avg'] = features[f'avg_last_5_{list(self.stat_mapping.keys())[0]}']
            features['vs_opponent_trend'] = 0.0
            
            return pd.DataFrame([features])
        
        except Exception as e:
            return None


if __name__ == "__main__":
    """Test the fetcher."""
    print("\n" + "="*70)
    print("🧪 NBA STATS FETCHER TEST")
    print("="*70)
    
    fetcher = NBAStatsFetcher()
    
    # Test with sample players
    test_players = [
        'LeBron James',
        'Stephen Curry',
        'Nikola Jokic',
    ]
    
    print(f"\n🎯 Testing with {len(test_players)} sample players")
    
    stats = fetcher.fetch_player_stats_for_list(
        player_names=test_players,
        season='2025-26'  # Current season
    )
    
    if not stats.empty:
        print("\n✅ SUCCESS! Sample results:")
        print(stats[['PLAYER_NAME', 'avg_last_5_player_points', 
                     'avg_last_5_player_assists']].to_string(index=False))
        
        cache_file = Config.DATA_DIR / "test_stats_cache.csv"
        stats.to_csv(cache_file, index=False)
        print(f"\n💾 Saved to: {cache_file}")
    else:
        print("\n❌ No stats fetched")
