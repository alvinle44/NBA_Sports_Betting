import pandas as pd
import numpy as np
from scripts.config import Config

class ArbitrageDetector:
    """
    Detects arbitrage opportunities and finds best lines across bookmakers.
    
    Features:
    - Best line shopping (find best odds per side)
    - Arbitrage detection (guaranteed profit opportunities)
    - Expected value calculation
    - Line comparison across all bookmakers
    """
    
    def __init__(self):
        self.bookmakers = Config.BOOKMAKERS
    
    def american_to_decimal(self, american_odds):
        """
        Convert American odds to decimal odds.
        
        Examples:
            -110 → 1.909
            +100 → 2.000
            -200 → 1.500
            +150 → 2.500
        """
        if american_odds > 0:
            return (american_odds / 100) + 1
        else:
            return (100 / abs(american_odds)) + 1
    
    def american_to_implied_prob(self, american_odds):
        """
        Convert American odds to implied probability.
        
        Examples:
            -110 → 52.4%
            +100 → 50.0%
            -200 → 66.7%
        """
        if american_odds < 0:
            return abs(american_odds) / (abs(american_odds) + 100)
        else:
            return 100 / (american_odds + 100)
    
    def calculate_arbitrage(self, over_odds, under_odds):
        """
        Calculate if arbitrage opportunity exists.
        
        Args:
            over_odds: American odds for Over (e.g., -110)
            under_odds: American odds for Under (e.g., +100)
        
        Returns:
            Dict with arbitrage info
        """
        # Convert to implied probabilities
        over_prob = self.american_to_implied_prob(over_odds)
        under_prob = self.american_to_implied_prob(under_odds)
        
        # Total probability (should be > 100% normally due to vig)
        total_prob = over_prob + under_prob
        
        # Arbitrage exists when total < 100%
        has_arb = total_prob < 1.0
        
        if has_arb:
            # Calculate profit percentage
            profit_pct = (1 / total_prob - 1) * 100
            
            # Calculate optimal bet sizing (Kelly criterion)
            over_stake_pct = (1 - under_prob) / (1 - total_prob)
            under_stake_pct = (1 - over_prob) / (1 - total_prob)
            
            return {
                'has_arbitrage': True,
                'profit_pct': profit_pct,
                'over_stake_pct': over_stake_pct * 100,
                'under_stake_pct': under_stake_pct * 100,
                'over_implied_prob': over_prob * 100,
                'under_implied_prob': under_prob * 100,
                'total_implied_prob': total_prob * 100
            }
        else:
            # Calculate vig (bookmaker edge)
            vig = (total_prob - 1) * 100
            
            return {
                'has_arbitrage': False,
                'vig': vig,
                'over_implied_prob': over_prob * 100,
                'under_implied_prob': under_prob * 100,
                'total_implied_prob': total_prob * 100
            }
    
    def find_best_lines(self, props_df):
        """
        Find best odds for each prop across all bookmakers.
        
        Args:
            props_df: DataFrame with props from multiple bookmakers
                     Must have: player_name, market, line, bookmaker, odds_over, odds_under
        
        Returns:
            DataFrame with best lines and arbitrage opportunities
        """
        if props_df.empty:
            return pd.DataFrame()
        
        # Group by prop (player + market + line)
        props_df['prop_key'] = (
            props_df['player_name'] + '_' +
            props_df['market'] + '_' +
            props_df['line'].astype(str)
        )
        
        best_lines = []
        
        for prop_key, group in props_df.groupby('prop_key'):
            if len(group) < 2:
                continue  # Need at least 2 bookmakers for comparison
            
            # Get base info
            player = group.iloc[0]['player_name']
            market = group.iloc[0]['market']
            line = group.iloc[0]['line']
            
            # Find best odds for each side
            best_over = group.loc[group['odds_over'].idxmax()]
            best_under = group.loc[group['odds_under'].idxmax()]
            
            # Check for arbitrage
            arb_info = self.calculate_arbitrage(
                best_over['odds_over'],
                best_under['odds_under']
            )
            
            # Calculate line value vs average
            avg_over = group['odds_over'].mean()
            avg_under = group['odds_under'].mean()
            
            over_value = best_over['odds_over'] - avg_over
            under_value = best_under['odds_under'] - avg_under
            
            best_lines.append({
                'player_name': player,
                'market': market,
                'line': line,
                
                # Best Over
                'best_over_book': best_over['bookmaker'],
                'best_over_odds': best_over['odds_over'],
                'over_value': over_value,
                
                # Best Under
                'best_under_book': best_under['bookmaker'],
                'best_under_odds': best_under['odds_under'],
                'under_value': under_value,
                
                # Arbitrage info
                'has_arbitrage': arb_info['has_arbitrage'],
                'arb_profit_pct': arb_info.get('profit_pct', 0),
                'vig': arb_info.get('vig', 0),
                
                # All bookmakers for reference
                'all_books': group[['bookmaker', 'odds_over', 'odds_under']].to_dict('records'),
                'num_books': len(group)
            })
        
        best_lines_df = pd.DataFrame(best_lines)
        
        # Sort by arbitrage opportunities first, then by value
        best_lines_df = best_lines_df.sort_values(
            ['has_arbitrage', 'arb_profit_pct', 'over_value'],
            ascending=[False, False, False]
        )
        
        return best_lines_df
    
    def calculate_expected_value(self, odds, win_probability):
        """
        Calculate expected value (EV) of a bet.
        
        EV = (Win Probability × Profit) - (Loss Probability × Stake)
        
        Args:
            odds: American odds (e.g., -110)
            win_probability: Your estimated win probability (0-1)
        
        Returns:
            Expected value as percentage of stake
        """
        decimal_odds = self.american_to_decimal(odds)
        
        # Profit on $100 bet
        if odds > 0:
            profit = odds
        else:
            profit = 100 / (abs(odds) / 100)
        
        # Calculate EV
        ev = (win_probability * profit) - ((1 - win_probability) * 100)
        
        # Return as percentage
        return ev
    
    def find_value_bets(self, props_df, predictions_df):
        """
        Find value bets where your model's probability differs from market.
        
        Args:
            props_df: Current props with odds
            predictions_df: Your model's predictions with probabilities
        
        Returns:
            DataFrame with value betting opportunities
        """
        # Merge props with predictions
        merged = props_df.merge(
            predictions_df[['player_name', 'market', 'line', 'predicted_prob', 'recommendation']],
            on=['player_name', 'market', 'line'],
            how='inner'
        )
        
        if merged.empty:
            return pd.DataFrame()
        
        value_bets = []
        
        for idx, row in merged.iterrows():
            # Get odds for recommended side
            if row['recommendation'] == 'OVER':
                odds = row['odds_over']
                model_prob = row['predicted_prob']
            elif row['recommendation'] == 'UNDER':
                odds = row['odds_under']
                model_prob = 1 - row['predicted_prob']
            else:
                continue
            
            # Calculate EV
            ev = self.calculate_expected_value(odds, model_prob)
            
            # Market's implied probability
            market_prob = self.american_to_implied_prob(odds) * 100
            
            # Edge (difference between your prob and market prob)
            edge = model_prob * 100 - market_prob
            
            # Only include if positive EV
            if ev > 0:
                value_bets.append({
                    'player_name': row['player_name'],
                    'market': row['market'],
                    'line': row['line'],
                    'recommendation': row['recommendation'],
                    'bookmaker': row['bookmaker'],
                    'odds': odds,
                    'model_prob': model_prob * 100,
                    'market_prob': market_prob,
                    'edge': edge,
                    'expected_value': ev,
                    'ev_pct': (ev / 100) * 100  # EV as percentage
                })
        
        value_df = pd.DataFrame(value_bets)
        
        if not value_df.empty:
            value_df = value_df.sort_values('expected_value', ascending=False)
        
        return value_df
    
    def print_arbitrage_report(self, best_lines_df):
        """Print formatted arbitrage opportunities."""
        arb_opps = best_lines_df[best_lines_df['has_arbitrage']]
        
        if arb_opps.empty:
            print("\n❌ No arbitrage opportunities found")
            return
        
        print(f"\n{'='*70}")
        print(f"💰 ARBITRAGE OPPORTUNITIES FOUND: {len(arb_opps)}")
        print(f"{'='*70}")
        
        for idx, row in arb_opps.iterrows():
            print(f"\n{row['player_name']} - {row['market']}")
            print(f"Line: {row['line']}")
            print(f"Profit: {row['arb_profit_pct']:.2f}%")
            print(f"Best Over:  {row['best_over_book']:12s} {row['best_over_odds']:+4.0f}")
            print(f"Best Under: {row['best_under_book']:12s} {row['best_under_odds']:+4.0f}")
            print(f"─" * 50)
    
    def print_best_lines_report(self, best_lines_df, top_n=20):
        """Print best line shopping opportunities."""
        # Filter for significant value (> 5 units difference)
        value_props = best_lines_df[
            (best_lines_df['over_value'].abs() > 5) |
            (best_lines_df['under_value'].abs() > 5)
        ].head(top_n)
        
        if value_props.empty:
            print("\n⚠️  No significant line value differences found")
            return
        
        print(f"\n{'='*70}")
        print(f"📊 BEST LINE SHOPPING OPPORTUNITIES (Top {top_n})")
        print(f"{'='*70}")
        
        for idx, row in value_props.iterrows():
            print(f"\n{row['player_name']} - {row['market']} {row['line']}")
            
            if abs(row['over_value']) > 5:
                print(f"  OVER:  {row['best_over_book']:12s} {row['best_over_odds']:+4.0f} "
                      f"(+{row['over_value']:.0f} vs avg)")
            
            if abs(row['under_value']) > 5:
                print(f"  UNDER: {row['best_under_book']:12s} {row['best_under_odds']:+4.0f} "
                      f"(+{row['under_value']:.0f} vs avg)")