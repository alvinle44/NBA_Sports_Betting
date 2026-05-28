# scripts/combo_prop_predictor.py

import pandas as pd
import numpy as np
from pathlib import Path
from scripts.config import Config

class ComboPropPredictor:
    """
    Predict combo props with confidence scores and Kelly bet sizing.
    """
    
    def __init__(self):
        self.combo_markets = {
            'player_points_rebounds_assists': ['player_points', 'player_rebounds', 'player_assists'],
            'player_points_rebounds': ['player_points', 'player_rebounds'],
            'player_points_assists': ['player_points', 'player_assists'],
            'player_rebounds_assists': ['player_assists', 'player_rebounds'],
        }
        
        # Correlation factors (based on NBA research)
        self.correlation_adjustments = {
            'player_points_rebounds_assists': 0.92,  # Slight negative (pts vs ast)
            'player_points_rebounds': 0.96,          # Minimal correlation
            'player_points_assists': 0.93,            # Negative (high usage)
            'player_rebounds_assists': 0.97,          # Positive (playmakers)
        }
    
    def calculate_combo_confidence(self, component_confidences, market):
        """Calculate confidence for combo prop with correlation adjustment."""
        base_confidence = min(component_confidences.values())
        correlation_factor = self.correlation_adjustments.get(market, 0.95)
        adjusted_confidence = base_confidence * correlation_factor
        return max(0, min(100, adjusted_confidence))
    
    def calculate_win_probability(self, predicted_value, line, confidence, bet_direction):
        """Calculate probability of winning the bet."""
        edge = predicted_value - line
        base_prob = confidence / 100
        edge_factor = min(abs(edge) / 5, 0.15)
        
        if bet_direction == 'OVER':
            win_prob = base_prob + edge_factor if edge > 0 else base_prob - edge_factor
        else:
            win_prob = base_prob + edge_factor if edge < 0 else base_prob - edge_factor
        
        return max(0.45, min(0.75, win_prob))
    
    def calculate_kelly_bet_size(self, win_prob, odds, bankroll=1000):
        """Calculate Kelly Criterion bet size."""
        if odds > 0:
            decimal_odds = (odds / 100) + 1
        else:
            decimal_odds = (100 / abs(odds)) + 1
        
        b = decimal_odds - 1
        p = win_prob
        q = 1 - p
        
        kelly_fraction = (b * p - q) / b
        
        if kelly_fraction <= 0:
            return {
                'kelly_full': 0,
                'kelly_half': 0,
                'kelly_quarter': 0,
                'recommended': 0,
                'kelly_fraction': 0,
                'recommendation': '❌ NO EDGE'
            }
        
        full_kelly = kelly_fraction * bankroll
        half_kelly = full_kelly * 0.5
        quarter_kelly = full_kelly * 0.25
        max_bet = bankroll * 0.05
        recommended = min(quarter_kelly, max_bet)
        recommended = round(recommended / 5) * 5
        
        if recommended < 5 and kelly_fraction > 0:
            recommended = 5
        
        return {
            'kelly_full': round(full_kelly, 2),
            'kelly_half': round(half_kelly, 2),
            'kelly_quarter': round(quarter_kelly, 2),
            'recommended': round(recommended, 2),
            'kelly_fraction': round(kelly_fraction, 4),
            'recommendation': f'✅ BET ${recommended:.0f}' if recommended >= 5 else '❌ SKIP'
        }
    
    def calculate_expected_value(self, win_prob, odds):
        """Calculate expected value of the bet."""
        if odds > 0:
            decimal_odds = (odds / 100) + 1
        else:
            decimal_odds = (100 / abs(odds)) + 1
        
        profit = decimal_odds - 1
        loss_prob = 1 - win_prob
        ev = (win_prob * profit) - (loss_prob * 1.0)
        return ev * 100
    
    def predict_combo_props(self, individual_predictions_df, combo_props_df, bankroll=1000):
        """
        Generate predictions for combo props with full confidence and bet sizing.
        """
        combo_predictions = []
        
        print(f"\n🎯 Predicting {len(combo_props_df)} combo props...")
        
        for idx, combo_prop in combo_props_df.iterrows():
            player = combo_prop['player_name']
            market = combo_prop['market']
            line = combo_prop['line']
            odds_over = combo_prop.get('odds_over', -110)
            odds_under = combo_prop.get('odds_under', -110)
            
            component_markets = self.combo_markets.get(market)
            if not component_markets:
                continue
            
            player_preds = individual_predictions_df[
                individual_predictions_df['player_name'] == player
            ]
            
            component_predictions = {}
            component_confidences = {}
            component_details = {}
            missing_components = []
            
            for component in component_markets:
                comp_pred = player_preds[player_preds['market'] == component]
                
                if comp_pred.empty:
                    missing_components.append(component)
                else:
                    pred_row = comp_pred.iloc[0]
                    component_predictions[component] = pred_row['predicted_value']
                    component_confidences[component] = pred_row['confidence']
                    component_details[component] = {
                        'predicted': pred_row['predicted_value'],
                        'confidence': pred_row['confidence'],
                        'line': pred_row.get('line', 0)
                    }
            
            if missing_components:
                print(f"  ⚠️  {player} - {market}: Missing {missing_components}")
                continue
            
            combined_prediction = sum(component_predictions.values())
            combined_confidence = self.calculate_combo_confidence(component_confidences, market)
            edge_vs_line = combined_prediction - line
            
            if abs(edge_vs_line) < 0.3:
                bet_direction = 'SKIP'
                win_prob = 0.5
                odds_to_use = -110
            elif edge_vs_line > 0:
                bet_direction = 'OVER'
                odds_to_use = odds_over
                win_prob = self.calculate_win_probability(
                    combined_prediction, line, combined_confidence, 'OVER'
                )
            else:
                bet_direction = 'UNDER'
                odds_to_use = odds_under
                win_prob = self.calculate_win_probability(
                    combined_prediction, line, combined_confidence, 'UNDER'
                )
            
            ev_pct = self.calculate_expected_value(win_prob, odds_to_use)
            kelly_sizing = self.calculate_kelly_bet_size(win_prob, odds_to_use, bankroll)
            
            if bet_direction == 'SKIP':
                recommendation = '⚠️ TOO CLOSE TO CALL'
            elif combined_confidence < 50:
                recommendation = '❌ LOW CONFIDENCE'
            elif ev_pct < 0:
                recommendation = '❌ NEGATIVE EV'
            elif kelly_sizing['recommended'] < 5:
                recommendation = '❌ EDGE TOO SMALL'
            else:
                recommendation = f"✅ BET {bet_direction} ${kelly_sizing['recommended']:.0f}"
            
            combo_predictions.append({
                'player_name': player,
                'market': market,
                'line': line,
                'predicted_value': round(combined_prediction, 1),
                'edge_vs_line': round(edge_vs_line, 1),
                'bet_direction': bet_direction,
                'confidence': round(combined_confidence, 1),
                'win_probability': round(win_prob * 100, 1),
                'expected_value_pct': round(ev_pct, 2),
                'kelly_fraction': kelly_sizing['kelly_fraction'],
                'kelly_full': kelly_sizing['kelly_full'],
                'kelly_half': kelly_sizing['kelly_half'],
                'kelly_quarter': kelly_sizing['kelly_quarter'],
                'recommended_bet': kelly_sizing['recommended'],
                'recommendation': recommendation,
                'odds_over': odds_over,
                'odds_under': odds_under,
                'bookmaker': combo_prop.get('bookmaker', 'Unknown'),
                'points_pred': component_details.get('player_points', {}).get('predicted', 0),
                'points_conf': component_details.get('player_points', {}).get('confidence', 0),
                'rebounds_pred': component_details.get('player_rebounds', {}).get('predicted', 0),
                'rebounds_conf': component_details.get('player_rebounds', {}).get('confidence', 0),
                'assists_pred': component_details.get('player_assists', {}).get('predicted', 0),
                'assists_conf': component_details.get('player_assists', {}).get('confidence', 0),
            })
        
        df = pd.DataFrame(combo_predictions)
        print(f"  ✅ Generated {len(df)} combo predictions")
        return df


def predict_all_props_including_combos(bankroll=1000):
    """Complete workflow with combo props and bet sizing."""
    from scripts.daily_predictor import DailyPredictor
    from datetime import datetime
    
    print("="*70)
    print("GENERATING PREDICTIONS (INDIVIDUAL + COMBO PROPS)")
    print("="*70)
    print(f"Bankroll: ${bankroll:,.0f}")
    
    # Step 1: Individual predictions
    print("\n📊 Step 1: Predicting individual stats...")
    predictor = DailyPredictor()
    individual_preds = predictor.run_daily_predictions(save_to_ongoing=False)
    
    if individual_preds.empty:
        print("  ⚠️  No individual predictions generated")
        return pd.DataFrame()
    
    # Add recommendation column if missing
    if 'recommendation' not in individual_preds.columns:
        individual_preds['recommendation'] = individual_preds.apply(
            lambda row: f"✅ BET {row['bet_direction']} ${row.get('kelly_bet_size', 0):.0f}" 
            if row.get('expected_value', 0) > 0 and row.get('confidence', 0) > 55
            else '❌ SKIP',
            axis=1
        )
    
    print(f"  ✅ Generated {len(individual_preds)} individual predictions")
    
    # Step 2: Fetch combo props
    print("\n📊 Step 2: Fetching combo props from The Odds API...")

    try:
        from scripts.live_odds_scraper import LiveOddsScraper
        scraper = LiveOddsScraper()
        all_props = scraper.get_todays_props()
        
        if all_props.empty:
            print("  ⚠️  No props fetched from API")
            combo_props = pd.DataFrame()
        else:
            combo_markets = [
                'player_points_rebounds_assists',
                'player_points_rebounds',
                'player_points_assists',
                'player_rebounds_assists',  # ✅ ADDED!
            ]
            
            combo_props = all_props[all_props['market'].isin(combo_markets)]
            
            print(f"  ✅ Found {len(combo_props)} combo props")
            if not combo_props.empty:
                for market in combo_markets:
                    count = len(combo_props[combo_props['market'] == market])
                    if count > 0:
                        print(f"     {market}: {count}")
    
    except Exception as e:
        print(f"  ⚠️  Error fetching combo props: {e}")
        import traceback
        traceback.print_exc()
        combo_props = pd.DataFrame()
    
    # Step 3: Generate combo predictions
    if not combo_props.empty:
        print("\n📊 Step 3: Generating combo predictions...")
        combo_predictor = ComboPropPredictor()
        combo_preds = combo_predictor.predict_combo_props(
            individual_preds, 
            combo_props,
            bankroll=bankroll
        )
    else:
        print("\n  ⚠️  No combo props available, skipping combo predictions")
        combo_preds = pd.DataFrame()
    
    # Step 4: Combine all predictions
    print("\n📊 Step 4: Combining predictions...")
    
    if not combo_preds.empty:
        all_predictions = pd.concat([individual_preds, combo_preds], ignore_index=True)
    else:
        all_predictions = individual_preds
    
    # ✅ Ensure predictions directory exists
    predictions_dir = Config.DATA_DIR / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 5: Save
    date_str = datetime.now().strftime('%Y%m%d')
    output_file = predictions_dir / f"predictions_all_{date_str}.csv"
    all_predictions.to_csv(output_file, index=False)
    
    print(f"\n✅ Saved {len(all_predictions)} predictions to {output_file}")
    
    # Step 6: Show summary
    print("\n" + "="*70)
    print("BETTING RECOMMENDATIONS")
    print("="*70)
    
    # ✅ FIX: Check if recommendation column exists
    if 'recommendation' in all_predictions.columns:
        best_bets = all_predictions[
            all_predictions['recommendation'].str.contains('BET', na=False, regex=False)
        ].sort_values('expected_value', ascending=False)
    else:
        # Fallback: filter by expected_value
        best_bets = all_predictions[
            (all_predictions.get('expected_value', 0) > 0) &
            (all_predictions.get('confidence', 0) > 55)
        ].sort_values('expected_value', ascending=False)
    
    if not best_bets.empty:
        print(f"\n🎯 TOP {min(15, len(best_bets))} BETTING OPPORTUNITIES:\n")
        
        # Select columns that exist
        display_cols = []
        for col in ['player_name', 'market', 'line', 'predicted_value', 
                   'bet_direction', 'confidence', 'expected_value', 
                   'kelly_bet_size', 'recommendation']:
            if col in best_bets.columns:
                display_cols.append(col)
        
        print(best_bets[display_cols].head(15).to_string(index=False))
        
        if 'kelly_bet_size' in best_bets.columns:
            total_recommended = best_bets['kelly_bet_size'].sum()
            print(f"\n💰 Total recommended wagers: ${total_recommended:.0f}")
            print(f"   (as % of bankroll: {total_recommended/bankroll*100:.1f}%)")
    else:
        print("\n⚠️  No strong betting opportunities found today")
    
    # Breakdown by market
    print("\n📊 BREAKDOWN BY MARKET:")
    if 'recommendation' in all_predictions.columns:
        market_summary = all_predictions.groupby('market').agg({
            'recommendation': lambda x: (x.str.contains('BET', na=False, regex=False)).sum(),
            'expected_value': 'mean',
            'confidence': 'mean'
        }).round(2)
        market_summary.columns = ['Bets Recommended', 'Avg EV%', 'Avg Confidence']
    else:
        market_summary = all_predictions.groupby('market').agg({
            'expected_value': 'mean',
            'confidence': 'mean'
        }).round(2)
        market_summary.columns = ['Avg EV%', 'Avg Confidence']
    
    print(market_summary)
    
    return all_predictions