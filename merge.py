"""
Merge Multiple Historical Odds Files
Combines all historical_odds_*.csv files into one master file
"""

import pandas as pd
from pathlib import Path
import glob
def merge_odds_files(data_dir="data"):
    """
    Merge all historical_odds_*.csv files in data directory.

    Args:
        data_dir: Directory containing CSV files

    Returns:
        Combined DataFrame
    """

    data_path = Path(data_dir)

    # Find all historical odds files
    # Ignore already combined files
    pattern = str(data_path / "historical_odds_*.csv")

    files = [
        f for f in glob.glob(pattern)
        if "combined" not in Path(f).name
    ]

    if not files:
        print(f"❌ No historical_odds_*.csv files found in {data_dir}/")
        return None

    print(f"\n{'='*70}")
    print(f"MERGING HISTORICAL ODDS FILES")
    print(f"{'='*70}")
    print(f"Found {len(files)} files:\n")

    for f in files:
        size_mb = Path(f).stat().st_size / (1024 * 1024)
        print(f"  {Path(f).name:50s} {size_mb:6.1f} MB")

    print(f"\n{'='*70}")
    print("Loading and combining files...")

    # Load all files
    dfs = []
    total_rows = 0

    for file in files:
        print(f"\n  Loading {Path(file).name}...")

        try:
            df = pd.read_csv(file)

            rows = len(df)
            total_rows += rows

            print(f"    Rows: {rows:,}")

            dfs.append(df)

        except Exception as e:
            print(f"    ❌ Failed to load: {e}")

    if not dfs:
        print("❌ No valid files loaded")
        return None

    # Combine all DataFrames
    print(f"\nCombining all files...")

    combined = pd.concat(dfs, ignore_index=True)

    print(f"  Total rows before dedup: {len(combined):,}")

    # Parse dates safely BEFORE dedup
    print("\nParsing dates...")

    combined['game_date'] = pd.to_datetime(
        combined['game_date'],
        format='mixed',
        utc=True,
        errors='coerce'
    )

    # Remove bad dates
    bad_dates = combined['game_date'].isna().sum()

    print(f"  Bad/unparseable dates: {bad_dates:,}")

    combined = combined.dropna(subset=['game_date'])

    # Remove duplicates
    combined = combined.drop_duplicates(
        subset=[
            'game_date',
            'player_name',
            'market',
            'line',
            'bet_type',
            'bookmaker'
        ],
        keep='first'
    )

    print(f"  Total rows after dedup:  {len(combined):,}")
    print(f"  Duplicates removed:      {total_rows - len(combined):,}")

    # Sort by date
    combined = combined.sort_values('game_date')

    if len(combined) == 0:
        print("❌ No rows remaining after cleaning")
        return None

    # Get date range
    min_date = combined['game_date'].min()
    max_date = combined['game_date'].max()

    print(f"\n{'='*70}")
    print(f"COMBINED DATASET SUMMARY")
    print(f"{'='*70}")

    print(
        f"Date Range:    "
        f"{min_date.strftime('%Y-%m-%d')} "
        f"to "
        f"{max_date.strftime('%Y-%m-%d')}"
    )

    print(f"Total Props:    {len(combined):,}")

    if 'player_name' in combined.columns:
        print(f"Unique Players: {combined['player_name'].nunique():,}")

    print(f"Unique Games:   {combined['game_date'].nunique():,}")

    # Breakdown by market
    if 'market' in combined.columns:

        print(f"\nProps by Market:")

        for market in combined['market'].value_counts().index:
            count = len(combined[combined['market'] == market])

            print(f"  {market:25s}: {count:6,}")

    # Save combined file
    output_file = data_path / (
        f"historical_odds_combined_"
        f"{min_date.strftime('%Y-%m-%d')}_"
        f"{max_date.strftime('%Y-%m-%d')}.csv"
    )

    print(f"\n{'='*70}")
    print(f"Saving combined file...")

    combined.to_csv(output_file, index=False)

    size_mb = output_file.stat().st_size / (1024 * 1024)

    print(f"✓ Saved: {output_file.name}")
    print(f"  Size: {size_mb:.1f} MB")

    print(f"{'='*70}\n")

    return combined, output_file


def merge_and_prepare_training_data(start_date=None, end_date=None):
    """
    Complete workflow:
    Merge odds + Prepare training data.
    """

    # Step 1: Merge odds files
    result = merge_odds_files()

    if result is None:
        return

    combined, output_file = result

    # Step 2: Optional filtering
    if start_date or end_date:

        print(f"\nFiltering by date...")

        if start_date:
            start_date = pd.to_datetime(start_date, utc=True)

            combined = combined[
                combined['game_date'] >= start_date
            ]

            print(
                f"  After start_date filter: "
                f"{len(combined):,} props"
            )

        if end_date:
            end_date = pd.to_datetime(end_date, utc=True)

            combined = combined[
                combined['game_date'] <= end_date
            ]

            print(
                f"  After end_date filter: "
                f"{len(combined):,} props"
            )

    if len(combined) == 0:
        print("❌ No data remaining after filtering")
        return

    # Step 3: Next step instructions
    print(f"\n{'='*70}")
    print("NEXT STEP: PREPARE TRAINING DATA")
    print(f"{'='*70}")

    min_date = combined['game_date'].min().strftime('%Y-%m-%d')
    max_date = combined['game_date'].max().strftime('%Y-%m-%d')

    print("Now run:")
    print(f"  python main.py --mode prepare_data \\")
    print(f"    --start {min_date} \\")
    print(f"    --end {max_date}")

    print(f"\nThis will:")
    print(f"  1. Load: {output_file.name}")
    print(f"  2. Fetch actual NBA results")
    print(f"  3. Build {len(combined):,} training examples")
    print(f"  4. Takes ~60-90 minutes")

    print(f"{'='*70}\n")


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description='Merge historical odds files'
    )

    parser.add_argument(
        '--merge-only',
        action='store_true',
        help='Only merge files'
    )

    parser.add_argument(
        '--start',
        help='Start date filter (YYYY-MM-DD)'
    )

    parser.add_argument(
        '--end',
        help='End date filter (YYYY-MM-DD)'
    )

    parser.add_argument(
        '--data-dir',
        default='data',
        help='Directory containing odds files'
    )

    args = parser.parse_args()

    if args.merge_only:
        merge_odds_files(args.data_dir)

    else:
        merge_and_prepare_training_data(
            args.start,
            args.end
        )





















