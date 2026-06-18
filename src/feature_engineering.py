import pandas as pd
import numpy as np
from datetime import timedelta

def load_processed_data():
    """
    Load processed games data.
    """
    return pd.read_csv('data/processed/processed_games.csv')

def engineer_features(df):
    """
    Engineer features for prediction: win %, avg pts, recent form, home/away.

    """
    # Sort by date
    df = df.sort_values('GAME_DATE').reset_index(drop=True)

    # Determine home/away
    df['HOME_GAME'] = df['MATCHUP'].str.contains('vs.', regex=False).astype(int)

    # Opponent team
    df['OPPONENT'] = df['MATCHUP'].str.split().str[-1]

    # Target: 1 if win, 0 if loss
    df['TARGET'] = (df['WL'] == 'W').astype(int)

    # ADDED: Points allowed
    if 'PLUS_MINUS' in df.columns:
        df['OPP_PTS'] = df['PTS'] - df['PLUS_MINUS']
    else:
        df['OPP_PTS'] = np.nan

    # Group by team and compute rolling stats
    teams = df['TEAM_ABBREVIATION'].unique()
    feature_dfs = []

    for team in teams:
        team_df = df[df['TEAM_ABBREVIATION'] == team].copy()
        team_df = team_df.sort_values('GAME_DATE').reset_index(drop=True)

        # Number of games played BEFORE this one
        team_df['GAMES_PLAYED'] = range(0, len(team_df))

        # Rolling stats (shift(1) before expanding/rolling)
        team_df['WIN_PCT'] = team_df['TARGET'].shift(1).expanding().mean()
        team_df['AVG_PTS'] = team_df['PTS'].shift(1).expanding().mean()
        team_df['AVG_OPP_PTS'] = team_df['OPP_PTS'].shift(1).expanding().mean()
        team_df['RECENT_WIN_PCT_5'] = team_df['TARGET'].shift(1).rolling(5, min_periods=1).mean()

        team_df['AVG_FG_PCT'] = team_df['FG_PCT'].shift(1).expanding().mean()
        team_df['AVG_FG3_PCT'] = team_df['FG3_PCT'].shift(1).expanding().mean()
        team_df['AVG_FT_PCT'] = team_df['FT_PCT'].shift(1).expanding().mean()

        # Home / Away split win pct
        home_mask = team_df['HOME_GAME'] == 1
        away_mask = team_df['HOME_GAME'] == 0

        team_df.loc[home_mask, 'HOME_WIN_PCT'] = (
            team_df.loc[home_mask, 'TARGET'].shift(1).expanding().mean()
        )
        team_df.loc[away_mask, 'AWAY_WIN_PCT'] = (
            team_df.loc[away_mask, 'TARGET'].shift(1).expanding().mean()
        )

        team_df['HOME_WIN_PCT'] = team_df['HOME_WIN_PCT'].ffill()
        team_df['AWAY_WIN_PCT'] = team_df['AWAY_WIN_PCT'].ffill()

        feature_dfs.append(team_df)

    features_df = pd.concat(feature_dfs).sort_values('GAME_DATE').reset_index(drop=True)

    # FIXED: fill remaining NaNs
    pct_cols = ['WIN_PCT', 'RECENT_WIN_PCT_5', 'HOME_WIN_PCT', 'AWAY_WIN_PCT',
                'AVG_FG_PCT', 'AVG_FG3_PCT', 'AVG_FT_PCT']
    for col in pct_cols:
        features_df[col] = features_df[col].fillna(0.5)

    for col in ['AVG_PTS', 'AVG_OPP_PTS']:
        features_df[col] = features_df[col].fillna(features_df[col].mean())

    return features_df

def create_matchup_features(df):
    """
    Create features for each matchup: diff in win pct, etc.
    """
    feature_base_cols = [
        'WIN_PCT', 'AVG_PTS', 'AVG_OPP_PTS', 'RECENT_WIN_PCT_5',
        'HOME_WIN_PCT', 'AWAY_WIN_PCT', 'AVG_FG_PCT', 'AVG_FG3_PCT', 'AVG_FT_PCT'
    ]

    # For each game, get features for both teams
    home_features = df[df['HOME_GAME'] == 1].copy()
    away_features = df[df['HOME_GAME'] == 0].copy()

    # Rename columns for home and away
    home_cols = {col: f'HOME_{col}' for col in feature_base_cols}
    away_cols = {col: f'AWAY_{col}' for col in feature_base_cols}

    home_features = home_features.rename(columns=home_cols)
    away_features = away_features.rename(columns=away_cols)

    # Merge on GAME_ID
    matchups = pd.merge(home_features[['GAME_ID', 'TEAM_ABBREVIATION'] + list(home_cols.values()) + ['TARGET']],
                        away_features[['GAME_ID', 'OPPONENT'] + list(away_cols.values())],
                        on='GAME_ID')

    # Rename for clarity
    matchups = matchups.rename(columns={'TEAM_ABBREVIATION': 'HOME_TEAM', 'OPPONENT': 'AWAY_TEAM'})

    # Compute differences
    matchups['WIN_PCT_DIFF'] = matchups['HOME_WIN_PCT'] - matchups['AWAY_WIN_PCT']
    matchups['AVG_PTS_DIFF'] = matchups['HOME_AVG_PTS'] - matchups['AWAY_AVG_PTS']

    return matchups

def save_features(df):
    """
    Save engineered features.
    """
    df.to_csv('data/processed/features.csv', index=False)

if __name__ == "__main__":
    df = load_processed_data()
    features_df = engineer_features(df)
    matchups_df = create_matchup_features(features_df)
    save_features(matchups_df)
    print("Feature engineering complete.")
