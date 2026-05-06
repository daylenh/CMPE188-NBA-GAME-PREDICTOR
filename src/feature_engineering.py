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
    df['HOME_GAME'] = df['MATCHUP'].str.contains('vs.').astype(int)
    
    # Opponent team
    df['OPPONENT'] = df['MATCHUP'].str.split().str[-1]
    
    # Target: 1 if win, 0 if loss
    df['TARGET'] = (df['WL'] == 'W').astype(int)
    
    # Group by team and compute rolling stats
    teams = df['TEAM_ABBREVIATION'].unique()
    feature_dfs = []
    
    for team in teams:
        team_df = df[df['TEAM_ABBREVIATION'] == team].copy()
        team_df = team_df.sort_values('GAME_DATE')
        
        # Cumulative games played
        team_df['GAMES_PLAYED'] = range(1, len(team_df) + 1)
        
        # Win percentage (rolling)
        team_df['WIN_PCT'] = team_df['TARGET'].expanding().mean()
        
        # Average points scored (rolling)
        team_df['AVG_PTS'] = team_df['PTS'].expanding().mean()
        
        # Average points allowed (need to get from opponent)
        # For simplicity, use PLUS_MINUS or calculate later
        
        # Recent 5 games win pct
        team_df['RECENT_WIN_PCT_5'] = team_df['TARGET'].rolling(5, min_periods=1).mean()
        
        # Home win pct
        home_games = team_df[team_df['HOME_GAME'] == 1]
        if not home_games.empty:
            home_games['HOME_WIN_PCT'] = home_games['TARGET'].expanding().mean()
            team_df = team_df.merge(home_games[['GAME_ID', 'HOME_WIN_PCT']], on='GAME_ID', how='left')
        else:
            team_df['HOME_WIN_PCT'] = np.nan
        
        # Away win pct
        away_games = team_df[team_df['HOME_GAME'] == 0]
        if not away_games.empty:
            away_games['AWAY_WIN_PCT'] = away_games['TARGET'].expanding().mean()
            team_df = team_df.merge(away_games[['GAME_ID', 'AWAY_WIN_PCT']], on='GAME_ID', how='left')
        else:
            team_df['AWAY_WIN_PCT'] = np.nan
        
        # Shooting efficiency
        team_df['AVG_FG_PCT'] = team_df['FG_PCT'].expanding().mean()
        team_df['AVG_FG3_PCT'] = team_df['FG3_PCT'].expanding().mean()
        team_df['AVG_FT_PCT'] = team_df['FT_PCT'].expanding().mean()
        
        feature_dfs.append(team_df)
    
    features_df = pd.concat(feature_dfs).sort_values('GAME_DATE').reset_index(drop=True)
    
    # Fill NaN with 0 or mean
    features_df = features_df.fillna(0)
    
    return features_df

def create_matchup_features(df):
    """
    Create features for each matchup: diff in win pct, etc.
    """
    # For each game, get features for both teams
    home_features = df[df['HOME_GAME'] == 1].copy()
    away_features = df[df['HOME_GAME'] == 0].copy()
    
    # Rename columns for home and away
    home_cols = {col: f'HOME_{col}' for col in ['WIN_PCT', 'AVG_PTS', 'RECENT_WIN_PCT_5', 'HOME_WIN_PCT', 'AWAY_WIN_PCT', 'AVG_FG_PCT', 'AVG_FG3_PCT', 'AVG_FT_PCT']}
    away_cols = {col: f'AWAY_{col}' for col in ['WIN_PCT', 'AVG_PTS', 'RECENT_WIN_PCT_5', 'HOME_WIN_PCT', 'AWAY_WIN_PCT', 'AVG_FG_PCT', 'AVG_FG3_PCT', 'AVG_FT_PCT']}
    
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
