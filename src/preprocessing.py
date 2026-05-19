import pandas as pd
import os

def load_raw_data():
    """
    Load raw data files.
    """
    games_path = 'data/games.csv'
    team_logs_path = 'data/raw/team_game_logs.csv'
    
    if os.path.exists(games_path):
        games_df = pd.read_csv(games_path)
    else:
        games_df = pd.DataFrame()
    
    if os.path.exists(team_logs_path):
        team_logs_df = pd.read_csv(team_logs_path)
    else:
        team_logs_df = pd.DataFrame()
    
    return games_df, team_logs_df

def preprocess_games_data(games_df):
    """
    Preprocess games data: handle missing values, convert types, etc.
    """
    # Drop duplicates
    games_df = games_df.drop_duplicates()
    
    # Convert GAME_DATE to datetime
    games_df['GAME_DATE'] = pd.to_datetime(games_df['GAME_DATE'])
    
    # Fill missing WL with 'N/A' or something, but for now assume complete
    games_df['WL'] = games_df['WL'].fillna('N/A')
    
    # Ensure numeric columns are float
    numeric_cols = ['MIN', 'PTS', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 
                    'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PLUS_MINUS']
    for col in numeric_cols:
        if col in games_df.columns:
            games_df[col] = pd.to_numeric(games_df[col], errors='coerce')
    
    return games_df

def merge_datasets(games_df, team_logs_df):
    """
    Merge games and team logs if needed. For now, games_df is sufficient.
    """
    # If team_logs_df is available, merge on GAME_ID and TEAM_ID
    if not team_logs_df.empty:
        merged_df = pd.merge(games_df, team_logs_df, on=['GAME_ID', 'TEAM_ID'], how='left')
    else:
        merged_df = games_df
    
    return merged_df

def save_processed_data(df):
    """
    Save processed data to data/processed/
    """
    os.makedirs('data/processed', exist_ok=True)
    df.to_csv('data/processed/processed_games.csv', index=False)

if __name__ == "__main__":
    games_df, team_logs_df = load_raw_data()
    games_df = preprocess_games_data(games_df)
    merged_df = merge_datasets(games_df, team_logs_df)
    save_processed_data(merged_df)
    print("Preprocessing complete.")
