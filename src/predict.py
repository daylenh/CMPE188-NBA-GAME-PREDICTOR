import pandas as pd
import joblib
from nba_api.stats.static import teams

def load_model_and_features():
    """
    Load trained model and feature columns.
    """
    model = joblib.load('models/nba_predictor.pkl')
    feature_cols = joblib.load('models/feature_cols.pkl')
    return model, feature_cols

def get_team_features(team_abbr, processed_df):
    """
    Get current features for a team from processed data.
    Calculate rolling statistics on the fly.
    """
    team_games = processed_df[processed_df['TEAM_ABBREVIATION'] == team_abbr].copy()
    if team_games.empty:
        return None
    
    # Sort by date
    team_games = team_games.sort_values('GAME_DATE')
    
    # Calculate rolling statistics
    team_games['WIN_PCT'] = (team_games['WL'] == 'W').expanding().mean()
    team_games['AVG_PTS'] = team_games['PTS'].expanding().mean()
    team_games['RECENT_WIN_PCT_5'] = (team_games['WL'] == 'W').rolling(5, min_periods=1).mean()
    
    # Home vs away performance
    home_games = team_games[team_games['MATCHUP'].str.contains('vs.')]
    away_games = team_games[team_games['MATCHUP'].str.contains('@')]
    
    if not home_games.empty:
        home_games['HOME_WIN_PCT'] = (home_games['WL'] == 'W').expanding().mean()
        team_games = team_games.merge(home_games[['GAME_ID', 'HOME_WIN_PCT']], on='GAME_ID', how='left')
    else:
        team_games['HOME_WIN_PCT'] = 0.5  # Default
    
    if not away_games.empty:
        away_games['AWAY_WIN_PCT'] = (away_games['WL'] == 'W').expanding().mean()
        team_games = team_games.merge(away_games[['GAME_ID', 'AWAY_WIN_PCT']], on='GAME_ID', how='left')
    else:
        team_games['AWAY_WIN_PCT'] = 0.5  # Default
    
    # Shooting stats
    team_games['AVG_FG_PCT'] = team_games['FG_PCT'].expanding().mean()
    team_games['AVG_FG3_PCT'] = team_games['FG3_PCT'].expanding().mean()
    team_games['AVG_FT_PCT'] = team_games['FT_PCT'].expanding().mean()
    
    # Get latest game features
    latest = team_games.iloc[-1]
    features = {
        'WIN_PCT': latest['WIN_PCT'],
        'AVG_PTS': latest['AVG_PTS'],
        'RECENT_WIN_PCT_5': latest['RECENT_WIN_PCT_5'],
        'HOME_WIN_PCT': latest.get('HOME_WIN_PCT', 0.5),
        'AWAY_WIN_PCT': latest.get('AWAY_WIN_PCT', 0.5),
        'AVG_FG_PCT': latest['AVG_FG_PCT'],
        'AVG_FG3_PCT': latest['AVG_FG3_PCT'],
        'AVG_FT_PCT': latest['AVG_FT_PCT']
    }
    
    return features

def predict_game(home_team, away_team):
    """
    Predict the outcome of a game between home_team and away_team.
    """
    model, feature_cols = load_model_and_features()
    
    # Load processed data
    processed_df = pd.read_csv('data/processed/processed_games.csv')
    
    home_features = get_team_features(home_team, processed_df)
    away_features = get_team_features(away_team, processed_df)
    
    if home_features is None or away_features is None:
        return "Data not available for one or both teams."
    
    # Create feature vector
    input_features = {}
    for col in feature_cols:
        if col.startswith('HOME_'):
            base_col = col.replace('HOME_', '')
            input_features[col] = home_features.get(base_col, 0)
        elif col.startswith('AWAY_'):
            base_col = col.replace('AWAY_', '')
            input_features[col] = away_features.get(base_col, 0)
    
    input_df = pd.DataFrame([input_features])
    
    # Predict
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0]
    
    winner = home_team if prediction == 1 else away_team
    prob_win = probability[1] if prediction == 1 else probability[0]
    
    return f"Predicted winner: {winner} with probability {prob_win:.2f}"

if __name__ == "__main__":
    # Example prediction
    result = predict_game('LAL', 'BOS')
    print(result)
