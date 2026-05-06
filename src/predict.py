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
    """
    # For simplicity, get the latest game features for the team
    team_games = processed_df[processed_df['TEAM_ABBREVIATION'] == team_abbr].sort_values('GAME_DATE')
    if team_games.empty:
        return None
    latest = team_games.iloc[-1]
    features = {col: latest[col] for col in processed_df.columns if col.startswith(('WIN_PCT', 'AVG_PTS', 'RECENT_WIN_PCT_5', 'HOME_WIN_PCT', 'AWAY_WIN_PCT', 'AVG_FG_PCT', 'AVG_FG3_PCT', 'AVG_FT_PCT'))}
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
