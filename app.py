import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.predict import predict_game
from nba_api.stats.static import teams

# Load data
@st.cache_data
def load_data():
    games_df = pd.read_csv('data/data/games.csv')
    processed_df = pd.read_csv('data/processed/processed_games.csv')
    return games_df, processed_df

games_df, processed_df = load_data()

# Get team list
all_teams = teams.get_teams()
team_options = [team['abbreviation'] for team in all_teams]

st.title("NBA Game Predictor")

st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Team Statistics", "Predict Game"])

if page == "Home":
    st.header("Welcome to NBA Predictor")
    st.write("This app uses historical NBA data to predict game outcomes and explore team statistics.")
    
    # Show recent games
    st.subheader("Recent Games")
    recent_games = games_df.sort_values('GAME_DATE', ascending=False).head(10)
    st.dataframe(recent_games[['GAME_DATE', 'TEAM_ABBREVIATION', 'MATCHUP', 'WL', 'PTS']])

elif page == "Team Statistics":
    st.header("Team Statistics")
    selected_team = st.selectbox("Select Team", team_options)
    
    team_games = processed_df[processed_df['TEAM_ABBREVIATION'] == selected_team]
    if not team_games.empty:
        st.subheader(f"{selected_team} Recent Performance")
        fig, ax = plt.subplots()
        team_games_sorted = team_games.sort_values('GAME_DATE')
        ax.plot(team_games_sorted['GAME_DATE'], team_games_sorted['WIN_PCT'], label='Win %')
        ax.plot(team_games_sorted['GAME_DATE'], team_games_sorted['AVG_PTS'], label='Avg Pts')
        ax.legend()
        st.pyplot(fig)
        
        st.subheader("Shooting Stats")
        st.write(f"Average FG%: {team_games['FG_PCT'].mean():.3f}")
        st.write(f"Average 3P%: {team_games['FG3_PCT'].mean():.3f}")
    else:
        st.write("No data available for this team.")

elif page == "Predict Game":
    st.header("Predict Game Outcome")
    col1, col2 = st.columns(2)
    with col1:
        home_team = st.selectbox("Home Team", team_options)
    with col2:
        away_team = st.selectbox("Away Team", team_options)
    
    if st.button("Predict"):
        if home_team == away_team:
            st.error("Home and away teams must be different.")
        else:
            result = predict_game(home_team, away_team)
            st.success(result)
