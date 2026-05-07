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
    
    team_games = processed_df[processed_df['TEAM_ABBREVIATION'] == selected_team].copy()
    if not team_games.empty:
        # Convert GAME_DATE to datetime
        team_games['GAME_DATE'] = pd.to_datetime(team_games['GAME_DATE'])
        
        # Sort by date and calculate rolling win percentage
        team_games = team_games.sort_values('GAME_DATE')
        team_games['WIN_PCT'] = (team_games['WL'] == 'W').expanding().mean()
        
        # Option to show only recent games
        show_recent_only = st.checkbox("Show only last 3 seasons", value=True)
        
        if show_recent_only:
            # Get the most recent date and go back 3 seasons (about 2.5 years)
            max_date = team_games['GAME_DATE'].max()
            min_date = max_date - pd.DateOffset(years=3)
            team_games = team_games[team_games['GAME_DATE'] >= min_date]
        
        st.subheader(f"{selected_team} Recent Performance")
        if len(team_games) > 0:
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(team_games['GAME_DATE'], team_games['WIN_PCT'], label='Win %', marker='o', markersize=3)
            ax.plot(team_games['GAME_DATE'], team_games['PTS'], label='Points', marker='s', markersize=3)
            ax.set_xlabel('Game Date')
            ax.set_ylabel('Value')
            ax.set_title(f'{selected_team} Performance Over Time')
            ax.legend()
            
            # Improve x-axis labels based on date range
            date_range = team_games['GAME_DATE'].max() - team_games['GAME_DATE'].min()
            import matplotlib.dates as mdates
            if date_range.days > 365:  # More than a year
                ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))  # Every 3 months
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))  # YYYY-MM
            else:
                ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))  # Every month
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))  # MM/DD
            
            ax.tick_params(axis='x', rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.write("No recent games found for this team.")
        
        st.subheader("Shooting Stats")
        st.write(f"Average FG%: {team_games['FG_PCT'].mean():.3f}")
        st.write(f"Average 3P%: {team_games['FG3_PCT'].mean():.3f}")
        st.write(f"Overall Win %: {team_games['WIN_PCT'].iloc[-1]:.3f}")
        
        # Recent form
        recent_games = team_games.tail(5)
        recent_wins = (recent_games['WL'] == 'W').sum()
        st.write(f"Recent Form (Last 5 games): {recent_wins}/5 wins")
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
