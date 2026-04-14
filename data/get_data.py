from nba_api.stats.endpoints import leaguegamefinder
import pandas as pd
import os

def fetch_nba_games():
    print("Downloading NBA game data...")
    gamefinder = leaguegamefinder.LeagueGameFinder()
    games = gamefinder.get_data_frames()[0]
    return games

def save_csv(df):
    os.makedirs("data", exist_ok=True)
    path = "data/games.csv"
    df.to_csv(path, index=False)
    print(f"Saved file to {path}")

if __name__ == "__main__":
    df = fetch_nba_games()
    print("Preview:")
    print(df.head())
    save_csv(df)