import pandas as pd
from nba_api.stats.endpoints import leaguegamefinder, teamgamelogs
from nba_api.stats.static import teams
import os

def collect_nba_data(seasons=['2022-23', '2023-24', '2024-25']):
    """
    Collect NBA game data for specified seasons.
    """
    all_games = []
    for season in seasons:
        print(f"Collecting data for season {season}")
        gamefinder = leaguegamefinder.LeagueGameFinder(season_nullable=season)
        games = gamefinder.get_data_frames()[0]
        all_games.append(games)
    
    df = pd.concat(all_games, ignore_index=True)
    return df

def collect_team_stats(seasons=['2022-23', '2023-24', '2024-25']):
    """
    Collect team game logs for specified seasons.
    """
    all_teams = teams.get_teams()
    team_ids = [team['id'] for team in all_teams]
    
    all_logs = []
    for team_id in team_ids:
        for season in seasons:
            try:
                logs = teamgamelogs.TeamGameLogs(team_id_nullable=team_id, season_nullable=season)
                df = logs.get_data_frames()[0]
                all_logs.append(df)
                print(f"Collected logs for team {team_id} in {season}")
            except Exception as e:
                print(f"Error collecting for team {team_id} in {season}: {e}")
    
    df = pd.concat(all_logs, ignore_index=True)
    return df

if __name__ == "__main__":
    # Create directories if not exist
    os.makedirs('data/raw', exist_ok=True)
    
    # Collect game data
    games_df = collect_nba_data()
    games_df.to_csv('data/raw/games.csv', index=False)
    
    # Collect team stats
    team_stats_df = collect_team_stats()
    team_stats_df.to_csv('data/raw/team_game_logs.csv', index=False)
    
    print("Data collection complete.")
