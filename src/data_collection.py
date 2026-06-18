import os
import sys
import time
import pandas as pd

# Paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
RAW_DATA_PATH = os.path.join(RAW_DATA_DIR, "games.csv")

# Seasons to collect
DEFAULT_SEASONS = ["2022-23", "2023-24", "2024-25"]

def fetch_games(
    seasons: list[str] = DEFAULT_SEASONS,
    save_path: str = RAW_DATA_PATH,
    request_delay: float = 0.8,
) -> pd.DataFrame:
    """
    Pull regular-season game logs for every NBA team for each season.

    Parameters
    ----------
    seasons: List of season strings, e.g. ["2022-23", "2023-24"]
    save_path: Where to write the combined CSV
    request_delay: Seconds to sleep between API calls

    Returns
    -------
    pd.DataFrame with all raw game-log rows
    """
    try:
        from nba_api.stats.endpoints import leaguegamefinder
    except ImportError:
        print("[ERROR] nba_api is not installed.  Run: pip install nba_api")
        sys.exit(1)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    all_frames: list[pd.DataFrame] = []

    for season in seasons:
        print(f"[data_collection] Fetching season {season} ...", flush=True)
        try:
            finder = leaguegamefinder.LeagueGameFinder(
                season_nullable=season,
                season_type_nullable="Regular Season",
                league_id_nullable="00",
            )
            df = finder.get_data_frames()[0]
            df["SEASON"] = season
            all_frames.append(df)
            print(f"  -> {len(df):,} rows retrieved")
        except Exception as exc:
            print(f"  [WARNING] Could not fetch {season}: {exc}")

        time.sleep(request_delay)

    if not all_frames:
        raise RuntimeError(
            "No data was fetched. Check your internet connection and that "
            "nba_api is correctly installed."
        )

    games = pd.concat(all_frames, ignore_index=True)

    keep_cols = [
        "SEASON", "GAME_ID", "GAME_DATE", "MATCHUP",
        "TEAM_ID", "TEAM_ABBREVIATION", "TEAM_NAME",
        "WL", "PTS", "FGM", "FGA", "FG_PCT",
        "FG3M", "FG3A", "FG3_PCT", "FTM", "FTA", "FT_PCT",
        "OREB", "DREB", "REB", "AST", "STL", "BLK", "TOV",
        "PLUS_MINUS",
    ]
    existing_keep = [c for c in keep_cols if c in games.columns]
    games = games[existing_keep]

    games.to_csv(save_path, index=False)
    print(f"\n[data_collection] Saved {len(games):,} rows -> {save_path}")
    return games

# Entry point
if __name__ == "__main__":
    fetch_games()
