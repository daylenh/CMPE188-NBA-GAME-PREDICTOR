# NBA Game Outcome Prediction and Interactive Sports Analytics App

## Team Members
- Tyler Co Seng
- Dayven Lenh

---

## Problem Statement
Can historical NBA team performance data be used to predict the outcome of future games, and can we build an interactive application that allows users to explore team statistics and generate predictions?

---

## Dataset / Data Sources
We are using multiple NBA-related datasets including:
- NBA game data, schedules, and results
- Team statistics and standings
- Player statistics

Primary data sources:
- NBA.com official statistics
- nba_api Python package (https://github.com/swar/nba_api)

---

## Planned System / Model Approach
- Collect NBA game and team data using nba_api
- Clean and merge datasets into a unified format
- Engineer features such as:
  - Win percentage
  - Average points scored/allowed
  - Recent 5–10 game performance
  - Home vs away performance
  - Shooting efficiency stats
- We plan to train a machine learning model to predict game outcomes (win/loss)
- We plan to evaluate feature importance to determine key factors in winning games

---

## Current Implementation Progress
- GitHub repository created, and team members contributed via commits
- Initial project structure set up
- Beginning data collection using nba_api
- Setting up preprocessing pipeline for NBA datasets

---

## How to Run
To run the full NBA prediction pipeline, simply execute:
```
python pipeline.py
```
This single command will automatically run the entire end-to-end workflow:

- Data Collection (data/get_data.py):
Fetches raw NBA data.
- Data Preprocessing (src/preprocessing.py):
Cleans and prepares the dataset.
- Feature Engineering (src/feature_engineering.py):
Builds model-ready features.
- Model Training (src/train_model.py):
Trains the machine learning model.
- Launch Streamlit App (app.py):
Starts the interactive web interface for predictions.
