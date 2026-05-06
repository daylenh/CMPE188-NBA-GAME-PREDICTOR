import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

def load_features():
    """
    Load engineered features.
    """
    return pd.read_csv('data/processed/features.csv')

def prepare_data(df):
    """
    Prepare data for training: select features, split.
    """
    # Feature columns
    feature_cols = [col for col in df.columns if col.startswith(('HOME_', 'AWAY_')) and col != 'HOME_TEAM' and col != 'AWAY_TEAM']
    
    X = df[feature_cols]
    y = df['TARGET']
    
    # Split into train and test (time-based split for realism, but for simplicity random)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test, feature_cols

def train_model(X_train, y_train):
    """
    Train a RandomForest classifier.
    """
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model.
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    print(f"Accuracy: {accuracy}")
    print(report)
    return accuracy, report

def save_model(model, feature_cols):
    """
    Save the trained model and feature list.
    """
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/nba_predictor.pkl')
    joblib.dump(feature_cols, 'models/feature_cols.pkl')

if __name__ == "__main__":
    df = load_features()
    X_train, X_test, y_train, y_test, feature_cols = prepare_data(df)
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    save_model(model, feature_cols)
    print("Model training complete.")
