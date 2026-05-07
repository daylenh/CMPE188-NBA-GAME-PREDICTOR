# NBA Predictor - Model Training Guide

## How to Run train_model.py

### Quick Start
```bash
cd /Users/day_lenh/VSCODE/CMPE188/NBA_Predictor
python src/train_model.py
```

### Prerequisites
Before running this script, ensure you have:
1. **Python virtual environment activated**
   ```bash
   source venv/bin/activate
   ```

2. **Required data files in place**:
   - `data/processed/features.csv` - The engineered features (created by `feature_engineering.py`)
   
3. **Required libraries installed**:
   - pandas
   - scikit-learn
   - joblib

---

## What Does This Program Do?

The `train_model.py` script trains a machine learning model to predict NBA game winners. Here's the step-by-step process:

### Step 1: Load Features
```python
def load_features():
    return pd.read_csv('data/processed/features.csv')
```
- Reads the engineered features CSV file containing game matchup data
- Features include: win percentages, recent form, shooting stats, home/away performance

### Step 2: Prepare Data
```python
def prepare_data(df):
    # Select feature columns (those starting with HOME_ or AWAY_)
    X = df[feature_cols]                    # Input features
    y = df['TARGET']                        # Target (1=home team wins, 0=away team wins)
    
    # Split into training (80%) and testing (20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
- Extracts 14 feature columns and the target variable
- Splits data: 80% for training, 20% for testing
- Uses `random_state=42` for reproducibility

### Step 3: Train Model
```python
def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
```
- Creates a Random Forest classifier with 100 decision trees
- Trains on the training data to learn patterns

### Step 4: Evaluate Model
```python
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
```
- Tests the model on unseen test data
- Calculates accuracy and detailed classification metrics

### Step 5: Save Model
```python
def save_model(model, feature_cols):
    joblib.dump(model, 'models/nba_predictor.pkl')
    joblib.dump(feature_cols, 'models/feature_cols.pkl')
```
- Saves the trained model as a pickle file
- Saves feature column names for later use in predictions

---

## Understanding the Output

When you run the script, you'll see output like this:

```
Accuracy: 0.7679226408802934
              precision    recall  f1-score   support

           0       0.78      0.69      0.73      1366
           1       0.76      0.83      0.80      1633

    accuracy                           0.77      2999
   macro avg       0.77      0.76      0.76      2999
weighted avg       0.77      0.77      0.77      2999

Model training complete.
```

### Breaking Down the Metrics:

**Accuracy: 0.7679 (76.79%)**
- The model correctly predicts the winner 77% of the time
- Out of 2,999 test games, it got about 2,301 predictions right

**Classification Report Breakdown:**

| Metric | Class 0 (Away Win) | Class 1 (Home Win) |
|--------|-------------------|------------------|
| **Precision** | 0.78 | 0.76 |
| **Recall** | 0.69 | 0.83 |
| **F1-Score** | 0.73 | 0.80 |
| **Support** | 1,366 games | 1,633 games |

**Understanding Each Metric:**

1. **Precision (0.78 for Away, 0.76 for Home)**
   - Of all games predicted as Away/Home wins, how many were correct?
   - 78% of predicted away wins were actually correct
   - 76% of predicted home wins were actually correct

2. **Recall (0.69 for Away, 0.83 for Home)**
   - Of all actual Away/Home wins, how many did we catch?
   - Model catches 69% of actual away wins
   - Model catches 83% of actual home wins
   - **Home team advantage!** Model better at predicting home wins

3. **F1-Score (average of Precision and Recall)**
   - Home wins: 0.80 (good prediction)
   - Away wins: 0.73 (decent prediction)

4. **Support (number of test samples)**
   - 1,366 away-team wins in test data
   - 1,633 home-team wins in test data
   - Total test set: 2,999 games

---

## Output Files Generated

After running this script, two files are created:

### 1. `models/nba_predictor.pkl`
- The trained Random Forest model
- Binary file (~1-5 MB typically)
- Used by `predict.py` to make predictions

### 2. `models/feature_cols.pkl`
- List of 14 feature column names
- Used to ensure new prediction data has the same features in the same order

---

## Full Execution Flow

```
┌─────────────────────────────────┐
│ Load features.csv               │
│ (Engineered data)               │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│ Prepare Data                    │
│ - Select HOME_* & AWAY_* cols   │
│ - Split 80/20                   │
│ - 2,399 training + 600 testing  │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│ Train RandomForest              │
│ - 100 decision trees            │
│ - Learn patterns from data      │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│ Evaluate on Test Set            │
│ - Make predictions              │
│ - Calculate accuracy: 76.79%    │
│ - Generate metrics report       │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│ Save Model & Features           │
│ - nba_predictor.pkl             │
│ - feature_cols.pkl              │
└──────────────���──────────────────┘
```

---

## Common Scenarios

### Scenario 1: Want to Retrain with New Data?
```bash
# First update your data
python src/preprocessing.py
python src/feature_engineering.py

# Then retrain
python src/train_model.py
```

### Scenario 2: Want Better Accuracy?
Try tweaking the model parameters:
```python
# In train_model.py, line 33
model = RandomForestClassifier(
    n_estimators=200,          # More trees
    max_depth=15,              # Limit tree depth
    min_samples_split=5,       # More splits
    random_state=42
)
```

### Scenario 3: Want to Use the Model?
```python
from src.predict import predict_game

# Make a prediction
result = predict_game('LAL', 'BOS')
print(result)  # "Predicted winner: LAL with probability 0.64"
```

---

## Summary

| Aspect | Details |
|--------|---------|
| **Language** | Python 3 |
| **Algorithm** | Random Forest (100 trees) |
| **Data Split** | 80% train, 20% test |
| **Test Set Size** | 2,999 games |
| **Model Accuracy** | 76.79% |
| **Output Files** | 2 pickle files (.pkl) |
| **Runtime** | ~5-30 seconds |
| **Dependencies** | pandas, scikit-learn, joblib |

The model is now ready to make predictions on new, unseen NBA games!

