import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

def load_model_and_features():
    """
    Load trained model and feature columns.
    """
    model = joblib.load('models/nba_predictor.pkl')
    feature_cols = joblib.load('models/feature_cols.pkl')
    return model, feature_cols

def feature_importance(model, feature_cols):
    """
    Compute and display feature importance.
    """
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({'feature': feature_cols, 'importance': importances})
    feature_importance_df = feature_importance_df.sort_values('importance', ascending=False)
    
    print("Feature Importances:")
    print(feature_importance_df)
    
    # Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance_df.head(10))
    plt.title('Top 10 Feature Importances')
    plt.savefig('report_assets/figures/feature_importance.png')
    plt.show()
    
    return feature_importance_df

if __name__ == "__main__":
    model, feature_cols = load_model_and_features()
    feature_importance(model, feature_cols)
    print("Evaluation complete.")
