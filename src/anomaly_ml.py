import pandas as pd
import numpy as np
import os
import json
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, auc, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

COUNTRY = 'DE'
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'outputs')

def train_anomaly_classifier():
    print("Anomaly Detection Training ")

    labels_path = os.path.join(OUTPUT_DIR, 'anomaly_labels_to_verify.csv')
    try:
        df_labels = pd.read_csv(labels_path, parse_dates=['timestamp'])
    except FileNotFoundError:
        print("Error: Labels file not found. Please complete Phase 1 labeling.")
        return

    
    
    data_path = os.path.join(DATA_DIR, f'{COUNTRY}.csv')
    df_features = pd.read_csv(data_path, parse_dates=['timestamp'], index_col='timestamp')

    df_features['hour'] = df_features.index.hour
    df_features['dayofweek'] = df_features.index.dayofweek

    df_features['lag_1'] = df_features['load'].shift(1)
    df_features['lag_24'] = df_features['load'].shift(24)
    df_features['lag_168'] = df_features['load'].shift(168)

    df_features['roll_mean_24'] = df_features['load'].rolling(24).mean()
    df_features['roll_std_24'] = df_features['load'].rolling(24).std()
    df_features['roll_mean_168'] = df_features['load'].rolling(168).mean()
    df_features['roll_std_168'] = df_features['load'].rolling(168).std()

    df_train = pd.merge(df_labels[['timestamp', 'label', 'error', 'z_score']], df_features, on='timestamp', how='inner')

    df_train = df_train.dropna()

    print(f"Training on {len(df_train)} labeled examples.")

    feature_cols = ['hour', 'dayofweek', 'lag_1', 'lag_24', 'lag_168', 
                    'roll_mean_24', 'roll_std_24', 'roll_mean_168', 'roll_std_168',
                    'error', 'z_score']
    X = df_train[feature_cols]
    y = df_train['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    model = LogisticRegression(class_weight='balanced', max_iter=1000)
    model.fit(X_train, y_train)

    y_probs = model.predict_proba(X_test)[:, 1]

    precision, recall, thresholds = precision_recall_curve(y_test, y_probs)
    pr_auc = auc(recall, precision)

    target_precision = 0.80
    valid_indices = np.where(precision >= target_precision)[0]

    if len(valid_indices) > 0:
        idx = valid_indices[0]
        if idx < len(thresholds):
            chosen_threshold = thresholds[idx]
        else:
            chosen_threshold = thresholds[-1]
        
        y_pred_custom = (y_probs >= chosen_threshold).astype(int)
        f1_at_p80 = f1_score(y_test, y_pred_custom)
        actual_precision = precision_score(y_test, y_pred_custom)
    else:
        print("Warning: Could not reach 0.80 precision.")
        f1_at_p80 = 0.0
        actual_precision = 0.0

    results = {
        "model": "LogisticRegression",
        "pr_auc": round(pr_auc, 4),
        "f1_at_p80": round(f1_at_p80, 4),
        "actual_precision": round(actual_precision, 4),
        "features": feature_cols
    }

    output_json_path = os.path.join(OUTPUT_DIR, 'anomaly_ml_eval.json')
    with open(output_json_path, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"Model trained. PR-AUC: {pr_auc:.4f}")
    print(f"Results saved to {output_json_path}")

    importance = list(zip(feature_cols, model.coef_[0]))
    importance.sort(key=lambda x: abs(x[1]), reverse=True)
    print("\nTop Feature Importance:")
    for feat, coef in importance:
        print(f"{feat}: {coef:.4f}")

if __name__ == "__main__":
    train_anomaly_classifier()