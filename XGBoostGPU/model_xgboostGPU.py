import sys
import os
import joblib
import time
from xgboost import XGBClassifier

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = script_dir if 'processed_data' in os.listdir(script_dir) else os.path.dirname(script_dir)
sys.path.append(project_root)

from visualization_utils import evaluate_and_plot

def run():
    print("Loading data...")
    data_dir = os.path.join(project_root, 'processed_data')
    
    try:
        X_train = joblib.load(os.path.join(data_dir, 'X_train.pkl'))
        X_test = joblib.load(os.path.join(data_dir, 'X_test.pkl'))
        y_train = joblib.load(os.path.join(data_dir, 'y_train.pkl'))
        y_test = joblib.load(os.path.join(data_dir, 'y_test.pkl'))
        label_mapping = joblib.load(os.path.join(data_dir, 'label_mapping.pkl'))
    except FileNotFoundError:
        print("Fehler: Daten nicht gefunden.")
        return

    # Gewichtung für Imbalance berechnen (Benign / Attack)
    # ca. 7.3 Mio / 576k = ~12.7
    neg_counts = (y_train == 0).sum()
    pos_counts = (y_train == 1).sum()
    scale_pos_weight = neg_counts / pos_counts

    print("Training XGBoost...")
    # scale_pos_weight behandelt die 13:1 Imbalance
    clf = XGBClassifier(
        n_estimators=100, 
        learning_rate=0.1, 
        scale_pos_weight=scale_pos_weight, 
        random_state=42, 
        n_jobs=-1,
        eval_metric='logloss',
        device="cuda" # GPU als device wählen
    )
    
    start_time = time.time()
    clf.fit(X_train, y_train)
    print(f"Training Time: {time.time() - start_time:.2f} seconds")

    evaluate_and_plot(clf, X_test, y_test, "XGBoostGPU", label_mapping)

if __name__ == "__main__":
    run()