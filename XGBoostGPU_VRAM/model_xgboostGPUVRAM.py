import sys
import os
import joblib
import time
from xgboost import XGBClassifier
import cupy as cp

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = script_dir if 'processed_data' in os.listdir(script_dir) else os.path.dirname(script_dir)
sys.path.append(project_root)

from visualization_utils import evaluate_and_plot

def run():
    print("Loading data into RAM...")
    data_dir = os.path.join(project_root, 'processed_data')
    
    try:
        # Load from disk to CPU RAM
        X_train_cpu = joblib.load(os.path.join(data_dir, 'X_train.pkl'))
        X_test_cpu = joblib.load(os.path.join(data_dir, 'X_test.pkl'))
        y_train_cpu = joblib.load(os.path.join(data_dir, 'y_train.pkl'))
        y_test_cpu = joblib.load(os.path.join(data_dir, 'y_test.pkl'))
        label_mapping = joblib.load(os.path.join(data_dir, 'label_mapping.pkl'))
    except FileNotFoundError:
        print("Error: Data files not found.")
        return

    # 1. Move everything to VRAM
    print("Transferring data to GPU (VRAM)...")
    X_train = cp.array(X_train_cpu)
    X_test = cp.array(X_test_cpu)
    y_train = cp.array(y_train_cpu)
    y_test = cp.array(y_test_cpu)

    # Calculate weights on GPU
    neg_counts = cp.sum(y_train == 0)
    pos_counts = cp.sum(y_train == 1)
    scale_pos_weight = float(neg_counts / pos_counts)

    # 2. Configure Classifier
    clf = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        tree_method='hist',
        device='cuda',
        eval_metric='logloss'
    )

    print("Training XGBoost on GPU...")
    start_time = time.time()
    clf.fit(X_train, y_train)
    print(f"Training Time: {time.time() - start_time:.2f} seconds")

    # 3. Evaluation
    print("Evaluating XGBoost on GPU (No data transfer)...")
    
    # We pass the CuPy arrays (X_test, y_test) to the evaluation.
    # XGBoost will detect they are already on the same device as the model.
    try:
        evaluate_and_plot(clf, X_test, y_test, "XGBoost", label_mapping)
    except Exception as e:
        print(f"\nNote: evaluate_and_plot might need CPU arrays for plotting: {e}")
        # If the plotting library (matplotlib) complains about CuPy,
        # we pass the model (on GPU) and CPU data. 
        # The warning may reappear but the results will be correct.
        evaluate_and_plot(clf, X_test_cpu, y_test_cpu, "XGBoostGPU_VRAM", label_mapping)

if __name__ == "__main__":
    run()