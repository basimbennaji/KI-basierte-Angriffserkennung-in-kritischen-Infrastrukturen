import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
import kagglehub

def run():
    print("Lade verarbeitete Daten...")
    X_train = joblib.load('processed_data/X_train.pkl')
    y_train = joblib.load('processed_data/y_train.pkl')

    print("Rekonstruiere originale Spaltennamen...")
    path = kagglehub.dataset_download("solarmainframe/ids-intrusion-csv")
    first_file = os.path.join(path, [f for f in os.listdir(path) if f.endswith('.csv')][0])
    df_sample = pd.read_csv(first_file, nrows=0)
    df_sample.columns = df_sample.columns.str.strip()
    
    cols_to_drop = ['Timestamp', 'Flow ID', 'Source IP', 'Src IP', 'Dst IP', 'Destination IP', 'Dst Port', 'Label']
    feature_names = [c for c in df_sample.columns if c not in cols_to_drop]

    print("Trainiere Decision Tree für die Analyse...")
    clf = DecisionTreeClassifier(random_state=42, class_weight='balanced')
    clf.fit(X_train, y_train)

    print("Berechne Feature Importances...")
    importances = clf.feature_importances_
    
    # Die Top 10 wichtigsten Features finden
    indices = np.argsort(importances)[::-1][:10]
    top_features = [feature_names[i] for i in indices]
    top_importances = importances[indices]

    print("\n--- TOP 5 WICHTIGSTE MERKMALE ---")
    for i in range(5):
        print(f"{i+1}. {top_features[i]} ({top_importances[i]:.4f})")

    # Plot erstellen
    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_importances, y=top_features, palette="viridis")
    plt.title("Top 10 Entscheidungsmerkmale (Feature Importance)")
    plt.xlabel("Wichtigkeit (Mean Decrease Impurity / Gini)")
    plt.ylabel("Netzwerk-Eigenschaft")
    plt.tight_layout()
    plt.savefig("feature_importance.png")
    print("\nGrafik als 'feature_importance.png' gespeichert!")
    plt.show()

if __name__ == "__main__":
    run()