# KI-basierte Angriffserkennung in kritischen Infrastrukturen (KRITIS)

Dieses Repository enthält den vollständigen Python-Quellcode für die Evaluierung verschiedener Machine-Learning-Modelle zur netzwerkbasierten Anomalieerkennung (Intrusion Detection). 

Das Projekt entstand im Rahmen einer wissenschaftlichen Untersuchung und fokussiert sich auf die technische Machbarkeit, Ressourceneffizienz und Praxistauglichkeit von KI-Modellen in kritischen Infrastrukturen. Ein besonderes Augenmerk liegt auf der **Minimierung von Fehlalarmen (False Positives)**, um *Alert Fatigue* (Alarmmüdigkeit) im Security Operations Center (SOC) zu vermeiden.

Als Datengrundlage dient der offizielle [CSE-CIC-IDS2018 Datensatz](https://www.unb.ca/cic/datasets/ids-2018.html).

---

## 🗂️ Projekt- und Ordnerstruktur

Das Repository ist modular aufgebaut, um die Datenvorverarbeitung sauber von der Modell-Evaluierung zu trennen. Jeder Algorithmus verfügt über einen eigenen Ordner mit dem zugehörigen Trainingsskript, den verwendeten Befehlen und den generierten Visualisierungen.

### ⚙️ Core-Skripte & Daten
* `data/` – Zielordner für die heruntergeladenen CSV-Rohdaten *(nicht im Repo enthalten)*.
* `processed_data/` – Zielordner für die durch das Preprocessing bereinigten und normalisierten Daten.
* `preprocessing.py` – Hauptskript für Datenbereinigung, Feature-Engineering, Z-Score-Normalisierung und Train-Test-Split.
* `visualization_utils.py` – Hilfsfunktionen zur Generierung einheitlicher Plots (Konfusionsmatrizen, ROC-Kurven).
* `requirements.txt` – Liste aller benötigten Python-Abhängigkeiten.

### 🧠 Machine-Learning-Modelle
* `cm_logistic_regression/` – Logistische Regression (Baseline)
* `decision_tree/` – Entscheidungsbaum
* `random_forest/` – Random Forest
* `neural_network/` – Künstliches Neuronales Netz (MLP)
* `XGBoost/` – Gradient Boosting (Standard CPU)
* `xgboostGPU/` – XGBoost (Hardwarebeschleunigt via CUDA)
* `xgboostGPUVRAM/` – XGBoost (Maximale Performance durch Vorabladen in den VRAM)

### 📊 Spezifische Auswertungen
* `feature_importance/` – Analyse und Visualisierung der relevantesten Netzwerk-Features für die Klassifikation.
* `errorXGB_GPU_LG/` – Generiert einen logarithmischen Vergleichsgraphen der Fehlklassifikationen zwischen XGBoost und der Logistischen Regression.

---

## 🚀 Installation & Voraussetzungen

1. **Repository klonen:**
   ```bash
   git clone [https://github.com/DEIN-USERNAME/DEIN-REPO-NAME.git](https://github.com/DEIN-USERNAME/DEIN-REPO-NAME.git)
   cd DEIN-REPO-NAME

2. **Virtuelle Umgebung erstellen & Abhängigkeiten installieren:**

    ```bash
    python -m venv venv
    source venv/bin/activate  # Auf Windows: venv\Scripts\activate
    pip install -r requirements.txt
(Hinweis: Für das performante Training des Neuronalen Netzes und der GPU-Varianten von XGBoost wird eine CUDA-fähige NVIDIA-Grafikkarte sowie die entsprechende PyTorch/XGBoost-GPU-Konfiguration benötigt.)

## 💻 Ausführung & Reproduktion
1. **Datenvorverarbeitung**

Laden Sie den CSE-CIC-IDS2018 Datensatz herunter und platzieren Sie die entpackten CSV-Dateien im Ordner data/. Führen Sie anschließend das Preprocessing aus:

    ```bash
    python preprocessing.py
Dies liest die Daten ein, entfernt irrelevante Identifikatoren (wie IP-Adressen), normalisiert die Werte und speichert die finalen Trainings- und Testdaten im Ordner processed_data/ ab.

2. **Modelle trainieren und evaluieren**

Wechseln Sie in den Ordner des gewünschten Modells und führen Sie das dortige Skript aus.

Beispiel für das XGBoost (GPU+VRAM) Modell:

    cd xgboostGPUVRAM
    python model_xgboostGPUVRAM.py

Das Skript lädt die vorverarbeiteten Daten, trainiert das Modell, evaluiert es auf dem Test-Set (ca. 1,57 Millionen Einträge) und speichert die Konfusionsmatrix sowie die ROC-Kurve als .png-Dateien im selben Ordner ab. Spezifische Konsolenparameter können der jeweiligen command.txt entnommen werden.

## 🏆 Kernergebnisse der Studie
Die Evaluierung auf dem zurückgehaltenen Test-Datensatz brachte folgende zentrale Erkenntnisse hervor:

1. Baumbasierte Modelle dominieren: XGBoost und Decision Trees sind komplexen Deep-Learning-Ansätzen (MLP) und linearen Modellen bei diesen tabellarischen Netzwerkdaten deutlich überlegen.

2. Minimierung von Alert Fatigue: Das XGBoost-Modell produzierte bei über 1,46 Millionen harmlosen Verbindungen lediglich 22 Fehlalarme (False Positives) und übersah nur 1 bis 3 tatsächliche Angriffe. Zum Vergleich: Das Neuronale Netz verursachte 1.371 Fehlalarme, die Logistische Regression sogar 15.906.

3. Extreme Ressourceneffizienz: Durch die Kombination von CUDA-Hardwarebeschleunigung und VRAM-Vorabladen (xgboostGPUVRAM) konnte die Trainingszeit für über 6 Millionen Datensätze auf unter 2 Sekunden reduziert werden. Dies ermöglicht ein ressourcenschonendes Continuous Learning im produktiven Betrieb.

## 📄 Lizenz
Dieses Projekt ist unter der MIT License lizenziert. Weitere Details finden Sie in der entsprechenden Lizenz-Datei.