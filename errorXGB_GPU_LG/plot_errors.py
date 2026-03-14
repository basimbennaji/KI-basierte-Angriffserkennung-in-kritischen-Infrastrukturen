import matplotlib.pyplot as plt
import numpy as np

# 1. Daten definieren
models = ['Logistische Regression', 'XGBoost (GPU)']
fp_values = [15906, 22]
fn_values = [998, 1]

# Positionen für die Balken auf der X-Achse
x = np.arange(len(models))
width = 0.35  # Breite der Balken

# 2. Plot erstellen
fig, ax = plt.subplots(figsize=(8, 6))

# Balken zeichnen
rects1 = ax.bar(x - width/2, fp_values, width, label='False Positives (Fehlalarme)', color='#1f77b4')
rects2 = ax.bar(x + width/2, fn_values, width, label='False Negatives (Übersehene Angriffe)', color='#ff7f0e')

# 3. Achsen, Labels und Titel setzen
ax.set_ylabel('Anzahl (logarithmische Skala)', fontsize=12)
ax.set_title('Vergleich der absoluten Fehlklassifikationen', fontsize=14, pad=20)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=12)
ax.legend(fontsize=11)

# Logarithmische Y-Achse aktivieren
ax.set_yscale('log')

# Den sichtbaren Bereich der Y-Achse etwas nach oben erweitern, 
# damit die Zahlen über den Balken nicht abgeschnitten werden
ax.set_ylim(0.5, 50000)

# 4. Funktion, um die exakten Zahlen auf die Balken zu schreiben
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{int(height)}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 5),  # 5 Punkte Abstand nach oben
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

autolabel(rects1)
autolabel(rects2)

# Layout anpassen, damit nichts abgeschnitten wird
fig.tight_layout()

# 5. Graph als hochauflösendes Bild speichern
plt.savefig('error_comparison.png', dpi=300, bbox_inches='tight')
print("Graph erfolgreich als 'error_comparison.png' gespeichert!")

# Bild anzeigen
plt.show()