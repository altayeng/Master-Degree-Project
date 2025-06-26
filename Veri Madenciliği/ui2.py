import pandas as pd
from datetime import datetime
import tkinter as tk
from tkinter import ttk
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, classification_report
import os

# oneDNN özel işlemlerini kapatma
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Dosya yolu tanımları
database_file = "C:\\Users\\altay\\Desktop\\Master's Degree\\Bitirme Tezi Kodlar\\otoparkalgi\\parking-space-counter-master\\parking_data2.txt"
output_file = "C:\\Users\\altay\\Desktop\\Master's Degree\\Bitirme Tezi Kodlar\\otoparkalgi\\parking-space-counter-master\\tercih_analizi.txt"

# Veriyi dosyadan oku ve analiz yap
data = []
with open(database_file, 'r') as f:
    for line in f:
        line = line.strip().split(',')
        if len(line) == 3:
            plate, date_str, time_str = line
            datetime_obj = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S")
            data.append({
                "plate": plate,
                "datetime": datetime_obj,
                "day_of_week": datetime_obj.strftime("%A"),
                "hour": datetime_obj.hour
            })

df = pd.DataFrame(data)

# Gün ve saat dilimi analizini yap
day_counts = df['day_of_week'].value_counts(normalize=True) * 100
df['time_period'] = pd.cut(df['hour'], bins=[0, 3, 6, 9, 12, 15, 18, 21, 24],
                           labels=['00:00-03:00', '03:00-06:00', '06:00-09:00', '09:00-12:00',
                                   '12:00-15:00', '15:00-18:00', '18:00-21:00', '21:00-24:00'],
                           right=False)
time_counts = df['time_period'].value_counts(normalize=True) * 100

most_common_day = day_counts.idxmax()
most_common_time = time_counts.idxmax()

# Veri ön işleme
encoder = OneHotEncoder(sparse_output=False, drop='first')  # İlk kategoriyi düşür
X_day = encoder.fit_transform(df[['day_of_week']])
X_hour = df[['hour']].values
X = np.hstack((X_day, X_hour))

# Hedef değişken (örnek olarak 'time_period' sınıflandırması)
y = df['time_period'].cat.codes

# Eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modellerin tanımlanması
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

# Modelleri eğitme ve değerlendirme
results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    results.append({
        "Model": name,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })

    print(f"\n{name} Classification Report:\n")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Sonuçları DataFrame'e dönüştürme
results_df = pd.DataFrame(results)

# Sonuçları dosyaya yaz
with open(output_file, 'w') as f:
    f.write("Makine Öğrenimi Model Sonuçları\n")
    f.write(results_df.to_string(index=False))

# UI Başlatma Fonksiyonu

def show_results_ui():
    root = tk.Tk()
    root.title("Model Değerlendirme Sonuçları")
    root.geometry("600x400")

    # Tablo oluşturma
    tree = ttk.Treeview(root, columns=("Model", "Precision", "Recall", "F1 Score"), show="headings", height=15)
    tree.heading("Model", text="Model")
    tree.heading("Precision", text="Precision")
    tree.heading("Recall", text="Recall")
    tree.heading("F1 Score", text="F1 Score")

    # Sonuçları ekle
    for _, row in results_df.iterrows():
        tree.insert("", "end", values=(row["Model"], f"{row['Precision']:.2f}", f"{row['Recall']:.2f}", f"{row['F1 Score']:.2f}"))

    tree.pack(expand=True, fill="both")
    root.mainloop()

# UI'yi başlat
show_results_ui()
