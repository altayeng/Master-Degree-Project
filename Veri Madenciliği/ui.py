import pandas as pd
from datetime import datetime
import tkinter as tk
from tkinter import ttk
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers  # type: ignore
import os

# Disable oneDNN optimizations
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# File path definitions
database_file = "C:\\Users\\altay\\Desktop\\Master's Degree\\Bitirme Tezi Kodlar\\otoparkalgi\\parking-space-counter-master\\parking_data2.txt"
output_file = "C:\\Users\\altay\\Desktop\\Master's Degree\\Bitirme Tezi Kodlar\\otoparkalgi\\parking-space-counter-master\\preference_analysis.txt"

# Read data from the file and perform analysis
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

# Analyze by day and time period
day_counts = df['day_of_week'].value_counts(normalize=True) * 100
df['time_period'] = pd.cut(df['hour'], bins=[0, 3, 6, 9, 12, 15, 18, 21, 24],
                           labels=['00:00-03:00', '03:00-06:00', '06:00-09:00', '09:00-12:00',
                                   '12:00-15:00', '15:00-18:00', '18:00-21:00', '21:00-24:00'],
                           right=False)
time_counts = df['time_period'].value_counts(normalize=True) * 100

most_common_day = day_counts.idxmax()
most_common_time = time_counts.idxmax()
least_common_day = day_counts.idxmin()
least_common_time = time_counts.idxmin()

# Prepare data for deep learning
encoder = OneHotEncoder(sparse_output=False, drop='first')  # Drop the first category
X_day = encoder.fit_transform(df[['day_of_week']])
X_hour = df[['hour']].values
X = np.hstack((X_day, X_hour))

# Target variable (number of vehicles detected per hour)
y = df.groupby(['day_of_week', 'hour']).size().reindex(pd.MultiIndex.from_product(
    [df['day_of_week'].unique(), range(24)], names=['day_of_week', 'hour']), fill_value=0).values

X_combined, y_combined = [], []

for day in df['day_of_week'].unique():
    for hour in range(24):
        day_hour_data = df[(df['day_of_week'] == day) & (df['hour'] == hour)]
        if not day_hour_data.empty:
            encoded_day = encoder.transform([[day]])
            hour_encoded = np.array([[hour]])
            for _ in range(len(day_hour_data)):
                X_combined.append(np.hstack((encoded_day, hour_encoded)).reshape(1, -1))
                y_combined.append(y[day_counts.index.get_loc(day)] if day in day_counts.index else 0)

X_combined = np.vstack(X_combined)
y_combined = np.array(y_combined)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_combined, y_combined, test_size=0.2, random_state=42)

# Build the model
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1)

# Make predictions
predictions = model.predict(X_test)

# Write results to file
with open(output_file, 'w') as f:
    f.write("Vehicle Detection Rate by Day of the Week (%)\n")
    f.write(day_counts.to_string())
    f.write("\n\nVehicle Detection Rate by Time Period (%)\n")
    f.write(time_counts.to_string())
    f.write(f"\n\nMost Preferred Day: {most_common_day}\n")
    f.write(f"Most Preferred Time Period: {most_common_time}\n")
    f.write(f"Least Preferred Day: {least_common_day}\n")
    f.write(f"Least Preferred Time Period: {least_common_time}\n")

    f.write("\n\nPredictions:\n")
    for pred in predictions:
        f.write(f"{pred[0]:.2f}\n")

# Create DataFrame for UI
analysis_df = pd.DataFrame({
    "Category": ["Vehicle Detection Rate by Day of the Week (%)"] + list(day_counts.index) +
                ["", "Vehicle Detection Rate by Time Period (%)"] + list(time_counts.index),
    "Values": [""] + list(day_counts.values) + ["", ""] + list(time_counts.values)
})

analysis_df.loc[len(analysis_df)] = ["Most Preferred Day", most_common_day]
analysis_df.loc[len(analysis_df)] = ["Most Preferred Time Period", most_common_time]
analysis_df.loc[len(analysis_df)] = ["Least Preferred Day", least_common_day]
analysis_df.loc[len(analysis_df)] = ["Least Preferred Time Period", least_common_time]

# Function to start UI
def show_analysis_ui():
    root = tk.Tk()
    root.title("Preference Analysis")
    root.geometry("500x400")

    title_label = tk.Label(root, text="Preference Analysis Results", font=("Helvetica", 16, "bold"))
    title_label.pack(pady=10)

    tree = ttk.Treeview(root, columns=("Category", "Values"), show="headings", height=15)
    tree.heading("Category", text="Category")
    tree.heading("Values", text="Values")

    for index, row in analysis_df.iterrows():
        tree.insert("", "end", values=(row["Category"], f"{row['Values']:.2f}%" if isinstance(row["Values"], (int, float)) else row["Values"]))

    tree.pack(expand=True, fill="both")
    root.mainloop()

# Start UI
show_analysis_ui()
