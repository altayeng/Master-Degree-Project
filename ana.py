import tkinter as tk
from tkinter import Canvas
import subprocess
import cv2
from PIL import Image, ImageTk

database_file = "C:\\Users\\altay\\Desktop\\Master's Degree\\Bitirme Tezi Kodlar\\plaka okuma\\parking_data.txt"

# Ana pencere oluşturma
root = tk.Tk()
root.title("Otopark Yönetimi")
root.geometry("800x600")

# Görüntüleri göstermek için Canvas
canvas = Canvas(root, width=640, height=480)
canvas.pack()

def update_parking_status():
    global database_file
    try:
        with open(database_file, 'r') as f:
            data = f.readlines()
            empty_slots = [line.strip() for line in data if line.startswith("Empty Spot")]
            if empty_slots:
                parking_status_label.config(text="\n".join(empty_slots))
            else:
                parking_status_label.config(text="No empty slots available.")
    except FileNotFoundError:
        parking_status_label.config(text="Database file not found.")
    except Exception as e:
        parking_status_label.config(text=f"An error occurred: {str(e)}")

def run_parking_status():
    try:
        subprocess.Popen(["python", r"C:\Users\altay\Desktop\Master's Degree\Bitirme Tezi Kodlar\otoparkalgi\parking-space-counter-master\main.py"])
        update_parking_status()
    except Exception as e:
        message_label.config(text=f"Hata: {e}")

def run_plate_recognition():
    try:
        subprocess.Popen(["python", r"C:\Users\altay\Desktop\Master's Degree\Bitirme Tezi Kodlar\plaka okuma\plate2.py"])
        update_parking_status()
    except Exception as e:
        message_label.config(text=f"Hata: {e}")

def display_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    imgtk = ImageTk.PhotoImage(image=img)
    canvas.create_image(0, 0, anchor="nw", image=imgtk)
    root.mainloop()

# Otopark durumu ve plaka okuma butonları
parking_status_button = tk.Button(root, text="Otopark Durumu", command=run_parking_status)
parking_status_button.pack()

plate_recognition_button = tk.Button(root, text="Plaka Okuma", command=run_plate_recognition)
plate_recognition_button.pack()

# Bilgilendirme alanı
message_label = tk.Label(root, text="")
message_label.pack()

# Park durumu bilgisi alanı
parking_status_label = tk.Label(root, text="", justify=tk.LEFT)
parking_status_label.pack()

# Ana döngü
root.mainloop()
