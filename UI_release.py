import cv2
import pytesseract
import os
import tkinter as tk
from tkinter import ttk, Label, Button
from PIL import Image, ImageTk
import threading
from datetime import datetime
import numpy as np
from util import get_parking_spots_bboxes, empty_or_not

# MQTT için gereken kütüphane
import paho.mqtt.client as mqtt

class ModernParkingSystem:
    def __init__(self, root):
        self.root = root
        self.root.title("Akıllı Otopark Yönetim Sistemi")
        self.root.configure(bg="#f5f6fa")
        self.root.geometry("1280x850")
        
        # Global değişkenler
        self.selected_spot = None
        self.spots = []
        
        # Doluluk %50 altındaysa indirim butonu aktif olsun diye takip edeceğimiz flag
        self.discount_active = False
        
        # Yollar
        self.PATHS = {
            'tessdata': r"C:\Program Files\Tesseract-OCR\tessdata",
            'tesseract': r'C:\Program Files\Tesseract-OCR\tesseract.exe',
            'mask': r"C:\Users\altay\Desktop\Master's Degree\Bitirme Tezi Kodlar\otoparkalgi\parking-space-counter-master\mask_1920_1080.png",
            'video': r"C:\Users\altay\Desktop\Master's Degree\Bitirme Tezi Kodlar\otoparkalgi\parking-space-counter-master\parking_1920_1080_loop.mp4",
            'parking_data': r"C:\Users\altay\Desktop\Master's Degree\Bitirme Tezi Kodlar\otoparkalgi\parking-space-counter-master\parking_data.txt",
            'plate_image': r"C:\Users\altay\Desktop\Master's Degree\Bitirme Tezi Kodlar\otoparkalgi\parking-space-counter-master\2.jpg"
        }
        
        # Tesseract ayarları
        os.environ["TESSDATA_PREFIX"] = self.PATHS['tessdata']
        pytesseract.pytesseract.tesseract_cmd = self.PATHS['tesseract']
        
        # Stil ayarları
        self.setup_styles()
        
        # MQTT bağlantısını başlat
        self.connect_mqtt()
        
        # Arayüzü oluştur
        self.create_gui()
        
        # Otopark analizini başlat
        self.start_analysis()

    def connect_mqtt(self):
        """MQTT sunucusuna bağlanır ve arka planda dinlemeyi başlatır."""
        self.mqtt_client = mqtt.Client()
        # Burada broker.emqx.io örnek broker’dır, isterseniz kendi broker’ınızı girebilirsiniz.
        self.mqtt_client.connect("broker.emqx.io", 1883, 60)
        self.mqtt_client.loop_start()

    def setup_styles(self):
        style = ttk.Style(self.root)
        style.theme_use("clam")
        
        style.configure(
            "Header.TLabel",
            font=("Segoe UI", 10, "bold"),
            foreground="#2c3e50",
            background="#f5f6fa",
            padding=2
        )
        
        style.configure(
            "Info.TLabel",
            font=("Segoe UI", 10),
            foreground="#34495e",
            background="#f5f6fa",
            padding=5
        )
        
        style.configure(
            "Status.TLabel",
            font=("Segoe UI", 10),
            foreground="#2980b9",
            background="#f5f6fa",
            padding=5
        )
        
        style.configure(
            "Action.TButton",
            font=("Segoe UI", 10),
            padding=10,
            background="#3498db"
        )

    def create_gui(self):
        # Ana konteyner
        main_container = ttk.Frame(self.root)
        main_container.pack(fill="both", expand=True, padx=20, pady=10)
        
        # Kontrol Paneli
        control_panel = ttk.Frame(main_container)
        control_panel.pack(fill="x", pady=10)
        
        # Plaka Okuma Bölümü
        plate_frame = ttk.LabelFrame(control_panel, text="Plaka Tanıma", padding=10)
        plate_frame.pack(fill="x", pady=5)
        
        self.plate_button = ttk.Button(
            plate_frame,
            text="Plaka Oku",
            style="Action.TButton",
            command=lambda: threading.Thread(target=self.read_plate).start()
        )
        self.plate_button.pack(side="left", padx=10)
        
        self.plate_label = ttk.Label(
            plate_frame,
            text="Araç Bekleniyor",
            style="Info.TLabel"
        )
        self.plate_label.pack(side="left", padx=10)
        
        # Durum Paneli
        status_frame = ttk.LabelFrame(control_panel, text="Otopark Durumu", padding=10)
        status_frame.pack(fill="x", pady=10)
        
        self.parking_label = ttk.Label(
            status_frame,
            text="Otopark durumu yükleniyor...",
            style="Status.TLabel"
        )
        self.parking_label.pack(side="left", padx=10)

        # İndirim Yap butonu (başlangıçta pasif)
        self.discount_button = ttk.Button(
            status_frame,
            text="İndirim Uygula",
            style="Action.TButton",
            command=self.publish_discount
        )
        self.discount_button.pack(side="left", padx=10)
        self.discount_button["state"] = "disabled"  # %50'nin altına düşene kadar kapalı kalsın
        
        self.selected_label = ttk.Label(
            status_frame,
            text="Seçilen Park Alanı: -",
            style="Info.TLabel"
        )
        self.selected_label.pack(side="right", padx=10)
        
        # Kullanıcı Bilgisi Paneli
        user_info_frame = ttk.LabelFrame(control_panel, text="Kullanıcı Bilgisi", padding=10)
        user_info_frame.pack(fill="x", pady=5)
        
        self.user_info_label = ttk.Label(
            user_info_frame,
            text="",
            style="Info.TLabel"
        )
        self.user_info_label.pack(fill="x", padx=10)
        
        # Video Görüntüleme
        video_frame = ttk.LabelFrame(main_container, text="Otopark Görüntüsü", padding=10)
        video_frame.pack(fill="both", expand=True, pady=10)

        self.parking_image_label = ttk.Label(video_frame)
        self.parking_image_label.pack(expand=True)
        
        # Görüntüye tıklama olayı
        self.parking_image_label.bind("<Button-1>", self.handle_click)

    def handle_click(self, event):
        """Videodaki park alanına tıklanması durumunda seçili alanı günceller."""
        # Etiket boyutu: 1000x600 -> Orijinal boyut: 1920x1080
        click_x = event.x * (1920/1000)
        click_y = event.y * (1080/600)
        
        for idx, (x, y, w, h) in enumerate(self.spots):
            if x <= click_x <= x+w and y <= click_y <= y+h:
                self.selected_spot = idx+1
                self.selected_label.config(text=f"Seçilen Park Alanı: {self.selected_spot}")
                break

    def process_frame(self, frame):
        """Plaka okumak için gelen görüntü üzerinde OCR işlemi yapar."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        filtered = cv2.bilateralFilter(gray, 11, 17, 17)
        edges = cv2.Canny(filtered, 170, 200)
        
        contours, _ = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
        
        for contour in contours:
            perimeter = cv2.arcLength(contour, True)
            epsilon = 0.018 * perimeter
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                plate_img = frame[y:y+h, x:x+w]
                
                plate_gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
                _, plate_thresh = cv2.threshold(plate_gray, 150, 255, cv2.THRESH_BINARY)
                
                text = pytesseract.image_to_string(plate_thresh, config="--psm 8")
                text = ''.join(filter(str.isalnum, text))
                
                if text.strip():
                    return text.strip()
        return None

    def get_or_update_plate_data(self, plate):
        """Plakayı kayıt dosyasına yazar ve eski/yeni müşteri durumunu belirler."""
        plate = plate.strip()
        date = datetime.now().strftime("%Y-%m-%d")
        hour = datetime.now().strftime("%H:%M:%S")
        
        # Kayıt dosyası yoksa oluştur
        if not os.path.exists(self.PATHS['parking_data']):
            with open(self.PATHS['parking_data'], 'w', encoding='utf-8') as f:
                f.write("Plate,Date,Hour\n")
        
        with open(self.PATHS['parking_data'], 'r', encoding='utf-8') as f:
            existing_plates = [line.split(",")[0] for line in f if line.strip() and "Plate" not in line]
        
        occurrences = existing_plates.count(plate)
        is_new = (occurrences == 0)
        new_score = min(occurrences + 1, 5)
        
        # Dosyaya yeni satır ekle
        with open(self.PATHS['parking_data'], 'a', encoding='utf-8') as f:
            f.write(f"{plate},{date},{hour}\n")
        
        return is_new, new_score, date, hour

    def read_plate(self):
        """Örnek bir resim üzerindeki plakayı okur ve ekrana yazar."""
        frame = cv2.imread(self.PATHS['plate_image'])
        if frame is None:
            self.plate_label.config(text="❌ Görüntü yüklenemedi!")
            return
        
        recognized_plate = self.process_frame(frame)
        
        if recognized_plate:
            is_new, score, date, hour = self.get_or_update_plate_data(recognized_plate)
            status_text = "🆕 Yeni Müşteri" if is_new else "↩️ Kayıtlı Müşteri"
            discount_text = ""

            # Skor 5 ise %20 indirim mesajı
            if score == 5:
                discount_text = "\n💰 İndirim Uygulandı: %20!"

            self.plate_label.config(text=f"📝 Algılanan Plaka: {recognized_plate}")
            self.user_info_label.config(
                text=f"{status_text}\n"
                     f"📅 Tarih: {date}\n"
                     f"🕒 Saat: {hour}\n"
                     f"⭐ Skor: {score}"
                     f"{discount_text}"
            )
        else:
            self.plate_label.config(text="❌ Yeni Müşteri!")
            self.user_info_label.config(text="")

    def analyze_parking(self):
        """Video kaynağından gelen kareleri analiz ederek otopark doluluğunu hesaplar."""
        cap = cv2.VideoCapture(self.PATHS['video'])
        mask = cv2.imread(self.PATHS['mask'], 0)
        
        connected_components = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)
        self.spots = get_parking_spots_bboxes(connected_components)
        spots_status = [None] * len(self.spots)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                # Döngü bitince yeniden başa sar
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            
            # Her karede park alanlarını kontrol et
            for idx, (x, y, w, h) in enumerate(self.spots):
                spot_img = frame[y:y+h, x:x+w]
                spots_status[idx] = empty_or_not(spot_img)
                color = (0, 255, 0) if spots_status[idx] else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, str(idx+1), (x+5, y+25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            
            empty_count = sum(spots_status)
            total = len(self.spots)
            fullness_percentage = (empty_count * 100) // total
            
            self.parking_label.config(
                text=f"🅿️ Boş Alan: {empty_count}/{total} | 📊 Doluluk: %{fullness_percentage}"
            )
            
            # Doluluk %50’nin altına düştüyse indirim butonunu aç
            if fullness_percentage < 50:
                self.discount_active = True
                self.discount_button["state"] = "normal"
            else:
                self.discount_active = False
                self.discount_button["state"] = "disabled"
            
            # Görüntüyü GUI'de göster
            frame = cv2.resize(frame, (1000, 600))
            img = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            self.parking_image_label.config(image=img)
            self.parking_image_label.image = img
            
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

    def publish_discount(self):
        """İndirim bildirimi MQTT üzerinden gönderilir."""
        if self.discount_active:
            # Doluluk %50 altındaysa indirim mesajı gönder
            message = "Otopark doluluk oranı %50'nin altında! İndirim uygulanmıştır."
            self.mqtt_client.publish("parking/discount", message)
        else:
            # Doluluk %50’nin üzerinde ise buton pasif olmalı ama yine de kontrol
            pass

    def start_analysis(self):
        """Analiz iş parçacığını başlatır."""
        threading.Thread(target=self.analyze_parking, daemon=True).start()

if __name__ == "__main__":
    root = tk.Tk()
    app = ModernParkingSystem(root)
    root.mainloop()
