import cv2
import pytesseract
import os
from datetime import datetime
import time  # Zaman ölçümü için ekledik

# Tesseract ortam değişkeni
os.environ["TESSDATA_PREFIX"] = r"C:\Program Files\Tesseract-OCR\tessdata"

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Kaynak dosya (.mp4 veya .jpg)
input_source = "C:\\Users\\altay\\Desktop\\Master's Degree\\Bitirme Tezi Kodlar\\otoparkalgi\\parking-space-counter-master\\2.jpg"
database_file = "C:\\Users\\altay\\Desktop\\Master's Degree\\Bitirme Tezi Kodlar\\otoparkalgi\\parking-space-counter-master\\parking_data.txt"

def save_plate_data(plate):
    plate = plate.strip()
    date = datetime.now().strftime("%Y-%m-%d")
    hour = datetime.now().strftime("%H:%M:%S")

    if not os.path.exists(database_file):
        with open(database_file, 'w') as f:
            f.write("Plate,Date,Hour\n")  # Başlık ekledik

    # Mevcut plakaları oku
    with open(database_file, 'r') as f:
        existing_plates = set(line.split(",")[0] for line in f if line.strip() and "Plate" not in line)

    # Plaka dosyada yoksa yeni kayıt ekle
    if plate not in existing_plates:
        with open(database_file, 'a') as f:
            f.write(f"{plate},{date},{hour}\n")

def process_frame(frame):
    start_time = time.time()  # İşlem başlangıç zamanı
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    filter = cv2.bilateralFilter(gray, 11, 17, 17)
    edges = cv2.Canny(filter, 170, 200)

    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    found_plate = False
    plate_text = ""

    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        epsilon = 0.018 * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            plate_img = frame[y:y+h, x:x+w]

            plate_gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
            _, plate_thresh = cv2.threshold(plate_gray, 150, 255, cv2.THRESH_BINARY)

            plate_text = pytesseract.image_to_string(plate_thresh, config="--psm 8")
            plate_text = plate_text.replace(" ", "").replace("\n", "")

            if plate_text.strip():
                print("Algılanan Plaka:", plate_text)
                cv2.drawContours(frame, [approx], -1, (0, 255, 0), 3)
                found_plate = True

                if plate_text != "":
                    save_plate_data(plate_text.strip())
                break

    end_time = time.time()  # İşlem bitiş zamanı
    elapsed_time = end_time - start_time

    if found_plate:
        print(f"Plaka algılandı: {plate_text} - Süre: {elapsed_time:.2f} saniye")
    else:
        print(f"Plaka bulunamadı - Süre: {elapsed_time:.2f} saniye")

    cv2.imshow("Detected Plate", frame)
    cv2.waitKey(0)

# Dosya uzantısını kontrol et
if input_source.lower().endswith('.mp4'):
    cap = cv2.VideoCapture(input_source)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        process_frame(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
elif input_source.lower().endswith('.jpg'):
    frame = cv2.imread(input_source)
    if frame is not None:
        process_frame(frame)
    else:
        print("Görüntü yüklenemedi.")
else:
    print("Desteklenmeyen dosya formatı.")

cv2.destroyAllWindows()
