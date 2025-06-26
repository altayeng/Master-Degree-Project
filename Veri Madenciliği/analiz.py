import pandas as pd
from datetime import datetime
from collections import defaultdict

# Dosya yolu
database_file = "C:\\Users\\altay\\Desktop\\Master's Degree\\Bitirme Tezi Kodlar\\otoparkalgi\\parking-space-counter-master\\parking_data.txt"
output_file = "C:\\Users\\altay\\Desktop\\Master's Degree\\Bitirme Tezi Kodlar\\otoparkalgi\\parking-space-counter-master\\tercih_analizi.txt"

# Plaka skorlarını tutmak için sözlük
plate_scores = defaultdict(int)

# Veriyi dosyadan oku
data = []
with open(database_file, 'r') as f:
    for line in f:
        line = line.strip().split(',')
        if len(line) == 3:
            plate, date_str, time_str = line
            datetime_obj = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S")
            
            # Plaka skorunu  
            if plate in plate_scores:
                plate_scores[plate] = min(plate_scores[plate] + 1, 5)  # Maksimum 5 skor
            else:
                plate_scores[plate] = 0  # Yeni plaka için skor 0
            
            data.append({
                "plate": plate,
                "datetime": datetime_obj,
                "day_of_week": datetime_obj.strftime("%A"),
                "hour": datetime_obj.hour,
                "score": plate_scores[plate]  # Skoru veri setine ekle
            })

df = pd.DataFrame(data)

# Gün ve saat dilimi analizi 
day_counts = df['day_of_week'].value_counts(normalize=True) * 100
df['time_period'] = pd.cut(df['hour'], bins=[0, 3, 6, 9, 12, 15, 18, 21, 24], 
                           labels=['00:00-03:00', '03:00-06:00', '06:00-09:00', '09:00-12:00',
                                   '12:00-15:00', '15:00-18:00', '18:00-21:00', '21:00-24:00'],
                           right=False)
time_counts = df['time_period'].value_counts(normalize=True) * 100

most_common_day = day_counts.idxmax()
most_common_time = time_counts.idxmax()

# Plaka skorlarının analizi 
plate_score_summary = df.groupby('plate')['score'].max().reset_index()
high_frequency_plates = plate_score_summary[plate_score_summary['score'] >= 3]

# Sonuçları tercih_analizi.txt isimli dosyaya yazdır
with open(output_file, 'w') as f:
    f.write("Haftanın Günlerine Göre Araç Algılama Oranı (%)\n")
    f.write(day_counts.to_string())
    f.write("\n\nSaat Dilimlerine Göre Araç Algılama Oranı (%)\n")
    f.write(time_counts.to_string())
    f.write(f"\n\nEn çok tercih edilen gün: {most_common_day}\n")
    f.write(f"En çok tercih edilen saat dilimi: {most_common_time}\n")
    
    # Sık gelen araçların bilgilerini yaz
    f.write("\n\nSık Gelen Araçlar (Skor >= 3):\n")
    f.write(high_frequency_plates.to_string())