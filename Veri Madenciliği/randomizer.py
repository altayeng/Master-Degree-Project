import random
import datetime
from typing import List, Dict, Tuple

class ParkingDataGenerator:
    def __init__(self):
        # Sabit apartman sakinleri ve araçları (gerçekçi senaryo için)
        self.residents: Dict[str, Dict] = {
            "34ABC123": {"work_schedule": "regular", "weekend_probability": 0.3},
            "34DEF456": {"work_schedule": "regular", "weekend_probability": 0.4},
            "34GHJ789": {"work_schedule": "late", "weekend_probability": 0.6},
            "34KLM012": {"work_schedule": "early", "weekend_probability": 0.5},
            "34NPR345": {"work_schedule": "irregular", "weekend_probability": 0.7},
            "34STU678": {"work_schedule": "regular", "weekend_probability": 0.4},
            "34VYZ901": {"work_schedule": "regular", "weekend_probability": 0.3},
            "34WQX234": {"work_schedule": "night", "weekend_probability": 0.2},
        }
        
        # Misafir plakaları için il kodları
        self.city_codes = [str(i).zfill(2) for i in range(1, 82)]
        
        # İş programı tanımlamaları
        self.work_schedules = {
            "regular": {"entry": (8, 9), "exit": (17, 18)},  # Normal mesai
            "early": {"entry": (7, 8), "exit": (16, 17)},    # Erken mesai
            "late": {"entry": (10, 11), "exit": (19, 20)},   # Geç mesai
            "night": {"entry": (20, 21), "exit": (5, 6)},    # Gece mesai
            "irregular": {"entry": (0, 23), "exit": (0, 23)} # Düzensiz
        }

    def generate_visitor_plate(self) -> str:
        """Misafir araç plakası oluşturur"""
        city = random.choice(self.city_codes)
        letters = ''.join(random.choices('ABCDEFGHIJKLMNOPRSTUVYZ', k=3))
        numbers = ''.join(str(random.randint(0, 9)) for _ in range(3))
        return f"{city}{letters}{numbers}"

    def generate_timestamp(self, date: datetime.datetime, schedule_type: str, is_entry: bool) -> datetime.datetime:
        """Belirli bir program türüne göre zaman damgası oluşturur"""
        schedule = self.work_schedules[schedule_type]
        if is_entry:
            hour_range = schedule["entry"]
        else:
            hour_range = schedule["exit"]
            
        hour = random.randint(hour_range[0], hour_range[1])
        minute = random.randint(0, 59)
        second = random.randint(0, 59)
        
        return date.replace(hour=hour, minute=minute, second=second)

    def generate_daily_records(self, date: datetime.datetime) -> List[Tuple[str, datetime.datetime]]:
        """Belirli bir gün için tüm giriş-çıkış kayıtlarını oluşturur"""
        records = []
        is_weekend = date.weekday() >= 5
        
        # Sakinler için kayıtlar
        for plate, info in self.residents.items():
            # Hafta sonu kontrolü
            if is_weekend and random.random() > info["weekend_probability"]:
                continue
                
            # Giriş-çıkış kayıtları
            schedule = info["work_schedule"]
            entry_time = self.generate_timestamp(date, schedule, True)
            records.append((plate, entry_time))
            
            # Çıkış kaydı - bazı araçlar gece kalabilir
            if random.random() < 0.9:  # %90 olasılıkla çıkış yapar
                exit_time = self.generate_timestamp(date, schedule, False)
                if exit_time > entry_time:  # Mantıklı bir çıkış zamanı ise ekle
                    records.append((plate, exit_time))
        
        # Misafir araçlar için kayıtlar
        visitor_count = random.randint(2 if is_weekend else 0, 5 if is_weekend else 3)
        for _ in range(visitor_count):
            plate = self.generate_visitor_plate()
            # Misafirler genelde 2-4 saat kalır
            entry_hour = random.randint(10, 20)
            entry_time = date.replace(hour=entry_hour, minute=random.randint(0, 59))
            records.append((plate, entry_time))
            
            if random.random() < 0.95:  # %95 olasılıkla çıkış yapar
                stay_duration = random.randint(1, 4)  # 1-4 saat arası kalış
                exit_time = entry_time + datetime.timedelta(hours=stay_duration)
                if exit_time.date() == date.date():  # Aynı gün içindeyse ekle
                    records.append((plate, exit_time))

        return sorted(records, key=lambda x: x[1])  # Zamana göre sırala

    def generate_dataset(self, start_date: str, end_date: str, output_file: str):
        """Belirtilen tarih aralığı için veri seti oluşturur"""
        start = datetime.datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.datetime.strptime(end_date, "%Y-%m-%d")
        current = start
        
        all_records = []
        while current <= end:
            daily_records = self.generate_daily_records(current)
            all_records.extend(daily_records)
            current += datetime.timedelta(days=1)

        # Dosyaya yazma
        with open(output_file, "w") as file:
            for plate, timestamp in all_records:
                file.write(f"{plate},{timestamp.strftime('%Y-%m-%d,%H:%M:%S')}\n")
                
        return len(all_records)

# Kullanım örneği
if __name__ == "__main__":
    generator = ParkingDataGenerator()
    
    # Dosya yolu
    database_file = "C:\\Users\\altay\\Desktop\\Master's Degree\\Bitirme Tezi Kodlar\\otoparkalgi\\parking-space-counter-master\\parking_data2.txt"
    
    # Veri üretimi
    total_records = generator.generate_dataset(
        start_date="2023-01-01",
        end_date="2024-10-31",
        output_file=database_file
    )
    
    print(f"Toplam {total_records} kayıt oluşturuldu ve {database_file} dosyasına yazıldı.")