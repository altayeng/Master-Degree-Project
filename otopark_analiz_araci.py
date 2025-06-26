import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

# Türkçe font ayarları
plt.rcParams['font.family'] = ['DejaVu Sans']
sns.set_style("whitegrid")
sns.set_palette("husl")

class OtoparkAnalizAraci:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Otopark Analiz ve Yönetim Sistemi")
        self.root.geometry("1400x900")
        self.root.configure(bg='#f0f0f0')

        # Veri yükleme
        self.veri_yukle()

        # Metrikleri hesapla
        self.metrikleri_hesapla()

        # UI oluştur
        self.ui_olustur()

    def veri_yukle(self):
        """Otopark verilerini yükle ve işle"""
        try:
            # Veri dosyası yolu
            dosya_yolu = "parking_data2.txt"

            # Veriyi oku
            veri = []
            with open(dosya_yolu, 'r') as f:
                for satir in f:
                    satir = satir.strip().split(',')
                    if len(satir) == 3:
                        plaka, tarih_str, saat_str = satir
                        tarih_saat = datetime.strptime(f"{tarih_str} {saat_str}", "%Y-%m-%d %H:%M:%S")
                        veri.append({
                            "plaka": plaka,
                            "tarih_saat": tarih_saat,
                            "tarih": tarih_saat.date(),
                            "saat": tarih_saat.hour,
                            "gun": tarih_saat.strftime("%A"),
                            "gun_tr": self.gun_cevirici(tarih_saat.strftime("%A")),
                            "hafta_gunu": tarih_saat.weekday(),
                            "ay": tarih_saat.month,
                            "yil": tarih_saat.year
                        })

            self.df = pd.DataFrame(veri)

            # Ek hesaplamalar
            self.df['saat_dilimi'] = pd.cut(self.df['saat'],
                                          bins=[0, 6, 12, 18, 24],
                                          labels=['Gece (00-06)', 'Sabah (06-12)',
                                                'Sonra (12-18)', 'Akşam (18-24)'],
                                          right=False)

            # Otopark kapasitesi (varsayılan)
            self.kapasite = 100

            print(f"Toplam {len(self.df)} kayıt yüklendi.")

        except Exception as e:
            messagebox.showerror("Hata", f"Veri yüklenirken hata oluştu: {str(e)}")

    def gun_cevirici(self, gun):
        """İngilizce gün adlarını Türkçeye çevir"""
        gun_sozlugu = {
            'Monday': 'Pazartesi',
            'Tuesday': 'Salı',
            'Wednesday': 'Çarşamba',
            'Thursday': 'Perşembe',
            'Friday': 'Cuma',
            'Saturday': 'Cumartesi',
            'Sunday': 'Pazar'
        }
        return gun_sozlugu.get(gun, gun)

    def metrikleri_hesapla(self):
        """Tüm otopark metriklerini hesapla"""

        # 1. Otopark Doluluk Oranı (ODO)
        gunluk_arac_sayisi = self.df.groupby('tarih').size()
        self.odo = (gunluk_arac_sayisi.mean() / self.kapasite) * 100

        # 2. Araç Yoğunluk Seviyesi (VD) - m² başına araç
        otopark_alani = 20  # m² (varsayılan)
        self.vd = gunluk_arac_sayisi.mean() / otopark_alani

        # 3. Zirve Zaman Faktörü (PTF)
        saatlik_arac = self.df.groupby(['tarih', 'saat']).size()
        zirve_saatler = saatlik_arac.groupby('saat').mean().nlargest(3).index
        zirve_ortalama = saatlik_arac.groupby('saat').mean()[zirve_saatler].mean()
        gunluk_ortalama = gunluk_arac_sayisi.mean()
        self.ptf = zirve_ortalama / gunluk_ortalama if gunluk_ortalama > 0 else 0

        # 4. Kullanıcı Süre Deseni (UDP) - varsayılan park süresi
        # Aynı plaka için ardışık kayıtlar arasındaki süre
        park_sureleri = []
        for plaka in self.df['plaka'].unique():
            plaka_verileri = self.df[self.df['plaka'] == plaka].sort_values('tarih_saat')
            if len(plaka_verileri) > 1:
                for i in range(len(plaka_verileri) - 1):
                    sure = (plaka_verileri.iloc[i+1]['tarih_saat'] -
                           plaka_verileri.iloc[i]['tarih_saat']).total_seconds() / 3600
                    if sure < 24:  # 24 saatten az ise geçerli park süresi
                        park_sureleri.append(sure)

        self.udp = np.mean(park_sureleri) if park_sureleri else 2.5  # varsayılan 2.5 saat

        # 5. Alan Kullanılabilirlik Endeksi (SAI)
        self.sai = 100 - self.odo

        # 6. Yoğunluk Seviyesi (CL)
        gunluk_giris_cikis = self.df.groupby('tarih').size()
        self.cl = gunluk_giris_cikis.mean() / self.kapasite

        # 7. Devir Hızı (TR)
        benzersiz_araclar = self.df.groupby('tarih')['plaka'].nunique()
        self.tr = benzersiz_araclar.mean() / self.kapasite

        # Ek analizler
        self.gunluk_dagilim = self.df['gun_tr'].value_counts()
        self.saatlik_dagilim = self.df['saat'].value_counts().sort_index()
        self.saat_dilimi_dagilim = self.df['saat_dilimi'].value_counts()

        # ML performans metriklerini hesapla
        self.ml_metrikleri_hesapla()

    def ml_metrikleri_hesapla(self):
        """Makine öğrenmesi performans metriklerini hesapla"""
        try:
            # Yoğunluk tahmini için özellikler oluştur
            # Saat, gün, ay bilgilerini kullanarak yoğunluk tahmini yapacağız

            # Saatlik yoğunluk hesapla (hedef değişken)
            saatlik_yogunluk = self.df.groupby(['tarih', 'saat']).size().reset_index()
            saatlik_yogunluk.columns = ['tarih', 'saat', 'arac_sayisi']

            # Yoğunluk seviyelerini kategorize et (düşük, orta, yüksek)
            yogunluk_esik_dusuk = saatlik_yogunluk['arac_sayisi'].quantile(0.33)
            yogunluk_esik_yuksek = saatlik_yogunluk['arac_sayisi'].quantile(0.67)

            def yogunluk_kategorisi(arac_sayisi):
                if arac_sayisi <= yogunluk_esik_dusuk:
                    return 0  # Düşük
                elif arac_sayisi <= yogunluk_esik_yuksek:
                    return 1  # Orta
                else:
                    return 2  # Yüksek

            saatlik_yogunluk['yogunluk_kategori'] = saatlik_yogunluk['arac_sayisi'].apply(yogunluk_kategorisi)

            # Özellikler oluştur
            saatlik_yogunluk['tarih_dt'] = pd.to_datetime(saatlik_yogunluk['tarih'])
            saatlik_yogunluk['hafta_gunu'] = saatlik_yogunluk['tarih_dt'].dt.dayofweek
            saatlik_yogunluk['ay'] = saatlik_yogunluk['tarih_dt'].dt.month
            saatlik_yogunluk['gun'] = saatlik_yogunluk['tarih_dt'].dt.day

            # Özellik matrisi ve hedef değişken
            X = saatlik_yogunluk[['saat', 'hafta_gunu', 'ay', 'gun']]
            y = saatlik_yogunluk['yogunluk_kategori']

            if len(X) > 10:  # Yeterli veri varsa
                # Veriyi böl
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

                # Model eğit
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)

                # Tahmin yap
                y_pred = model.predict(X_test)

                # Performans metriklerini hesapla
                self.accuracy = accuracy_score(y_test, y_pred)
                self.precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                self.recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                self.f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                self.conf_matrix = confusion_matrix(y_test, y_pred)

                # Özellik önem derecesi
                self.feature_importance = model.feature_importances_
                self.feature_names = ['Saat', 'Hafta Günü', 'Ay', 'Gün']

            else:
                # Yeterli veri yoksa varsayılan değerler
                self.accuracy = 0.0
                self.precision = 0.0
                self.recall = 0.0
                self.f1 = 0.0
                self.conf_matrix = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
                self.feature_importance = np.array([0.0, 0.0, 0.0, 0.0])
                self.feature_names = ['Saat', 'Hafta Günü', 'Ay', 'Gün']

        except Exception as e:
            print(f"ML metrikleri hesaplanırken hata: {e}")
            # Hata durumunda varsayılan değerler
            self.accuracy = 0.0
            self.precision = 0.0
            self.recall = 0.0
            self.f1 = 0.0
            self.conf_matrix = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
            self.feature_importance = np.array([0.0, 0.0, 0.0, 0.0])
            self.feature_names = ['Saat', 'Hafta Günü', 'Ay', 'Gün']

    def metrikleri_dosyaya_yaz(self):
        """Tüm metrikleri ve ML performans sonuçlarını dosyaya yaz"""
        try:
            dosya_adi = f"otopark_analiz_raporu_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

            with open(dosya_adi, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("OTOPARK ANALİZ VE YÖNETİM SİSTEMİ - DETAYLI RAPOR\n")
                f.write("=" * 80 + "\n")
                f.write(f"Rapor Tarihi: {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}\n")
                f.write(f"Analiz Edilen Veri Sayısı: {len(self.df)} kayıt\n")
                f.write(f"Analiz Edilen Tarih Aralığı: {self.df['tarih'].min()} - {self.df['tarih'].max()}\n")
                f.write(f"Otopark Kapasitesi: {self.kapasite} araç\n\n")

                # OTOPARK METRİKLERİ
                f.write("1. OTOPARK PERFORMANS METRİKLERİ\n")
                f.write("-" * 50 + "\n")
                f.write(f"Otopark Doluluk Oranı (ODO): {self.odo:.2f}%\n")
                f.write("   → Otoparkın ne kadarının dolu olduğunu gösterir\n\n")

                f.write(f"Alan Kullanılabilirlik Endeksi (SAI): {self.sai:.2f}%\n")
                f.write("   → Anlık kullanılabilir alan oranını gösterir\n\n")

                f.write(f"Zirve Zaman Faktörü (PTF): {self.ptf*100:.2f}%\n")
                f.write("   → En yoğun saatlerin günlük ortalamaya oranı\n\n")

                f.write(f"Kullanıcı Süre Deseni (UDP): {self.udp:.2f} saat\n")
                f.write("   → Araçların ortalama park kalma süresi\n\n")

                f.write(f"Yoğunluk Seviyesi (CL): {self.cl*100:.2f}%\n")
                f.write("   → Giriş-çıkış yoğunluğu oranı\n\n")

                f.write(f"Devir Hızı (TR): {self.tr*100:.2f}%\n")
                f.write("   → Park yerlerinin günlük kullanım oranı\n\n")

                f.write(f"Araç Yoğunluk Seviyesi (VD): {self.vd:.4f} araç/m²\n")
                f.write("   → Birim alan başına düşen araç sayısı\n\n")

                # İSTATİSTİKSEL BİLGİLER
                f.write("2. İSTATİSTİKSEL BİLGİLER\n")
                f.write("-" * 50 + "\n")
                f.write(f"Toplam Giriş Sayısı: {len(self.df)}\n")
                f.write(f"Benzersiz Araç Sayısı: {self.df['plaka'].nunique()}\n")
                f.write(f"Analiz Edilen Gün Sayısı: {self.df['tarih'].nunique()}\n")
                f.write(f"En Yoğun Gün: {self.gunluk_dagilim.index[0]} ({self.gunluk_dagilim.iloc[0]} giriş)\n")
                f.write(f"En Yoğun Saat: {self.saatlik_dagilim.index[0]}:00 ({self.saatlik_dagilim.iloc[0]} giriş)\n\n")

                # GÜNLÜK DAĞILIM
                f.write("3. GÜNLÜK KULLANIM DAĞILIMI\n")
                f.write("-" * 50 + "\n")
                for gun, sayi in self.gunluk_dagilim.items():
                    oran = (sayi / len(self.df)) * 100
                    f.write(f"{gun}: {sayi} giriş ({oran:.1f}%)\n")
                f.write("\n")

                # SAATLİK DAĞILIM (EN YOĞUN 10 SAAT)
                f.write("4. EN YOĞUN 10 SAATLİK DİLİM\n")
                f.write("-" * 50 + "\n")
                en_yogun_saatler = self.saatlik_dagilim.nlargest(10)
                for saat, sayi in en_yogun_saatler.items():
                    oran = (sayi / len(self.df)) * 100
                    f.write(f"{saat:02d}:00 - {saat+1:02d}:00: {sayi} giriş ({oran:.1f}%)\n")
                f.write("\n")

                # SAAT DİLİMİ DAĞILIMI
                f.write("5. SAAT DİLİMİ DAĞILIMI\n")
                f.write("-" * 50 + "\n")
                for dilim, sayi in self.saat_dilimi_dagilim.items():
                    oran = (sayi / len(self.df)) * 100
                    f.write(f"{dilim}: {sayi} giriş ({oran:.1f}%)\n")
                f.write("\n")

                # MAKİNE ÖĞRENMESİ PERFORMANS METRİKLERİ
                f.write("6. MAKİNE ÖĞRENMESİ PERFORMANS METRİKLERİ\n")
                f.write("-" * 50 + "\n")
                f.write("Model: Random Forest Classifier (Yoğunluk Tahmini)\n")
                f.write("Hedef: Otopark yoğunluk seviyesi tahmini (Düşük/Orta/Yüksek)\n\n")

                f.write(f"Doğruluk (Accuracy): {self.accuracy:.4f} ({self.accuracy*100:.2f}%)\n")
                f.write("   → Doğru tahmin edilen örneklerin toplam örneklere oranı\n\n")

                f.write(f"Kesinlik (Precision): {self.precision:.4f} ({self.precision*100:.2f}%)\n")
                f.write("   → Pozitif tahmin edilen örneklerin gerçekten pozitif olanların oranı\n\n")

                f.write(f"Duyarlılık (Recall): {self.recall:.4f} ({self.recall*100:.2f}%)\n")
                f.write("   → Gerçek pozitif örneklerin doğru tahmin edilme oranı\n\n")

                f.write(f"F1 Skoru: {self.f1:.4f} ({self.f1*100:.2f}%)\n")
                f.write("   → Precision ve Recall'un harmonik ortalaması\n\n")

                # KARIŞIKLIK MATRİSİ
                f.write("7. KARIŞIKLIK MATRİSİ (CONFUSION MATRIX)\n")
                f.write("-" * 50 + "\n")
                f.write("Satırlar: Gerçek Değerler, Sütunlar: Tahmin Edilen Değerler\n")
                f.write("0: Düşük Yoğunluk, 1: Orta Yoğunluk, 2: Yüksek Yoğunluk\n\n")
                f.write("     Tahmin\n")
                f.write("     0   1   2\n")
                f.write("   +-----------\n")
                for i, row in enumerate(self.conf_matrix):
                    f.write(f"{i}  | {row[0]:2d}  {row[1]:2d}  {row[2]:2d}\n")
                f.write("\n")

                # ÖZELLİK ÖNEMİ
                f.write("8. ÖZELLİK ÖNEMİ (FEATURE IMPORTANCE)\n")
                f.write("-" * 50 + "\n")
                for i, (feature, importance) in enumerate(zip(self.feature_names, self.feature_importance)):
                    f.write(f"{feature}: {importance:.4f} ({importance*100:.2f}%)\n")
                f.write("\n")

                # SONUÇ VE ÖNERİLER
                f.write("9. SONUÇ VE ÖNERİLER\n")
                f.write("-" * 50 + "\n")

                # Doluluk durumu değerlendirmesi
                if self.odo < 30:
                    f.write("• Otopark doluluk oranı düşük. Pazarlama stratejileri geliştirilebilir.\n")
                elif self.odo < 70:
                    f.write("• Otopark doluluk oranı optimal seviyede.\n")
                else:
                    f.write("• Otopark doluluk oranı yüksek. Kapasite artırımı düşünülebilir.\n")

                # PTF değerlendirmesi
                if self.ptf < 0.2:
                    f.write("• Zirve zaman faktörü düşük. Yoğunluk dengeli dağılmış.\n")
                elif self.ptf < 0.5:
                    f.write("• Zirve zaman faktörü orta seviyede. Belirli saatlerde yoğunlaşma var.\n")
                else:
                    f.write("• Zirve zaman faktörü yüksek. Belirli saatlerde aşırı yoğunluk.\n")

                # ML model performansı
                if self.accuracy > 0.8:
                    f.write("• ML model performansı mükemmel. Yoğunluk tahminleri güvenilir.\n")
                elif self.accuracy > 0.6:
                    f.write("• ML model performansı iyi. Yoğunluk tahminleri kullanılabilir.\n")
                else:
                    f.write("• ML model performansı geliştirilmeli. Daha fazla veri gerekli.\n")

                f.write("\n" + "=" * 80 + "\n")
                f.write("Rapor sonu - Otopark Analiz ve Yönetim Sistemi\n")
                f.write("=" * 80 + "\n")

            print(f"Rapor başarıyla kaydedildi: {dosya_adi}")
            messagebox.showinfo("Başarılı", f"Detaylı analiz raporu kaydedildi:\n{dosya_adi}")

        except Exception as e:
            print(f"Rapor kaydedilirken hata: {e}")
            messagebox.showerror("Hata", f"Rapor kaydedilirken hata oluştu: {str(e)}")

    def ui_olustur(self):
        """Ana kullanıcı arayüzünü oluştur"""

        # Ana başlık
        baslik_frame = tk.Frame(self.root, bg='#2c3e50', height=80)
        baslik_frame.pack(fill='x', padx=10, pady=5)
        baslik_frame.pack_propagate(False)

        baslik_label = tk.Label(baslik_frame,
                               text="🚗 OTOPARK ANALİZ VE YÖNETİM SİSTEMİ 🚗",
                               font=('Arial', 18, 'bold'),
                               fg='white', bg='#2c3e50')
        baslik_label.pack(side='left', expand=True)

        # Rapor oluşturma butonu
        rapor_button = tk.Button(baslik_frame,
                                text="📄 Detaylı Rapor Oluştur",
                                font=('Arial', 12, 'bold'),
                                bg='#27ae60', fg='white',
                                command=self.metrikleri_dosyaya_yaz,
                                relief='raised', bd=3)
        rapor_button.pack(side='right', padx=20, pady=15)

        # Notebook (sekmeli yapı)
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=5)

        # Sekmeler
        self.ozet_sekmesi()
        self.metrik_sekmesi()
        self.grafik_sekmesi()
        self.isi_haritasi_sekmesi()
        self.trend_sekmesi()

    def ozet_sekmesi(self):
        """Özet bilgiler sekmesi"""
        ozet_frame = ttk.Frame(self.notebook)
        self.notebook.add(ozet_frame, text="📊 Genel Özet")

        # Sol panel - Metrik kartları
        sol_frame = tk.Frame(ozet_frame, bg='#ecf0f1')
        sol_frame.pack(side='left', fill='both', expand=True, padx=5, pady=5)

        # Metrik kartları
        metrikler = [
            ("Otopark Doluluk Oranı", f"{self.odo:.1f}%", "#e74c3c", "Otoparkın ne kadarının dolu olduğu"),
            ("Alan Kullanılabilirlik", f"{self.sai:.1f}%", "#27ae60", "Kullanılabilir boş alan oranı"),
            ("Zirve Zaman Faktörü", f"{self.ptf*100:.1f}%", "#f39c12", "En yoğun saatlerin günlük ortalamaya oranı"),
            ("Ortalama Park Süresi", f"{self.udp:.1f} saat", "#3498db", "Araçların ortalama park kalma süresi"),
            ("Devir Hızı", f"{self.tr*100:.1f}%", "#9b59b6", "Park yerlerinin günlük kullanım oranı"),
            ("Yoğunluk Seviyesi", f"{self.cl*100:.1f}%", "#e67e22", "Giriş-çıkış yoğunluğu oranı")
        ]

        for i, (baslik, deger, renk, aciklama) in enumerate(metrikler):
            kart = tk.Frame(sol_frame, bg=renk, relief='raised', bd=2)
            kart.pack(fill='x', padx=10, pady=2)

            tk.Label(kart, text=baslik, font=('Arial', 10, 'bold'),
                    fg='white', bg=renk).pack(pady=2)
            tk.Label(kart, text=deger, font=('Arial', 14, 'bold'),
                    fg='white', bg=renk).pack(pady=2)
            tk.Label(kart, text=aciklama, font=('Arial', 8),
                    fg='white', bg=renk, wraplength=200).pack(pady=2)

        # Sağ panel - Hızlı istatistikler
        sag_frame = tk.Frame(ozet_frame, bg='#ecf0f1')
        sag_frame.pack(side='right', fill='both', expand=True, padx=5, pady=5)

        # İstatistik tablosu
        stats_frame = tk.LabelFrame(sag_frame, text="Hızlı İstatistikler",
                                   font=('Arial', 12, 'bold'))
        stats_frame.pack(fill='both', expand=True, padx=10, pady=10)

        istatistikler = [
            ("Toplam Giriş Sayısı", len(self.df)),
            ("Yeni Araç Sayısı", self.df['plaka'].nunique()),
            ("Analiz Edilen Gün Sayısı", self.df['tarih'].nunique()),
            ("En Yoğun Gün", self.gunluk_dagilim.index[0]),
            ("En Yoğun Saat", f"{self.saatlik_dagilim.index[0]}:00"),
            ("Otopark Kapasitesi", self.kapasite)
        ]

        for i, (baslik, deger) in enumerate(istatistikler):
            row_frame = tk.Frame(stats_frame)
            row_frame.pack(fill='x', padx=10, pady=2)

            tk.Label(row_frame, text=f"{baslik}:", font=('Arial', 10),
                    anchor='w').pack(side='left')
            tk.Label(row_frame, text=str(deger), font=('Arial', 10, 'bold'),
                    anchor='e').pack(side='right')

    def metrik_sekmesi(self):
        """Detaylı metrikler sekmesi"""
        metrik_frame = ttk.Frame(self.notebook)
        self.notebook.add(metrik_frame, text="📈 Detaylı Metrikler")

        # Ana container - scroll olmadan
        main_container = tk.Frame(metrik_frame, bg='#ecf0f1')
        main_container.pack(fill='both', expand=True, padx=10, pady=10)

        # Sol ve sağ paneller oluştur
        sol_panel = tk.Frame(main_container, bg='#ecf0f1')
        sol_panel.pack(side='left', fill='both', expand=True, padx=5)

        sag_panel = tk.Frame(main_container, bg='#ecf0f1')
        sag_panel.pack(side='right', fill='both', expand=True, padx=5)

        # Metrik açıklamaları ve formülleri - kompakt format
        metrik_bilgileri = [
            {
                "baslik": "Otopark Doluluk Oranı (ODO)",
                "deger": f"{self.odo:.1f}%",
                "aciklama": "Otoparkın ne kadarının kullanıldığını gösterir"
            },
            {
                "baslik": "Alan Kullanılabilirlik Endeksi (SAI)",
                "deger": f"{self.sai:.1f}%",
                "aciklama": "Anlık kullanılabilir alan oranını gösterir"
            },
            {
                "baslik": "Zirve Zaman Faktörü (PTF)",
                "deger": f"{self.ptf*100:.1f}%",
                "aciklama": "En yoğun saatlerin günlük ortalamaya oranı"
            },
            {
                "baslik": "Kullanıcı Süre Deseni (UDP)",
                "deger": f"{self.udp:.1f} saat",
                "aciklama": "Ortalama park süresini gösterir"
            },
            {
                "baslik": "Yoğunluk Seviyesi (CL)",
                "deger": f"{self.cl*100:.1f}%",
                "aciklama": "Giriş-çıkış yoğunluğunu gösterir"
            },
            {
                "baslik": "Devir Hızı (TR)",
                "deger": f"{self.tr*100:.1f}%",
                "aciklama": "Park yerlerinin günlük kullanım oranı"
            },
            {
                "baslik": "Araç Yoğunluk Seviyesi (VD)",
                "deger": f"{self.vd:.3f} araç/m²",
                "aciklama": "Birim alan başına düşen araç sayısı"
            }
        ]

        # Sol panele ilk 4 metrik
        for i, metrik in enumerate(metrik_bilgileri[:4]):
            panel = sol_panel

            # Metrik kartı - kompakt
            kart_frame = tk.LabelFrame(panel, text=metrik["baslik"],
                                      font=('Arial', 10, 'bold'), fg='#2c3e50')
            kart_frame.pack(fill='x', padx=5, pady=5)

            # Değer
            deger_frame = tk.Frame(kart_frame, bg='#3498db')
            deger_frame.pack(fill='x', padx=3, pady=3)
            tk.Label(deger_frame, text=metrik['deger'],
                    font=('Arial', 12, 'bold'), fg='white', bg='#3498db').pack(pady=3)

            # Açıklama
            tk.Label(kart_frame, text=metrik['aciklama'],
                    font=('Arial', 9), wraplength=300, justify='center', fg='#7f8c8d').pack(padx=3, pady=3)

        # Sağ panele kalan 3 metrik
        for i, metrik in enumerate(metrik_bilgileri[4:]):
            panel = sag_panel

            # Metrik kartı - kompakt
            kart_frame = tk.LabelFrame(panel, text=metrik["baslik"],
                                      font=('Arial', 10, 'bold'), fg='#2c3e50')
            kart_frame.pack(fill='x', padx=5, pady=5)

            # Değer
            deger_frame = tk.Frame(kart_frame, bg='#3498db')
            deger_frame.pack(fill='x', padx=3, pady=3)
            tk.Label(deger_frame, text=metrik['deger'],
                    font=('Arial', 12, 'bold'), fg='white', bg='#3498db').pack(pady=3)

            # Açıklama
            tk.Label(kart_frame, text=metrik['aciklama'],
                    font=('Arial', 9), wraplength=300, justify='center', fg='#7f8c8d').pack(padx=3, pady=3)

    def grafik_sekmesi(self):
        """Grafik görselleştirmeleri sekmesi"""
        grafik_frame = ttk.Frame(self.notebook)
        self.notebook.add(grafik_frame, text="📊 Grafikler")

        # Grafik notebook
        grafik_notebook = ttk.Notebook(grafik_frame)
        grafik_notebook.pack(fill='both', expand=True, padx=5, pady=5)

        # Günlük dağılım grafiği
        self.gunluk_grafik(grafik_notebook)

        # Saatlik dağılım grafiği
        self.saatlik_grafik(grafik_notebook)

        # Saat dilimi grafiği
        self.saat_dilimi_grafik(grafik_notebook)

    def gunluk_grafik(self, parent):
        """Günlük kullanım dağılım grafiği"""
        frame = ttk.Frame(parent)
        parent.add(frame, text="Günlük Dağılım")

        fig = Figure(figsize=(10, 6), dpi=100)
        ax = fig.add_subplot(111)

        # Pasta grafiği
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8']
        _, _, autotexts = ax.pie(self.gunluk_dagilim.values,
                                labels=self.gunluk_dagilim.index,
                                autopct='%1.1f%%',
                                colors=colors,
                                startangle=90)

        ax.set_title('Günlere Göre Otopark Kullanım Dağılımı', fontsize=14, fontweight='bold')

        # Güzelleştirme
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')

        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

    def saatlik_grafik(self, parent):
        """Saatlik kullanım dağılım grafiği"""
        frame = ttk.Frame(parent)
        parent.add(frame, text="Saatlik Dağılım")

        fig = Figure(figsize=(12, 6), dpi=100)
        ax = fig.add_subplot(111)

        # Bar grafiği
        bars = ax.bar(self.saatlik_dagilim.index, self.saatlik_dagilim.values,
                     color='#3498db', alpha=0.7, edgecolor='#2c3e50', linewidth=1)

        ax.set_title('Saatlere Göre Otopark Kullanım Dağılımı', fontsize=14, fontweight='bold')
        ax.set_xlabel('Saat', fontsize=12)
        ax.set_ylabel('Araç Sayısı', fontsize=12)
        ax.grid(True, alpha=0.3)

        # Değerleri barların üstüne yaz
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{int(height)}', ha='center', va='bottom', fontweight='bold')

        ax.set_xticks(range(24))
        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

    def saat_dilimi_grafik(self, parent):
        """Saat dilimi dağılım grafiği"""
        frame = ttk.Frame(parent)
        parent.add(frame, text="Saat Dilimleri")

        fig = Figure(figsize=(10, 6), dpi=100)
        ax = fig.add_subplot(111)

        # Horizontal bar grafiği
        colors = ['#2c3e50', '#e74c3c', '#f39c12', '#27ae60']
        bars = ax.barh(self.saat_dilimi_dagilim.index, self.saat_dilimi_dagilim.values,
                      color=colors, alpha=0.8)

        ax.set_title('Saat Dilimlerine Göre Otopark Kullanımı', fontsize=14, fontweight='bold')
        ax.set_xlabel('Araç Sayısı', fontsize=12)
        ax.grid(True, alpha=0.3, axis='x')

        # Değerleri barların yanına yaz
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + 1, bar.get_y() + bar.get_height()/2.,
                   f'{int(width)}', ha='left', va='center', fontweight='bold')

        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

    def isi_haritasi_sekmesi(self):
        """Isı haritası görselleştirmeleri sekmesi"""
        isi_frame = ttk.Frame(self.notebook)
        self.notebook.add(isi_frame, text="🔥 Isı Haritaları")

        # Isı haritası notebook
        isi_notebook = ttk.Notebook(isi_frame)
        isi_notebook.pack(fill='both', expand=True, padx=5, pady=5)

        # Günlük-Saatlik ısı haritası
        self.gunluk_saatlik_isi_haritasi(isi_notebook)

        # Aylık ısı haritası
        self.aylik_isi_haritasi(isi_notebook)

    def gunluk_saatlik_isi_haritasi(self, parent):
        """Günlük ve saatlik kullanım ısı haritası"""
        frame = ttk.Frame(parent)
        parent.add(frame, text="Günlük-Saatlik Isı Haritası")

        fig = Figure(figsize=(14, 8), dpi=100)
        ax = fig.add_subplot(111)

        # Pivot tablo oluştur
        pivot_data = self.df.pivot_table(
            values='plaka',
            index='gun_tr',
            columns='saat',
            aggfunc='count',
            fill_value=0
        )

        # Günleri doğru sırayla düzenle
        gun_sirasi = ['Pazartesi', 'Salı', 'Çarşamba', 'Perşembe', 'Cuma', 'Cumartesi', 'Pazar']
        pivot_data = pivot_data.reindex(gun_sirasi)

        # Isı haritası oluştur
        im = ax.imshow(pivot_data.values, cmap='YlOrRd', aspect='auto', interpolation='nearest')

        # Eksen etiketleri
        ax.set_xticks(range(len(pivot_data.columns)))
        ax.set_xticklabels([f"{i}:00" for i in pivot_data.columns])
        ax.set_yticks(range(len(pivot_data.index)))
        ax.set_yticklabels(pivot_data.index)

        # Başlık ve etiketler
        ax.set_title('Günlük ve Saatlik Otopark Kullanım Yoğunluğu', fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Saat', fontsize=12)
        ax.set_ylabel('Gün', fontsize=12)

        # Colorbar
        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Araç Sayısı', fontsize=12)

        # Değerleri hücrelere yaz
        for i in range(len(pivot_data.index)):
            for j in range(len(pivot_data.columns)):
                value = pivot_data.iloc[i, j]
                if value > 0:
                    text_color = 'white' if value > pivot_data.values.max() * 0.6 else 'black'
                    ax.text(j, i, str(int(value)), ha='center', va='center',
                           color=text_color, fontweight='bold', fontsize=8)

        # X ekseni etiketlerini döndür
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

    def aylik_isi_haritasi(self, parent):
        """Aylık kullanım ısı haritası"""
        frame = ttk.Frame(parent)
        parent.add(frame, text="Aylık Isı Haritası")

        fig = Figure(figsize=(12, 8), dpi=100)
        ax = fig.add_subplot(111)

        # Aylık ve günlük pivot tablo
        self.df['gun_sayi'] = self.df['tarih_saat'].dt.day
        pivot_monthly = self.df.pivot_table(
            values='plaka',
            index='ay',
            columns='gun_sayi',
            aggfunc='count',
            fill_value=0
        )

        # Isı haritası
        im = ax.imshow(pivot_monthly.values, cmap='Blues', aspect='auto', interpolation='nearest')

        # Eksen ayarları
        ax.set_xticks(range(len(pivot_monthly.columns)))
        ax.set_xticklabels(pivot_monthly.columns)
        ax.set_yticks(range(len(pivot_monthly.index)))

        ay_isimleri = ['Ocak', 'Şubat', 'Mart', 'Nisan', 'Mayıs', 'Haziran',
                      'Temmuz', 'Ağustos', 'Eylül', 'Ekim', 'Kasım', 'Aralık']
        ax.set_yticklabels([ay_isimleri[i-1] for i in pivot_monthly.index])

        ax.set_title('Aylık Otopark Kullanım Yoğunluğu', fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Ayın Günü', fontsize=12)
        ax.set_ylabel('Ay', fontsize=12)

        # Colorbar
        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Araç Sayısı', fontsize=12)

        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

    def trend_sekmesi(self):
        """Trend analizi sekmesi"""
        trend_frame = ttk.Frame(self.notebook)
        self.notebook.add(trend_frame, text="📈 Trend Analizi")

        # Trend notebook
        trend_notebook = ttk.Notebook(trend_frame)
        trend_notebook.pack(fill='both', expand=True, padx=5, pady=5)

        # Günlük trend
        self.gunluk_trend(trend_notebook)

        # Haftalık trend
        self.haftalik_trend(trend_notebook)

        # Metrik karşılaştırması
        self.metrik_karsilastirma(trend_notebook)

    def gunluk_trend(self, parent):
        """Günlük kullanım trendi"""
        frame = ttk.Frame(parent)
        parent.add(frame, text="Günlük Trend")

        fig = Figure(figsize=(14, 8), dpi=100)
        ax = fig.add_subplot(111)

        # Günlük araç sayısı
        gunluk_arac = self.df.groupby('tarih').size().reset_index()
        gunluk_arac.columns = ['tarih', 'arac_sayisi']

        # Çizgi grafiği
        ax.plot(gunluk_arac['tarih'], gunluk_arac['arac_sayisi'],
               marker='o', linewidth=2, markersize=4, color='#3498db')

        # Trend çizgisi (moving average)
        if len(gunluk_arac) > 7:
            gunluk_arac['trend'] = gunluk_arac['arac_sayisi'].rolling(window=7, center=True).mean()
            ax.plot(gunluk_arac['tarih'], gunluk_arac['trend'],
                   linewidth=3, color='#e74c3c', alpha=0.8, label='7 Günlük Ortalama')

        ax.set_title('Günlük Otopark Kullanım Trendi', fontsize=14, fontweight='bold')
        ax.set_xlabel('Tarih', fontsize=12)
        ax.set_ylabel('Araç Sayısı', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend()

        # X ekseni tarih formatı
        fig.autofmt_xdate()

        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

    def haftalik_trend(self, parent):
        """Haftalık kullanım trendi"""
        frame = ttk.Frame(parent)
        parent.add(frame, text="Haftalık Trend")

        fig = Figure(figsize=(12, 8), dpi=100)
        ax = fig.add_subplot(111)

        # Haftalık veri
        self.df['hafta'] = self.df['tarih_saat'].dt.isocalendar().week
        haftalik_arac = self.df.groupby(['hafta', 'gun_tr']).size().unstack(fill_value=0)

        # Çoklu çizgi grafiği
        colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c', '#34495e']
        for i, gun in enumerate(haftalik_arac.columns):
            ax.plot(haftalik_arac.index, haftalik_arac[gun],
                   marker='o', linewidth=2, label=gun, color=colors[i % len(colors)])

        ax.set_title('Haftalık Günlere Göre Otopark Kullanım Trendi', fontsize=14, fontweight='bold')
        ax.set_xlabel('Hafta', fontsize=12)
        ax.set_ylabel('Araç Sayısı', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

    def metrik_karsilastirma(self, parent):
        """Metrik karşılaştırma grafiği"""
        frame = ttk.Frame(parent)
        parent.add(frame, text="Metrik Karşılaştırması")

        fig = Figure(figsize=(12, 8), dpi=100)
        ax = fig.add_subplot(111)

        # Metrik verileri (normalize edilmiş)
        metrikler = {
            'Doluluk Oranı': self.odo / 100,
            'Kullanılabilirlik': self.sai / 100,
            'Zirve Faktörü': min(self.ptf, 1),      # PTF zaten yüzde formatında
            'Devir Hızı': min(self.tr, 1),          # TR zaten yüzde formatında
            'Yoğunluk': min(self.cl, 1)             # CL zaten yüzde formatında
        }

        # Radar chart benzeri bar chart
        metrik_isimleri = list(metrikler.keys())
        metrik_degerleri = list(metrikler.values())

        bars = ax.bar(metrik_isimleri, metrik_degerleri,
                     color=['#e74c3c', '#27ae60', '#f39c12', '#9b59b6', '#e67e22'],
                     alpha=0.8, edgecolor='#2c3e50', linewidth=2)

        # Değerleri barların üstüne yaz
        for bar, deger in zip(bars, metrik_degerleri):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{deger:.2f}', ha='center', va='bottom', fontweight='bold')

        ax.set_title('Otopark Performans Metrikleri Karşılaştırması', fontsize=14, fontweight='bold')
        ax.set_ylabel('Normalize Değer (0-1)', fontsize=12)
        ax.set_ylim(0, 1.2)
        ax.grid(True, alpha=0.3, axis='y')

        # X ekseni etiketlerini döndür
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

    def run(self):
        """Uygulamayı başlat"""
        self.root.mainloop()

# Uygulamayı başlat
if __name__ == "__main__":
    app = OtoparkAnalizAraci()
    app.run()