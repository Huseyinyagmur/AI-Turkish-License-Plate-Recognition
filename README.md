# AI-Based Turkish License Plate Recognition System

🚗 Türk araç plakalarının görüntü ve videolarda tespit edilmesi, metin olarak okunması, doğrulanması ve CSV'ye kaydedilmesi için geliştirilen **YOLO11s tabanlı** bilgisayarlı görü projesi.

> **Proje durumu:** Plaka tespiti, ByteTrack tabanlı takip, crop çıkarımı, PaddleOCR ile metin okuma, Türk plaka doğrulama ve CSV kayıt hattı tamamlanmıştır.

## 1. Proje Özeti

Bu projenin amacı, araç görüntülerindeki Türk plakalarını tek bir sınıf olarak tespit eden, güçlü ve genişletilebilir bir altyapı oluşturmaktır. Model; farklı ışık koşulları, bakış açıları ve araç türlerinde plaka bölgesini bulmak üzere özel veri seti ile eğitilmiştir.

🎯 Tespit edilen plaka kırpımları PaddleOCR katmanına aktarılır; metin temizlenir, Türk plaka formatına göre doğrulanır ve CSV kaydı üretilir. ByteTrack ile takip tamamlanmıştır; dashboard ve otomatik raporlama sonraki aşamalardır.

## 2. Özellikler

- YOLO11s ile Türk plakası nesne tespiti
- Tek sınıflı (`license_plate`) özel eğitim altyapısı
- Train / validation / test veri hazırlama hattı
- YOLO etiket formatı ve koordinat aralığı doğrulaması
- Eksik veya bozuk etiketleri ayıklayan veri seti split betiği
- Eğitilen model için değerlendirme ve metrik takibi
- Görüntüler üzerinde hızlı çıkarım desteği
- PaddleOCR ile plaka metni okuma
- Türkçe karakter normalizasyonu, metin temizleme ve regex ile plaka doğrulama
- Toplu OCR işleme ve CSV export
- ByteTrack multi-object tracking ve Track ID assignment
- Track-bazlı OCR voting ile en güvenilir plaka sonucunun seçilmesi
- Vehicle Re-Identification Prevention: track-bazlı oylama ile tekrar kayıtların azaltılması
- Automated PDF Reporting ile pipeline analiz sonuçlarının profesyonel raporlanması
- Takip ve raporlama için modüler mimari temeli

## 3. Kullanılan Teknolojiler

| Teknoloji | Kullanım amacı |
| --- | --- |
| 🧠 Python | Uygulama ve veri işleme dili |
| 🧠 Ultralytics YOLO11s | Plaka nesne tespiti ve model eğitimi |
| PyTorch | Derin öğrenme çalışma zamanı |
| OpenCV | Görüntü okuma, işleme ve görselleştirme |
| YAML | YOLO veri seti yapılandırması |
| PaddleOCR | Tespit edilen plaka crop'larının metin olarak okunması |
| CSV | OCR sonuçlarının kayıt altına alınması |
| ByteTrack | Video içindeki plaka nesnelerinin Track ID ile takibi |

## 4. Veri Seti

Ham veri seti aşağıdaki yapıda tutulur:

```text
datasets/raw/turkish_plate_dataset/
├── images/
└── labels/
```

Her görsel için aynı ada sahip bir `.txt` etiket dosyası bulunur. Etiketler YOLO biçimindedir:

```text
class_id x_center y_center width height
```

Koordinatlar görüntü boyutuna göre normalize edilir ve `0–1` aralığında olmalıdır. Projede tek sınıf kullanılır:

```yaml
0: license_plate
```

`src/utils/split_dataset.py` betiği geçerli görsel-etiket çiftlerini denetler, karıştırır ve veriyi `%70 train`, `%20 validation`, `%10 test` oranıyla YOLO düzeninde hazırlar. Eksik veya bozuk etiketli örnekler işleme alınmaz.

## 5. Proje Mimarisi

Güncel işlem hattı:

```text
Video / Image
      ↓
YOLO11s Detection
      ↓
Plate Crop Extraction
      ↓
PaddleOCR
      ↓
Plate Text Cleaning
      ↓
Turkish Plate Validation
      ↓
CSV Logging
```

Tracking modu için işlem hattı:

```text
Video
  ↓
YOLO11s
  ↓
ByteTrack
  ↓
Crop Extraction
  ↓
PaddleOCR
  ↓
Turkish Plate Validation
  ↓
Track Voting
  ↓
Final CSV
  ↓
PDF Report
```

```text
Ham görseller + YOLO etiketleri
            │
            ▼
Veri doğrulama ve train/val/test ayırma
            │
            ▼
YOLO11s özel model eğitimi
            │
            ▼
Plaka tespiti ve performans değerlendirmesi
            │
            ├──► PaddleOCR ile plaka okuma ve CSV kaydı
            ├──► ByteTrack ile araç/plaka takibi
            └──► Dashboard / raporlama (planlanan)
```

⚡ Modüler yapı sayesinde tespit, OCR, takip, kayıt ve arayüz bileşenleri birbirinden bağımsız geliştirilebilir.

## 6. Klasör Yapısı

```text
AI-Turkish-License-Plate-Recognition/
├── datasets/
│   ├── raw/                         # Ham görseller ve YOLO etiketleri
│   └── processed/                   # Train/val/test olarak hazırlanmış veri
├── models/
│   ├── detection/                   # Tespit modeli çıktıları
│   └── ocr/                         # OCR modelleri için ayrılan alan
├── src/
│   ├── detection/
│   │   └── crop_plates.py           # Tespit ve plaka crop çıkarımı
│   ├── ocr/
│   │   ├── test_ocr.py              # Tek crop OCR testi
│   │   ├── plate_cleaner.py         # Metin temizleme ve doğrulama
│   │   └── batch_ocr.py             # Toplu OCR ve CSV export
│   ├── tracking/                    # Takip modülleri
│   ├── logging/                     # Kayıt modülleri
│   ├── dashboard/                   # Dashboard bileşenleri
│   └── utils/                       # Yardımcı betikler
├── outputs/                         # Tahminler, kırpımlar, loglar ve raporlar
├── notebooks/                       # Deney ve analiz defterleri
├── requirements.txt                 # Python bağımlılıkları
└── README.md
```

## 7. Model Eğitimi

Eğitimden önce veri seti YOLO klasör yapısına dönüştürülür:

```bash
python src/utils/split_dataset.py
```

Bu işlem `datasets/processed/turkish_plate_yolo/data.yaml` dosyasını üretir. Yapılandırma; eğitim, doğrulama ve test klasörlerini tanımlar. Eğitimde YOLO11s başlangıç ağırlıkları kullanılarak Türk plakası sınıfı için fine-tuning uygulanır.

## 8. Eğitim Sonuçları

📊 Özel Türk plaka veri seti üzerinde gerçekleştirilen model değerlendirmesi aşağıdaki sonuçları vermiştir:

| Metrik | Sonuç |
| --- | ---: |
| Precision | **96.6%** |
| Recall | **99.0%** |
| mAP@50 | **99.4%** |
| mAP@50-95 | **88.8%** |

Bu sonuçlar, modelin plaka örneklerini yüksek doğrulukla bulduğunu ve farklı IoU eşiklerinde tutarlı performans gösterdiğini işaret eder.

## 9. Performans Metrikleri Açıklaması

- **Precision (96.6%)**: Modelin plaka olarak işaretlediği bölgelerin ne kadarının gerçekten plaka olduğunu gösterir. Yüksek precision, yanlış pozitiflerin düşük olduğu anlamına gelir.
- **Recall (99.0%)**: Veri setindeki gerçek plakaların ne kadarının model tarafından bulunduğunu gösterir. Bu değer plaka kaçırma oranının oldukça düşük olduğunu belirtir.
- **mAP@50 (99.4%)**: Tahmin ile gerçek kutu arasındaki IoU eşiği `0.50` iken hesaplanan ortalama kesinliktir. Temel tespit başarısını özetler.
- **mAP@50-95 (88.8%)**: `0.50` ile `0.95` arasındaki birden fazla IoU eşiğinin ortalamasıdır. Kutu konumlandırma kalitesini daha sıkı bir ölçütle değerlendirir.

🎯 Bu metrikler veri seti dağılımına bağlıdır; gerçek sahadaki performans kamera açısı, çözünürlük, ışık, hareket bulanıklığı ve plaka görünürlüğüne göre değişebilir.

## 10. Kurulum

Projeyi bilgisayarınıza alın ve sanal ortam oluşturun:

```bash
git clone https://github.com/<kullanici-adi>/AI-Turkish-License-Plate-Recognition.git
cd AI-Turkish-License-Plate-Recognition
python -m venv .venv
```

Sanal ortamı etkinleştirin:

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate
```

Bağımlılıkları yükleyin:

```bash
pip install -r requirements.txt
```

`requirements.txt` henüz oluşturulmadıysa temel çalışma ortamı için aşağıdaki komut kullanılabilir:

```bash
pip install ultralytics opencv-python "paddleocr==2.7.*" pandas reportlab
```

> OCR betikleri PaddleOCR 2.7.x API'si ile uyumludur.

## 11. Eğitim Komutu Örneği

Veri split işlemi tamamlandıktan sonra YOLO11s eğitimi şu şekilde başlatılabilir:

```bash
yolo detect train \
  model=yolo11s.pt \
  data=datasets/processed/turkish_plate_yolo/data.yaml \
  epochs=100 \
  imgsz=640 \
  project=models/detection \
  name=turkish_plate_yolo11s
```

Eğitim parametreleri; epoch sayısı, görüntü boyutu, batch size ve cihaz seçimi kullanıcının donanımına ve veri setine göre güncellenebilir.

## 12. Görüntü veya Video Prediction

Eğitilmiş ağırlıklarla bir görsel veya video kaynağında plaka tespiti yapmak için:

```bash
yolo detect predict \
  model=models/detection/best.pt \
  source=data/test.mp4 \
  conf=0.25 \
  save=True
```

`source` değerine görüntü yolu verildiğinde aynı komut fotoğraf üzerinde de çalışır.

## 13. Plate Crop Extraction

Tespit edilen plakaları görüntü veya videodan crop olarak kaydetmek için:

```bash
python src/detection/crop_plates.py \
  --source data/test.mp4 \
  --model models/detection/best.pt \
  --output outputs/crops \
  --frame-step 10
```

`--frame-step 10` videodaki her onuncu kareyi işler. Tek görüntü kaynaklarında frame skipping uygulanmaz.

## 14. Tek Crop OCR Testi

```bash
python src/ocr/test_ocr.py --image outputs/crops/sample.jpg
```

Komut, bulunan her metin için OCR metnini ve güven skorunu terminale yazar.

## 15. Batch OCR ve CSV Export

`outputs/crops/` içindeki tüm `.jpg`, `.jpeg` ve `.png` crop'larını OCR ile okuyup CSV dosyasına aktarın:

```bash
python src/ocr/batch_ocr.py --input outputs/crops --output outputs/logs/plate_results.csv
```

İsteğe bağlı olarak minimum OCR güven eşiği verilebilir:

```bash
python src/ocr/batch_ocr.py --min-ocr-conf 0.70
```

CSV şu alanları içerir:

```text
file_name, raw_text, cleaned_text, ocr_confidence, is_valid, image_path
```

Metin temizleme aşamasında boşluklar ve özel karakterler kaldırılır; Türkçe karakterler Latin karşılıklarına dönüştürülür. Doğrulama, `01-81` il kodu, 1-3 harf ve 2-4 rakam kuralına göre yapılır. Eşik altındaki OCR sonuçları CSV'ye yazılır, ancak geçersiz olarak işaretlenir.

## 16. Gelecek Çalışmalar

- Duplicate plate filtering ile tekrarlanan plakaların filtrelenmesi
- Tespit, crop, OCR ve kayıt aşamalarını birleştiren full video OCR pipeline
- Track kimlikleri arasında uzun süreli vehicle re-identification
- Sonuçların izlenebileceği dashboard
- Otomatik raporlama ve bildirimler

## 17. ByteTrack ile Video Takibi

Video kaynaklarındaki plakaları Track ID ile izlemek ve her track için OCR oylaması yapmak için:

```bash
python src/main.py --source data/test.mp4 --tracking
```

PDF raporu ile birlikte çalıştırmak için:

```bash
python src/main.py --source data/test.mp4 --tracking --report
```

Tracking modu `outputs/tracking_crops/` altında Track ID içeren crop'lar üretir. Ayrıntılı takip sonuçları `outputs/logs/tracking_results.csv`, track özeti `outputs/logs/tracking_summary.csv` ve oylama sonucu `outputs/logs/final_tracked_plates.csv` dosyasına yazılır. `--report` parametresi bu sonuçlardan `outputs/reports/license_plate_report.pdf` dosyasını üretir.

🚗 Nihai hedef; kamera akışından plakayı tespit eden, okuyan, takip eden ve denetlenebilir kayıtlar üreten uçtan uca bir akıllı plaka tanıma sistemi oluşturmaktır.

## 18. Lisans ve Sorumluluk Reddi

Bu proje eğitim, araştırma ve portföy amaçlı geliştirilmiştir. Kullanımdan önce uygun bir lisans dosyası eklenmeli ve kullanılan veri setlerinin lisans koşulları ayrıca doğrulanmalıdır.

Plaka bilgileri kişisel veri veya kişisel veriyle ilişkilendirilebilir nitelikte olabilir. Sistemi gerçek ortamlarda kullanmadan önce yürürlükteki kişisel verilerin korunması, gizlilik, kamera kullanımı ve yerel mevzuat yükümlülüklerinin değerlendirilmesi kullanıcının sorumluluğundadır. Bu proje hukuka aykırı izleme, takip veya veri toplama amacıyla kullanılmamalıdır.
