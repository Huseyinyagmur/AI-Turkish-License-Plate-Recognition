# AI-Based Turkish License Plate Recognition System

🚗 Türk araç plakalarının görüntüler üzerinde güvenilir biçimde tespit edilmesi için geliştirilen, **YOLO11s tabanlı** profesyonel bir bilgisayarlı görü projesi. Sistem, özel olarak hazırlanan Türk plaka veri seti üzerinde eğitilmiş nesne tespit modelini kullanır. İlerleyen aşamalarda tespit edilen plakaların PaddleOCR ile okunması, takip edilmesi ve kayıt altına alınması hedeflenmektedir.

> **Proje durumu:** Plaka tespit modeli eğitilmiş ve değerlendirilmiştir. OCR, gerçek zamanlı işleme ve kayıt bileşenleri planlanan geliştirme aşamasındadır.

## 1. Proje Özeti

Bu projenin amacı, araç görüntülerindeki Türk plakalarını tek bir sınıf olarak tespit eden, güçlü ve genişletilebilir bir altyapı oluşturmaktır. Model; farklı ışık koşulları, bakış açıları ve araç türlerinde plaka bölgesini bulmak üzere özel veri seti ile eğitilmiştir.

🎯 İlk aşamanın odağı yüksek doğruluklu plaka tespitidir. Sonraki aşamalarda tespit edilen plaka kırpımları OCR katmanına aktarılacak, video akışındaki araçlar takip edilecek ve anlamlı kayıtlar üretilecektir.

## 2. Özellikler

- YOLO11s ile Türk plakası nesne tespiti
- Tek sınıflı (`license_plate`) özel eğitim altyapısı
- Train / validation / test veri hazırlama hattı
- YOLO etiket formatı ve koordinat aralığı doğrulaması
- Eksik veya bozuk etiketleri ayıklayan veri seti split betiği
- Eğitilen model için değerlendirme ve metrik takibi
- Görüntüler üzerinde hızlı çıkarım desteği
- OCR, takip ve raporlama için modüler mimari temeli

## 3. Kullanılan Teknolojiler

| Teknoloji | Kullanım amacı |
| --- | --- |
| 🧠 Python | Uygulama ve veri işleme dili |
| 🧠 Ultralytics YOLO11s | Plaka nesne tespiti ve model eğitimi |
| PyTorch | Derin öğrenme çalışma zamanı |
| OpenCV | Görüntü okuma, işleme ve görselleştirme |
| YAML | YOLO veri seti yapılandırması |
| PaddleOCR *(planlanan)* | Tespit edilen plakaların metin olarak okunması |
| ByteTrack *(planlanan)* | Video içindeki araç/plaka takibi |

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
            ├──► PaddleOCR ile plaka okuma (planlanan)
            ├──► ByteTrack ile araç takibi (planlanan)
            └──► CSV / dashboard / raporlama (planlanan)
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
│   ├── detection/                   # Tespit modülleri
│   ├── ocr/                         # OCR modülleri
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

`requirements.txt` henüz oluşturulmadıysa temel eğitim ortamı için aşağıdaki komut kullanılabilir:

```bash
pip install ultralytics opencv-python
```

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

## 12. Çıkarım (Inference) Örneği

Eğitilmiş ağırlıklarla bir görsel üzerinde plaka tespiti yapmak için:

```bash
yolo detect predict \
  model=models/detection/turkish_plate_yolo11s/weights/best.pt \
  source=ornek_gorsel.jpg \
  conf=0.25 \
  save=True
```

Tahmin görselleri varsayılan olarak Ultralytics çıktı dizinine kaydedilir. Uygulama katmanında bu sonuçlar `outputs/predictions/` altında düzenli biçimde saklanacak şekilde genişletilebilir.

## 13. Gelecek Çalışmalar

- PaddleOCR entegrasyonu ile tespit edilen plakaların metin olarak okunması
- Gerçek zamanlı video işleme desteği
- ByteTrack entegrasyonu ile araç ve plaka takibi
- Duplicate plate filtering ile tekrarlanan kayıtların filtrelenmesi
- CSV logging ile tespit geçmişinin saklanması
- Dashboard ve raporlama sistemi
- Tam otomatik plaka kayıt sistemi

🚗 Nihai hedef; kamera akışından plakayı tespit eden, okuyan, takip eden ve denetlenebilir kayıtlar üreten uçtan uca bir akıllı plaka tanıma sistemi oluşturmaktır.

## 14. Lisans ve Sorumluluk Reddi

Bu proje eğitim, araştırma ve portföy amaçlı geliştirilmiştir. Kullanımdan önce uygun bir lisans dosyası eklenmeli ve kullanılan veri setlerinin lisans koşulları ayrıca doğrulanmalıdır.

Plaka bilgileri kişisel veri veya kişisel veriyle ilişkilendirilebilir nitelikte olabilir. Sistemi gerçek ortamlarda kullanmadan önce yürürlükteki kişisel verilerin korunması, gizlilik, kamera kullanımı ve yerel mevzuat yükümlülüklerinin değerlendirilmesi kullanıcının sorumluluğundadır. Bu proje hukuka aykırı izleme, takip veya veri toplama amacıyla kullanılmamalıdır.

<!--

## Project Structure

```text
AI-Turkish-License-Plate-Recognition/
├── datasets/
├── models/
├── src/
├── outputs/
├── notebooks/
└── README.md
