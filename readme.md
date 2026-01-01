# Erytroscope - Eritrosit Hız Analiz Sistemi

## Genel Bakış

Erytroscope, kapiller damar videolarından eritrosit (kırmızı kan hücresi) akış hızını analiz eden profesyonel bir tıbbi görüntü analiz uygulamasıdır. Uygulama, Space-Time Diagram (STD) ve Hough Transform yöntemlerini kullanarak otomatik hız ölçümü yapar.

**Geliştirici:** Ibrahim UNAL  
**Danışman:** Prof. Dr. Ugur AKSU  
**Proje:** TÜBİTAK Eritrosit Hız Analizi

---

## Sistem Mimarisi

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              ERYTROSCOPE                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌────────────┐ │
│  │   main.py    │───▶│preprocessing │───▶│analysis_engine│───▶│  utils.py  │ │
│  │   (GUI)      │    │     .py      │    │     .py      │    │ (Outputs)  │ │
│  └──────────────┘    └──────────────┘    └──────────────┘    └────────────┘ │
│         │                   │                   │                           │
│         ▼                   ▼                   ▼                           │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                   │
│  │sam_processor │    │raft_stabilizer│   │vessel_processing│                │
│  │     .py      │    │     .py      │    │     .py      │                   │
│  └──────────────┘    └──────────────┘    └──────────────┘                   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Kullanılan Teknolojiler

### 1. PySide6 (Qt for Python)

**Kullanım Amacı:** Modern masaüstü uygulaması arayüzü oluşturmak

**Kullanılan Bileşenler:**
- `QMainWindow` - Ana uygulama penceresi
- `QThread` - Analiz işlemlerini arka planda çalıştırma (UI donmasını önleme)
- `Signal/Slot` - Thread iletişimi ve progress callback
- `QTimer` - Canlı önizleme için frame güncelleme
- `QSlider` - Frame navigasyonu
- `QProgressBar` - Analiz ilerleme göstergesi
- `QFileDialog` - Video dosyası seçimi
- `QInputDialog` - FPS ve scale düzenleme
- `QMessageBox` - Hata ve bilgi mesajları

**Neden PySide6?**
- Qt framework'ünün resmi Python binding'i
- Cross-platform destek (Windows, macOS, Linux)
- Modern ve özelleştirilebilir widget'lar
- Signal/slot mekanizması ile güvenli thread iletişimi
- Yüksek performanslı rendering

---

### 2. OpenCV (opencv-python, opencv-contrib-python)

**Kullanım Amacı:** Görüntü işleme ve bilgisayarlı görü

**Kullanılan Fonksiyonlar ve Amaçları:**

| Fonksiyon | Dosya | Amaç |
|-----------|-------|------|
| `cv2.VideoCapture` | main.py, analysis_engine.py | Video dosyası okuma |
| `cv2.cvtColor` | Tüm modüller | BGR/RGB/Grayscale dönüşümü |
| `cv2.selectROI` | main.py | Kullanıcıdan ROI seçimi alma |
| `cv2.resize` | main.py | Video frame boyutlandırma |
| `cv2.GaussianBlur` | preprocessing.py, vessel_processing.py | Gürültü azaltma |
| `cv2.createCLAHE` | preprocessing.py | Adaptive histogram equalization |
| `cv2.Canny` | analysis_engine.py | Kenar tespiti (STD için) |
| `cv2.HoughLinesP` | analysis_engine.py | Çizgi tespiti (eritrosit izleri) |
| `cv2.calcOpticalFlowPyrLK` | preprocessing.py | Lucas-Kanade sparse optical flow |
| `cv2.calcOpticalFlowFarneback` | vessel_processing.py, sam_processor.py | Dense optical flow |
| `cv2.findTransformECC` | preprocessing.py | ECC stabilizasyon |
| `cv2.estimateAffinePartial2D` | preprocessing.py | Affine transform tahmini |
| `cv2.warpAffine` | preprocessing.py, raft_stabilizer.py | Frame dönüşümü uygulama |
| `cv2.goodFeaturesToTrack` | preprocessing.py | Shi-Tomasi köşe noktaları |
| `cv2.adaptiveThreshold` | vessel_processing.py, sam_processor.py | Adaptive eşikleme |
| `cv2.morphologyEx` | vessel_processing.py, sam_processor.py | Morfolojik işlemler |
| `cv2.erode/dilate` | sam_processor.py, vessel_processing.py | Skeletonization |
| `cv2.findContours` | sam_processor.py | Kontur bulma |
| `cv2.Sobel` | vessel_processing.py, sam_processor.py | Gradient hesaplama |
| `cv2.matchTemplate` | raft_stabilizer.py | Template matching |
| `cv2.Laplacian` | preprocessing.py | Keskinlik/gürültü ölçümü |

**Neden OpenCV?**
- Endüstri standardı görüntü işleme kütüphanesi
- Optimize edilmiş C++ backend'i
- Kapsamlı video işleme desteği
- Çok sayıda bilgisayarlı görü algoritması

---

### 3. PyTorch ve TorchVision

**Kullanım Amacı:** Derin öğrenme tabanlı optical flow (RAFT) ve segmentasyon (SAM)

**RAFT Stabilizer (raft_stabilizer.py):**

```python
from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
```

| Bileşen | Amaç |
|---------|------|
| `raft_small` | RAFT (Recurrent All-pairs Field Transforms) optical flow modeli |
| `torch.device` | GPU/MPS/CPU seçimi |
| `torch.no_grad()` | İnference modunda gradient hesaplamasını devre dışı bırakma |
| `tensor.to(device)` | Model ve tensor'ları uygun cihaza taşıma |

**RAFT Nedir?**
- ECCV 2020'de tanıtılan state-of-the-art optical flow algoritması
- Dense (piksel bazında) hareket tahmini
- Kamera sarsıntısı kompanzasyonu için kullanılıyor
- ROI takibi için global motion hesaplama

**Device Seçimi:**
```python
if torch.cuda.is_available():
    self.device = torch.device('cuda')      # NVIDIA GPU
elif torch.backends.mps.is_available():
    self.device = torch.device('mps')       # Apple Silicon GPU
else:
    self.device = torch.device('cpu')       # CPU fallback
```

---

### 4. Segment Anything Model (SAM)

**Kullanım Amacı:** Damar segmentasyonu (sam_processor.py)

**Bileşenler:**
```python
from segment_anything import sam_model_registry, SamPredictor
```

| Sınıf | Amaç |
|-------|------|
| `SAMVesselSegmenter` | SAM ile damar segmentasyonu |
| `LightweightVesselSegmenter` | SAM olmadan klasik segmentasyon (fallback) |
| `SAMLiveTracker` | Canlı video takibi için SAM |
| `HybridVesselTracker` | SAM + Optical Flow hibrit yaklaşım |

**SAM Özellikleri:**
- Meta AI tarafından geliştirilen zero-shot segmentasyon modeli
- Point prompt ile segmentasyon (damar üzerine tıklama)
- Box prompt ile segmentasyon (ROI kutusu)
- Multi-mask output ile en iyi mask seçimi
- Mask propagation ile ardışık frame'lerde takip

**Model Dosyası:**
- `sam_vit_b_01ec64.pth` - ViT-B (Vision Transformer Base) checkpoint

---

### 5. NumPy

**Kullanım Amacı:** Sayısal hesaplamalar ve array işlemleri

**Kullanılan İşlemler:**

| İşlem | Dosya | Amaç |
|-------|-------|------|
| `np.array` | Tüm modüller | Python listelerini array'e dönüştürme |
| `np.zeros` | preprocessing.py, analysis_engine.py | Boş görüntü/matris oluşturma |
| `np.mean/median` | utils.py, raft_stabilizer.py | İstatistik hesaplama |
| `np.std` | utils.py, preprocessing.py | Standart sapma |
| `np.percentile` | analysis_engine.py, utils.py | Yüzdelik dilim hesaplama |
| `np.clip` | raft_stabilizer.py, vessel_processing.py | Değerleri sınırlama |
| `np.sqrt` | sam_processor.py, analysis_engine.py | Kare kök |
| `np.arctan2` | raft_stabilizer.py | Açı hesaplama |
| `np.convolve` | raft_stabilizer.py, vessel_processing.py | 1D konvolüsyon (smoothing) |
| `np.corrcoef` | vessel_processing.py | Korelasyon katsayısı |
| `np.argsort` | sam_processor.py, vessel_processing.py | Sıralama indeksleri |
| `np.linspace` | vessel_processing.py | Eşit aralıklı değerler |
| `np.column_stack/where` | sam_processor.py, vessel_processing.py | Skeleton noktalarını çıkarma |

---

### 6. SciPy

**Kullanım Amacı:** Uzamsal veri yapıları (vessel_processing.py)

```python
from scipy.spatial import KDTree
```

**KDTree Kullanımı:**
- Centerline noktalarını sıralamak için
- En yakın komşu araması (k-nearest neighbors)
- Skeleton noktalarını düzgün bir çizgi haline getirme

---

### 7. Matplotlib

**Kullanım Amacı:** Histogram görselleştirme (utils.py)

```python
import matplotlib.pyplot as plt
```

**Kullanılan Özellikler:**
- `plt.hist()` - Hız dağılımı histogramı
- `plt.axvline()` - Ortalama, medyan, yüzdelik çizgileri
- `plt.savefig()` - PNG olarak kaydetme
- `BytesIO` - Memory içinde görüntü oluşturma

---

## Modül Detayları

### main.py - Ana Uygulama

**Sınıflar:**

| Sınıf | Amaç |
|-------|------|
| `EritrosidAnalyzer` | Ana pencere ve uygulama mantığı |
| `AnalysisThread` | Arka planda analiz çalıştırma |
| `ModernButton` | Özelleştirilmiş stil butonları |
| `InfoCard` | İstatistik gösterim kartları |
| `ImageViewerDialog` | STD ve histogram görüntüleme dialog'u |
| `AboutDialog` | Hakkında bilgisi |

**Ana Akış:**
1. Video yükleme ve metadata okuma (FPS, frame sayısı)
2. ROI seçimi (cv2.selectROI)
3. Damar merkez çizgisi tespiti (VesselProcessor veya SAM)
4. Analiz başlatma (AnalysisThread)
5. Sonuçları gösterme

**Canlı Önizleme Sistemi:**
- RAFT tabanlı ROI takibi
- Real-time flow vektörleri görselleştirme
- Anlık hız tahmini
- Drift kompanzasyonu

---

### preprocessing.py - Ön İşleme

**VideoPreprocessor Sınıfı:**

| Metod | Amaç |
|-------|------|
| `stabilize_frame()` | Sparse optical flow ile frame stabilizasyonu |
| `stabilize_roi_frames()` | ECC (Enhanced Correlation Coefficient) ile ROI stabilizasyonu |
| `detect_background_mode()` | SNR'a göre otomatik mod seçimi |
| `compute_mean_image()` | Ortalama görüntü hesaplama (background subtraction için) |
| `apply_background_removal_mode1()` | Mean + CLAHE modu |
| `apply_background_removal_mode2()` | Direkt CLAHE modu |
| `convert_to_optical_density()` | OD = -log(I/I0) dönüşümü |
| `measure_quality()` | Kontrast, gürültü, SNR ölçümü |

**Stabilizasyon Akışı:**
```
Frame N-1, Frame N
       │
       ▼
goodFeaturesToTrack (Köşe noktaları)
       │
       ▼
calcOpticalFlowPyrLK (Nokta takibi)
       │
       ▼
estimateAffinePartial2D (Transform hesaplama)
       │
       ▼
warpAffine (Frame düzeltme)
```

---

### analysis_engine.py - Analiz Motoru

**SpaceTimeDiagramAnalyzer Sınıfı:**

| Metod | Amaç |
|-------|------|
| `analyze_video()` | Ana analiz pipeline'ı |
| `create_space_time_diagram()` | STD matris oluşturma |
| `detect_lines_hough()` | Hough Transform çizgi tespiti |
| `calculate_speed_from_line()` | Çizgi eğiminden hız hesaplama |
| `analyze_all_lines()` | Tüm çizgileri analiz etme |
| `check_aliasing()` | Nyquist limit kontrolü |

**STD (Space-Time Diagram) Oluşturma:**
```
Her frame için:
    ├─ Centerline boyunca intensity profili çıkar
    ├─ Profili STD matrisin bir sütununa yerleştir
    └─ Damar merkez çizgisini takip et (drift kompanzasyonu)

Sonuç: [Merkez çizgi noktaları x Frame sayısı] boyutunda 2D matris
       Eritrosit hareketleri diagonal çizgiler olarak görünür
```

**Hız Hesaplama Formülü:**
```
Delta_time_seconds = abs(x2 - x1) / FPS
Delta_space_um = abs(y2 - y1) × pixel_to_um
Speed = Delta_space_um / Delta_time_seconds (um/s)
```

**Aliasing Kontrolü:**
```
Nyquist_limit = (ROI_height / 2) × pixel_to_um × FPS
Eğer speed > Nyquist_limit → Aliasing şüphesi
```

---

### raft_stabilizer.py - RAFT Tabanlı Stabilizasyon

**RAFTStabilizer Sınıfı:**

| Metod | Amaç |
|-------|------|
| `compute_flow()` | RAFT ile dense optical flow hesaplama |
| `get_global_motion()` | Median flow ile global hareket |
| `stabilize_roi()` | ROI pozisyonunu güncelleme |
| `_template_match_correction()` | Template matching ile düzeltme |
| `_verify_vessel_position()` | Damar merkezini doğrulama |

**Stabilizasyon Stratejisi:**
1. ROI çevresinde optical flow hesapla
2. Global motion'ı (kamera hareketi) çıkar
3. ROI'yi yeni pozisyona taşı
4. Her 5 frame'de damar merkezi doğrula
5. Her 10 frame'de template matching düzeltmesi
6. Maximum drift kontrolü
7. Periyodik cumulative offset reset

**RAFTVesselTracker:**
- RAFTStabilizer'ı kullanarak ROI takibi
- Centerline noktalarını ROI ile birlikte güncelleme

---

### sam_processor.py - SAM Tabanlı Segmentasyon

**SAMVesselSegmenter Sınıfı:**

| Metod | Amaç |
|-------|------|
| `load_model()` | SAM checkpoint yükleme |
| `segment_vessel_with_points()` | Point prompt ile segmentasyon |
| `segment_vessel_with_box()` | Box prompt ile segmentasyon |
| `segment_vessel_auto()` | Otomatik merkez noktası ile segmentasyon |
| `extract_centerline_from_mask()` | Mask'tan centerline çıkarma |
| `_skeletonize()` | Morfolojik skeletonization |

**LightweightVesselSegmenter (SAM Olmadan):**
- Adaptive threshold ile segmentasyon
- Frangi filter ile damar tespiti (Hessian tabanlı)
- En büyük kontur seçimi

**SAMLiveTracker:**
- SAM ile gerçek zamanlı damar takibi
- Mask propagation (önceki mask'ı sonraki frame'e aktarma)
- Periyodik tam inference
- Reference point güncelleme

---

### vessel_processing.py - Damar İşleme

**VesselProcessor Sınıfı:**

| Metod | Amaç |
|-------|------|
| `extract_centerline_from_roi()` | ROI'dan centerline çıkarma |
| `auto_detect_centerline()` | Otomatik damar merkezi tespiti |
| `detect_vessel_direction()` | Damar yönü tespiti (dikey/yatay/çapraz) |
| `get_intensity_profile_along_centerline()` | Centerline boyunca intensity |
| `track_centerline()` | Frame'ler arası centerline takibi |
| `track_roi()` | Farneback optical flow ile ROI takibi |
| `_trace_vessel_centerline()` | Yoğunluk profiline göre centerline |
| `_smooth_centerline_curve()` | Hareketli ortalama ile smoothing |
| `template_match_vessel()` | Template matching ile damar bulma |

**Centerline Çıkarma Algoritması:**
1. Görüntüyü ters çevir (damar karanlık → açık)
2. Sütun bazında toplam yoğunluk hesapla
3. En yüksek yoğunluklu sütunu bul
4. Her satırda lokal maksimumu takip et
5. Sonuçları smooth et (window=7)

**ROI Takip Sistemi (Farneback):**
```
Motion Compensation Pipeline:
    1. Dense optical flow hesapla (Farneback)
    2. Median flow ile global motion bul
    3. Global motion'ı çıkar
    4. ROI içindeki median motion'ı al
    5. Küçük hareketleri filtrele (<0.3 px)
    6. Büyük ani hareketleri sınırla (<8 px)
    7. Drift kontrolü ve düzeltme
```

---

### utils.py - Yardımcı Fonksiyonlar

**ResultsManager:**
- CSV formatında sonuç kaydetme
- İstatistikler (ortalama, medyan, SD, min, max)
- Yüzdelikler (P25, P50, P75, P90, P95)
- Ölçüm sayıları (valid, aliasing, toplam)
- Her bir hız değeri

**HistogramGenerator:**
- Matplotlib ile histogram oluşturma
- Ortalama ve medyan çizgileri (kırmızı/yeşil)
- P25 ve P75 işaretçileri (turuncu)
- n ve SD bilgi kutusu

**QualityChecker:**
- Nyquist limit hesaplama
- Aliasing kontrolü
- Geçerli/şüpheli hız ayrımı

---

## Veri Akışı

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         VİDEO DOSYASI                                    │
│                      (AVI, MP4, MOV)                                     │
└────────────────────────────┬────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      1. VİDEO YÜKLEME                                    │
│  • cv2.VideoCapture ile dosya açma                                       │
│  • FPS otomatik tespit (CAP_PROP_FPS)                                   │
│  • Frame sayısı (CAP_PROP_FRAME_COUNT)                                  │
│  • İlk frame görüntüleme                                                 │
└────────────────────────────┬────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       2. ROI SEÇİMİ                                      │
│  • cv2.selectROI ile kullanıcı seçimi                                   │
│  • Damar merkez çizgisi çıkarma (VesselProcessor)                       │
│  • Otomatik veya manuel onay                                             │
└────────────────────────────┬────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    3. FRAME OKUMA VE STABİLİZASYON                       │
│  • Her frame için:                                                       │
│    ├─ Sparse optical flow stabilizasyonu (preprocessing.py)             │
│    ├─ RAFT ile ROI takibi (raft_stabilizer.py)                          │
│    └─ ROI offset kaydetme                                                │
└────────────────────────────┬────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                   4. ROI FRAME STABİLİZASYONU                            │
│  • ECC (Enhanced Correlation Coefficient) ile                            │
│  • Referans frame'e göre hizalama                                        │
└────────────────────────────┬────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                  5. ARKA PLAN MOD TESPİTİ                                │
│  • SNR hesaplama (preprocessing.py)                                      │
│  • Yüksek SNR → Mean + CLAHE                                             │
│  • Düşük SNR → Direkt CLAHE                                              │
└────────────────────────────┬────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    6. FRAME İŞLEME                                       │
│  • Background removal                                                    │
│  • CLAHE uygulama                                                        │
│  • Düşük kontrastsa Optical Density dönüşümü                            │
└────────────────────────────┬────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                 7. STD (SPACE-TIME DIAGRAM) OLUŞTURMA                    │
│  • Her frame için centerline boyunca intensity profili                   │
│  • Profilleri yan yana dizerek 2D matris                                 │
│  • Centerline takibi (drift kompanzasyonu)                               │
└────────────────────────────┬────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                   8. HOUGH TRANSFORM                                     │
│  • Gaussian blur                                                         │
│  • Canny edge detection                                                  │
│  • HoughLinesP ile çizgi tespiti                                        │
│  • Eğim ve uzunluk filtreleme                                           │
└────────────────────────────┬────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    9. HIZ HESAPLAMA                                      │
│  • Her çizgi için eğim → hız dönüşümü                                   │
│  • Minimum hız filtresi (200 um/s)                                       │
│  • Aliasing kontrolü (Nyquist limit)                                     │
└────────────────────────────┬────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                   10. İSTATİSTİK HESAPLAMA                               │
│  • Ortalama, medyan, standart sapma                                      │
│  • Minimum, maksimum                                                     │
│  • Yüzdelikler (P25, P50, P75, P90, P95)                                │
│  • Geçerli/aliasing ölçüm sayıları                                      │
└────────────────────────────┬────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      11. ÇIKTILAR                                        │
│  • GUI'de istatistik kartları                                            │
│  • STD görselleştirme (çizgilerle)                                       │
│  • Histogram (Matplotlib)                                                │
│  • CSV export                                                            │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Teknik Parametreler

### Uzaysal Ölçek
```
546 pixel = 1000 um
1 pixel = 1.832 um
```

### Hough Transform Parametreleri
```python
rho = 1                    # Mesafe çözünürlüğü (pixel)
theta = np.pi/180          # Açı çözünürlüğü (radyan)
threshold = 15             # Minimum oy sayısı
minLineLength = 10         # Minimum çizgi uzunluğu (pixel)
maxLineGap = 8             # Maksimum çizgi boşluğu (pixel)
```

### Hız Filtreleme
```python
min_slope = 0.05           # Minimum eğim
max_slope = 20.0           # Maksimum eğim
min_speed = 200.0          # Minimum hız (um/s)
```

### RAFT Stabilizasyon
```python
max_drift = 50             # Maksimum ROI kayması (pixel)
reset_interval = 120       # Cumulative offset reset aralığı (frame)
template_update_interval = 30  # Template güncelleme aralığı
max_motion = 15            # Frame başı maksimum hareket (pixel)
```

### Farneback Optical Flow
```python
pyr_scale = 0.5            # Pyramid scale
levels = 4                 # Pyramid seviyeleri
winsize = 21               # Pencere boyutu
iterations = 5             # İterasyon sayısı
poly_n = 7                 # Polynomial genişletme boyutu
poly_sigma = 1.5           # Gaussian sigma
```

---

## Kurulum

### Gereksinimler

```bash
python >= 3.8
```

### Bağımlılıklar (requirements.txt)

```
opencv-python>=4.8.0
opencv-contrib-python>=4.8.0
PySide6>=6.5.0
numpy>=1.24.0
matplotlib>=3.7.0
scipy>=1.10.0
torch>=2.0.0
torchvision>=0.15.0
segment-anything>=1.0
```

### Kurulum Adımları

```bash
# 1. Repoyu klonla
git clone https://github.com/ibrahimunalozdensolition/Analiz-T-bitak.git
cd Analiz-T-bitak

# 2. Virtual environment oluştur
python3 -m venv venv

# 3. Aktive et
source venv/bin/activate  # Linux/macOS
# veya
.\venv\Scripts\activate   # Windows

# 4. Bağımlılıkları yükle
pip install -r requirements.txt
```

### SAM Model Kurulumu (Opsiyonel)

SAM (Segment Anything Model) daha iyi damar segmentasyonu sağlar. Model dosyası büyük boyutlu (358 MB) olduğundan manuel indirilmelidir:

```bash
# SAM ViT-B model dosyasını indir
curl -L -o sam_vit_b_01ec64.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

# veya wget ile:
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

**Not:** SAM model olmadan da uygulama çalışır - klasik görüntü işleme yöntemleri otomatik olarak kullanılır.

---

## Kullanım

### Uygulamayı Başlat
```bash
source venv/bin/activate
python main.py
```

### İş Akışı

1. **Video Yükle** - Video dosyasını seç (.avi, .mp4, .mov)
2. **FPS Kontrol** - Otomatik tespit edilir, gerekirse düzenle
3. **Bölge Seç** - Damar bölgesini fare ile seç (ROI)
4. **Merkez Çizgisi** - Otomatik tespit edilir, onay ver veya manuel çiz
5. **Tam Analiz** - Analizi başlat
6. **Sonuçlar** - İstatistikleri incele
7. **STD Göster** - Space-Time Diagram'ı incele
8. **Histogram Göster** - Hız dağılımını görüntüle
9. **CSV Kaydet** - Sonuçları dışa aktar

### Canlı Önizleme

- ROI seçildikten sonra "Live Preview" butonu aktif olur
- RAFT tabanlı gerçek zamanlı takip
- Anlık hız gösterimi
- Flow vektörleri görselleştirme

---

## Dosya Yapısı

```
Analiz-T-bitak/
├── main.py                 # Ana uygulama ve GUI
├── analysis_engine.py      # STD oluşturma, Hough Transform, hız hesaplama
├── preprocessing.py        # Video stabilizasyonu, CLAHE, OD dönüşümü
├── vessel_processing.py    # Damar tespiti, centerline çıkarma
├── sam_processor.py        # SAM segmentasyonu, hibrit takip
├── raft_stabilizer.py      # RAFT optical flow, ROI stabilizasyonu
├── utils.py                # CSV export, histogram, yardımcı fonksiyonlar
├── requirements.txt        # Python bağımlılıkları
├── sam_vit_b_01ec64.pth    # SAM model dosyası (opsiyonel)
├── protokol.pdf            # Analiz protokolü
├── readme.md               # Bu dosya
└── *.avi                   # Örnek video dosyaları
```

---

## Arayüz

### Ana Pencere Düzeni

```
┌─────────────────────────────────────────────────────────────────────────┐
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │ Erythrocyte Velocity Analysis    [STD + Hough Transform] [About][X]│ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│  ┌─────────────────────────────────────────┐ ┌─────────────────────────┐│
│  │                                         │ │ Video Information       ││
│  │                                         │ │ ┌─────────────────────┐ ││
│  │                                         │ │ │ Video: hasta_1.avi  │ ││
│  │           VIDEO GÖRÜNTÜSÜ               │ │ │ FPS: 30.0      [Edit]│ ││
│  │                                         │ │ │ Scale: 1.832   [Edit]│ ││
│  │                                         │ │ │ Frame Count: 500    │ ││
│  │                                         │ │ └─────────────────────┘ ││
│  │                                         │ │                         ││
│  │                                         │ │ Analysis Results        ││
│  │                                         │ │ ┌─────────────────────┐ ││
│  │                                         │ │ │ Average Speed       │ ││
│  │                                         │ │ │     456.7 um/s      │ ││
│  ├─────────────────────────────────────────┤ │ │ Median Speed        │ ││
│  │[Load Video][Select Region][Full Analysis]│ │ │     432.1 um/s      │ ││
│  │[Live Preview]                            │ │ │ Standard Deviation  │ ││
│  │                                         │ │ │      89.3 um/s      │ ││
│  ├─────────────────────────────────────────┤ │ │ Min: 234.5 um/s     │ ││
│  │ Frame: 125 / 500                        │ │ │ Max: 789.2 um/s     │ ││
│  │ [◀ Previous] ════════════ [Next ▶]      │ │ └─────────────────────┘ ││
│  │                                         │ │ Valid: 45 | Aliasing: 3││
│  └─────────────────────────────────────────┘ │                         ││
│                                              │ Outputs                  ││
│                                              │ [Show STD]              ││
│                                              │ [Show Histogram]        ││
│                                              │ [Save CSV]              ││
│                                              └─────────────────────────┘│
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │ Ready                                                              │ │
│  └────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
```

### Renk Şeması

| Öğe | Renk | Hex |
|-----|------|-----|
| Arka plan | Açık gri | #F5F5F5 |
| Kartlar | Beyaz | #FFFFFF |
| Primary buton | Mavi | #2196F3 |
| Başarı | Yeşil | #4CAF50 |
| Uyarı | Turuncu | #FF9800 |
| Hata | Kırmızı | #f44336 |
| ROI çerçevesi | Yeşil | (0, 255, 0) |
| Centerline | Mavi | (255, 0, 0) |
| Hough çizgileri | Kırmızı | (0, 0, 255) |

---

## Protokol Uyumluluğu

Bu uygulama protocol.pdf'deki tüm temel gereksinimleri karşılar:

- [x] Video stabilizasyonu
- [x] Hibrit arka plan temizleme
- [x] Adaptif OD dönüşümü
- [x] Otomatik damar düzeltmesi
- [x] STD oluşturma
- [x] Hough Transform
- [x] Eğim tabanlı hız hesaplama
- [x] Aliasing kontrolü
- [x] Histogram çıktısı
- [x] İstatistiksel analiz
- [x] CSV export

---

## Hata Ayıklama

### Sık Karşılaşılan Sorunlar

**SAM model bulunamıyor:**
```
SAM checkpoint bulunamadi. Lutfen sam_vit_b_01ec64.pth dosyasini indirin
```
Çözüm: Model dosyasını indirin veya klasik segmentasyon otomatik kullanılır.

**FPS okunamıyor:**
```
FPS could not be read, using default 30
```
Çözüm: Edit butonu ile doğru FPS değerini girin.

**Yetersiz veri:**
```
Yetersiz veri - Valid olcum sayisi (X) 10'un altinda
```
Çözüm: ROI boyutunu artırın veya daha net bir damar bölgesi seçin.

---

## Lisans

Bu proje TÜBİTAK projesi kapsamında geliştirilmiştir.
