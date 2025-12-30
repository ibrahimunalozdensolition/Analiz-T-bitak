# ROI İçinde Damar Takibi - Kamera Hareketi Kompanzasyonu Araştırma Sonuçları

## Problem Tanımı
ROI (Region of Interest) içinde seçilen damar, kamera hareketi nedeniyle başka bir noktaya kayıyor. Bu durum, eritrosit hız analizi için kritik bir sorun teşkil ediyor.

---

## 1. KLASİK GÖRÜNTÜ İŞLEME YÖNTEMLERİ

### 1.1 Optical Flow Tabanlı Takip
| Yöntem | Açıklama | Avantajlar | Dezavantajlar |
|--------|----------|------------|---------------|
| **Lucas-Kanade** | Sparse optical flow, köşe noktalarını takip eder | Hızlı, az hesaplama | Büyük hareketlerde başarısız |
| **Farneback** | Dense optical flow, tüm piksellerin hareketini hesaplar | Detaylı hareket bilgisi | Yavaş, gürültüye hassas |
| **RAFT** | Derin öğrenme tabanlı optical flow | Çok doğru, sub-pixel hassasiyet | GPU gerektirir |

### 1.2 Video Stabilization
| Yöntem | Açıklama | Kullanım |
|--------|----------|----------|
| **Phase Correlation** | Fourier tabanlı kayma tespiti | Global kamera hareketi tespiti |
| **ECC (Enhanced Correlation Coefficient)** | Sub-pixel doğruluk, affine/homography | Kamera dönme ve ölçekleme |
| **Feature Matching (ORB/SIFT)** | Özellik noktaları eşleştirme | Kompleks kamera hareketleri |

### 1.3 Nesne Takip Algoritmaları
| Algoritma | Özellik | Performans |
|-----------|---------|------------|
| **CSRT** | Discriminative correlation filter | Yüksek doğruluk, orta hız |
| **KCF** | Kernelized correlation filter | Hızlı, düşük kaynak |
| **MOSSE** | Minimum output sum of squared error | Çok hızlı, düşük doğruluk |
| **MIL** | Multiple instance learning | Oklüzyona dayanıklı |

---

## 2. YAPAY ZEKA TABANLI ÇÖZÜMLER

### 2.1 SAM (Segment Anything Model)
**Geliştirici:** Meta AI  
**Kaynak:** https://github.com/facebookresearch/segment-anything

| Özellik | Değer |
|---------|-------|
| Görüntü segmentasyonu | ✅ Mükemmel |
| Video takip | ❌ Native desteklemiyor |
| Her frame inference | Yavaş (~100-500ms/frame CPU) |
| GPU gereksinimi | Önerilir |

**Kullanım:** İlk frame'de damar segmentasyonu için ideal, sonraki frame'lerde mask propagation gerekli.

### 2.2 SAM 2 (Segment Anything Model 2)
**Geliştirici:** Meta AI (2024)  
**Kaynak:** https://github.com/facebookresearch/segment-anything-2

| Özellik | Değer |
|---------|-------|
| Video segmentasyonu | ✅ Native destek |
| Memory mechanism | ✅ Önceki frame'leri hatırlar |
| Promptable | ✅ Point, box, mask prompt |
| Real-time | △ GPU ile mümkün |

**Önemli:** SAM 2, video için özel olarak tasarlanmış. Memory attention mekanizması ile önceki frame'lerdeki bilgileri kullanarak takip yapıyor.

### 2.3 XMem (eXtreme Memory for Video Object Segmentation)
**Kaynak:** https://github.com/hkchengrex/XMem

| Özellik | Değer |
|---------|-------|
| Uzun video desteği | ✅ Memory bank sistemi |
| Mask propagation | ✅ Çok etkili |
| Hız | Orta (~30 FPS GPU) |

**Kullanım:** İlk frame'de mask verilir, sonraki frame'lerde otomatik propagate eder.

### 2.4 Cutie (Video Object Segmentation)
**Kaynak:** https://github.com/hkchengrex/Cutie

| Özellik | Değer |
|---------|-------|
| Object-level memory | ✅ |
| Pixel-level memory | ✅ |
| Real-time | △ GPU gerekli |

### 2.5 Track Anything Model (TAM)
**Kaynak:** https://github.com/gaomingqi/Track-Anything

| Özellik | Değer |
|---------|-------|
| SAM + XMem kombinasyonu | ✅ |
| İnteraktif takip | ✅ |
| One-click tracking | ✅ |

### 2.6 CoTracker (Point Tracking)
**Geliştirici:** Meta AI  
**Kaynak:** https://github.com/facebookresearch/co-tracker

| Özellik | Değer |
|---------|-------|
| Dense point tracking | ✅ |
| Video boyunca takip | ✅ |
| Oklüzyon yönetimi | ✅ |

---

## 3. HİBRİT YAKLAŞIMLAR

### 3.1 SAM + Optical Flow
```
İlk Frame → SAM ile segmentasyon → Mask
↓
Sonraki Frameler → Optical flow ile mask warping
↓
Her N frame → SAM ile re-detection
```

### 3.2 SAM + Kalman Filter
```
SAM ile ilk tespit → Kalman state başlatma
↓
Optical flow ile hareket tahmini → Kalman predict
↓
SAM ile periyodik doğrulama → Kalman update
```

### 3.3 Phase Correlation + SAM
```
Phase correlation ile global kamera hareketi tespiti
↓
ROI ve centerline'ı kompanse et
↓
Her N frame'de SAM ile doğrulama
```

---

## 4. PERFORMANS KARŞILAŞTIRMASI

| Yöntem | Doğruluk | Hız (FPS) | GPU Gerekli | Uygulama Zorluğu |
|--------|----------|-----------|-------------|------------------|
| Optical Flow (Farneback) | Orta | 30+ | Hayır | Kolay |
| Phase Correlation | İyi | 50+ | Hayır | Kolay |
| ECC Registration | Çok İyi | 20+ | Hayır | Orta |
| CSRT Tracker | İyi | 25+ | Hayır | Kolay |
| SAM (her frame) | Mükemmel | 2-10 | Önerilir | Orta |
| SAM 2 | Mükemmel | 10-30 | Evet | Orta |
| XMem | Çok İyi | 20-30 | Evet | Orta |
| CoTracker | Çok İyi | 15-25 | Evet | Zor |

---

## 5. ÖNERİLEN ÇÖZÜMLER

### Öneri 1: Hafif Çözüm (GPU Yok)
```python
# Phase Correlation + Periyodik SAM
1. Phase correlation ile global motion tespit
2. ROI ve centerline'ı kompanse et
3. Her 30 frame'de SAM ile mask güncelle
4. Aradaki frame'lerde mask warp et
```
**Avantaj:** CPU'da çalışır, basit implementasyon

### Öneri 2: Orta Çözüm (GPU Var)
```python
# SAM 2 Video Segmentation
1. İlk frame'de SAM 2 ile damar segment et
2. Video boyunca SAM 2'nin memory mekanizması ile takip
3. Otomatik mask propagation
```
**Avantaj:** En doğru sonuç, native video desteği

### Öneri 3: Gelişmiş Çözüm
```python
# XMem + SAM kombinasyonu
1. SAM ile ilk mask oluştur
2. XMem ile video boyunca propagate et
3. Confidence düşerse SAM ile re-segment
```
**Avantaj:** Uzun videolarda stabil

---

## 6. UYGULAMA ÖNCELİKLERİ

### Hemen Uygulanabilir (Mevcut Sistemle)
1. **Phase Correlation** - Kamera hareketini tespit et ve kompanse et
2. **ECC Registration** - Sub-pixel doğrulukla hizalama
3. **Periyodik SAM** - Her N frame'de yeniden tespit

### Orta Vadeli (Yeni Model Entegrasyonu)
1. **SAM 2** - Video için özel tasarlanmış model
2. **XMem** - Memory-based mask propagation

### İleri Düzey (Araştırma)
1. **CoTracker** - Dense point tracking
2. **Custom CNN** - Damar-spesifik model eğitimi

---

## 7. KAYNAKLAR

### GitHub Repositories
- SAM: https://github.com/facebookresearch/segment-anything
- SAM 2: https://github.com/facebookresearch/segment-anything-2
- XMem: https://github.com/hkchengrex/XMem
- Cutie: https://github.com/hkchengrex/Cutie
- TAM: https://github.com/gaomingqi/Track-Anything
- CoTracker: https://github.com/facebookresearch/co-tracker
- RAFT: https://github.com/princeton-vl/RAFT

### Akademik Makaleler
- "Segment Anything" - Kirillov et al., 2023
- "SAM 2: Segment Anything in Images and Videos" - Meta AI, 2024
- "XMem: Long-Term Video Object Segmentation with an Atkinson-Shiffrin Memory Model"
- "RAFT: Recurrent All-Pairs Field Transforms for Optical Flow"
- "Real-Time Segmentation of Non-Rigid Surgical Tools based on Deep Learning and Tracking" - arXiv:2009.03016

### OpenCV Dokümantasyonu
- Optical Flow: https://docs.opencv.org/4.x/d4/dee/tutorial_optical_flow.html
- Video Stabilization: https://docs.opencv.org/4.x/d5/d50/group__videostab.html
- Object Tracking: https://docs.opencv.org/4.x/d9/df8/group__tracking.html

---

## 8. SONUÇ

En uygun çözüm, mevcut sistemin performans gereksinimlerine bağlıdır:

| Senaryo | Önerilen Çözüm |
|---------|----------------|
| Real-time gerekli, GPU yok | Phase Correlation + Periyodik SAM |
| Doğruluk kritik, GPU var | SAM 2 Video Segmentation |
| Uzun videolar | XMem + SAM hibrit |
| En yüksek doğruluk | CoTracker + SAM |

**Mevcut sistem için en pratik çözüm:**
1. Phase Correlation ile global kamera hareketini tespit et
2. ROI ve centerline'ı bu harekete göre kompanse et
3. Her 5-10 frame'de SAM ile mask'ı güncelle
4. Confidence skorunu izle, düşerse hemen SAM çalıştır

