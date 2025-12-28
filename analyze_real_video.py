import cv2
import numpy as np

video_files = ['10_hasta_sl.avi', '14_hasta_sl.avi', '22_hasta_sl.avi']

print("=" * 60)
print("GERÇEK HASTA VİDEOLARI ANALİZİ")
print("=" * 60)

for video_path in video_files:
    print(f"\n{'='*60}")
    print(f"Video: {video_path}")
    print(f"{'='*60}")
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"❌ Video açılamadı: {video_path}")
        continue
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    
    print(f"\n📹 Video Özellikleri:")
    print(f"   FPS: {fps}")
    print(f"   Boyut: {width} x {height} piksel")
    print(f"   Toplam kare: {frame_count}")
    print(f"   Süre: {duration:.2f} saniye")
    
    ret, frame = cap.read()
    if ret:
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            print(f"   Format: Renkli (BGR)")
        else:
            gray = frame
            print(f"   Format: Grayscale")
        
        print(f"\n🔍 Görüntü Analizi:")
        print(f"   Ortalama parlaklık: {np.mean(gray):.1f}")
        print(f"   Kontrast (std): {np.std(gray):.1f}")
        print(f"   Min-Max: {np.min(gray)} - {np.max(gray)}")
        
        center_roi = gray[height//2-50:height//2+50, width//2-100:width//2+100]
        print(f"   Merkez ROI ortalama: {np.mean(center_roi):.1f}")
        
        threshold = np.mean(gray) - np.std(gray)
        dark_pixels = np.sum(gray < threshold)
        dark_ratio = dark_pixels / (width * height) * 100
        print(f"   Koyu piksel oranı: {dark_ratio:.1f}%")
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, 10)
    ret, frame2 = cap.read()
    
    if ret and frame2 is not None:
        if len(frame2.shape) == 3:
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        else:
            gray2 = frame2
        
        frame_diff = cv2.absdiff(gray, gray2)
        motion_level = np.mean(frame_diff)
        
        print(f"\n🎬 Hareket Analizi (Kare 0 vs 10):")
        print(f"   Ortalama fark: {motion_level:.2f}")
        print(f"   Hareket seviyesi: {'Yüksek' if motion_level > 10 else 'Orta' if motion_level > 5 else 'Düşük'}")
    
    cap.release()
    
    print(f"\n⚙️ ÖNERİLEN AYARLAR:")
    print(f"   FPS: {fps} {'✅ (Otomatik tespit)' if fps > 0 and fps < 1000 else '❌ (Manuel gir: 25-30)'}")
    print(f"   Ölçek: 1.832 μm/pixel ✅ (546 pixel = 1000 μm)")
    print(f"   ROI: Damar bölgesini ortalayacak şekilde")
    
    expected_speed_range = "50-500 μm/s (Kapiller damar)"
    print(f"   Beklenen hız: {expected_speed_range}")

print(f"\n{'='*60}")
print("ÖZET VE ÖNERİLER")
print(f"{'='*60}")
print("\n✅ TÜM HASTA VİDEOLARI İÇİN:")
print("   • Ölçek: 1.832 μm/pixel (SABİT)")
print("   • FPS: Otomatik tespit edilecek (genelde 25-30)")
print("   • ROI: Damarın net göründüğü orta kısım")
print("   • Beklenen sonuç: 50-500 μm/s arası")
print("\n⚠️  NOT:")
print("   • Her video farklı FPS'e sahip olabilir")
print("   • Ölçek TÜM videolar için AYNI (1.832)")
print("   • Düşük/yüksek değerler: FPS veya ROI kontrol et")

