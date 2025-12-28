import cv2
import numpy as np

video_path = 'sentetik_test_video_v3_fixed.avi'
cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)
pixel_to_um = 1.832

ret, frame1 = cap.read()
if len(frame1.shape) == 3:
    frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

cap.set(cv2.CAP_PROP_POS_FRAMES, 10)
ret, frame2 = cap.read()
if len(frame2.shape) == 3:
    frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

cap.release()

print("=== Manuel Pozisyon Takibi ===")
print(f"Kare 0 ve Kare 10 karsilastiriliyor")
print(f"10 kare arasi beklenen hareket: 18.20 * 10 = 182 pixel")
print()

center_y = frame1.shape[0] // 2
roi1 = frame1[center_y-10:center_y+10, :]
roi2 = frame2[center_y-10:center_y+10, :]

threshold = 80
mask1 = (roi1 < threshold).astype(np.uint8) * 255
mask2 = (roi2 < threshold).astype(np.uint8) * 255

contours1, _ = cv2.findContours(mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours2, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print(f"Kare 0'da {len(contours1)} parcacik tespit edildi")
print(f"Kare 10'da {len(contours2)} parcacik tespit edildi")
print()

if len(contours1) > 0 and len(contours2) > 0:
    centers1 = []
    for cnt in contours1:
        M = cv2.moments(cnt)
        if M['m00'] > 0:
            cx = int(M['m10'] / M['m00'])
            centers1.append(cx)
    
    centers2 = []
    for cnt in contours2:
        M = cv2.moments(cnt)
        if M['m00'] > 0:
            cx = int(M['m10'] / M['m00'])
            centers2.append(cx)
    
    centers1 = sorted(centers1)
    centers2 = sorted(centers2)
    
    print("Parcacik merkezleri (x pozisyonlari):")
    print(f"Kare 0:  {centers1[:5]}...")
    print(f"Kare 10: {centers2[:5]}...")
    print()
    
    if len(centers1) >= 3 and len(centers2) >= 3:
        displacements = []
        for i in range(min(3, len(centers1), len(centers2))):
            if i < len(centers2):
                disp = centers2[i] - centers1[i]
                displacements.append(disp)
                print(f"Parcacik {i+1}: {centers1[i]} -> {centers2[i]} = {disp} pixel")
        
        if displacements:
            avg_disp = np.mean(displacements)
            avg_disp_per_frame = avg_disp / 10
            
            print()
            print("=== SONUC ===")
            print(f"10 kare arasi ortalama hareket: {avg_disp:.1f} pixel")
            print(f"Kare basina hareket: {avg_disp_per_frame:.2f} pixel/kare")
            print(f"Beklenen: 18.20 pixel/kare")
            print(f"Fark: {((avg_disp_per_frame/18.20)-1)*100:+.1f}%")
            print()
            
            speed = avg_disp_per_frame * pixel_to_um * fps
            print(f"Hesaplanan hiz: {speed:.1f} um/s")
            print(f"Hedef hiz: 1000.0 um/s")
            
            if abs(speed - 1000) < 100:
                print("\n✓ Video DOGRU olusturulmus!")
            else:
                print(f"\n✗ Hala sorun var ({((speed/1000)-1)*100:+.1f}% hata)")

