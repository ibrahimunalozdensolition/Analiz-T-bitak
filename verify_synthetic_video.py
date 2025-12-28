import cv2
import numpy as np

video_path = 'sentetik_test_video_v3_fixed.avi'
cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"=== Sentetik Video Analizi ===")
print(f"Video: {video_path}")
print(f"FPS: {fps}")
print(f"Toplam kare: {total_frames}")
print(f"Boyut: {width}x{height}")
print(f"Sure: {total_frames/fps:.2f} saniye")
print()

ret, prev_frame = cap.read()
if len(prev_frame.shape) == 3:
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
else:
    prev_gray = prev_frame

center_y = height // 2
roi_height = 50

prev_roi = prev_gray[center_y-roi_height//2:center_y+roi_height//2, :]

pixel_to_um = 1.832
frame_count = 0
total_displacement = 0
measurements = []

print("Optical Flow ile gercek hareket olcumu:")
print("-" * 50)

for i in range(min(30, total_frames-1)):
    ret, curr_frame = cap.read()
    if not ret:
        break
    
    if len(curr_frame.shape) == 3:
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    else:
        curr_gray = curr_frame
    
    curr_roi = curr_gray[center_y-roi_height//2:center_y+roi_height//2, :]
    
    flow = cv2.calcOpticalFlowFarneback(
        prev_roi, curr_roi, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )
    
    avg_displacement_x = np.mean(flow[..., 0])
    
    um_per_frame = avg_displacement_x * pixel_to_um
    um_per_second = um_per_frame * fps
    
    measurements.append(um_per_second)
    total_displacement += avg_displacement_x
    
    if i % 10 == 0:
        print(f"Kare {i+1}: {avg_displacement_x:.2f} pixel/kare = {um_per_second:.1f} um/s")
    
    prev_roi = curr_roi.copy()
    frame_count += 1

cap.release()

avg_displacement = total_displacement / frame_count
avg_speed = np.mean(measurements)

print()
print("=== SONUCLAR ===")
print(f"Ortalama yer degistirme: {avg_displacement:.2f} pixel/kare")
print(f"Beklenen: 18.20 pixel/kare")
print(f"Fark: {((avg_displacement/18.20)-1)*100:+.1f}%")
print()
print(f"Olculen gercek hiz: {avg_speed:.1f} um/s")
print(f"Hedef hiz: 1000.0 um/s")
print(f"Fark: {((avg_speed/1000)-1)*100:+.1f}%")
print()

if abs(avg_speed - 1000) < 50:
    print("✓ Sentetik video DOGRU olusturulmus!")
    print("Sorun: Analiz pipeline'inda (UZD/Hough)")
else:
    print("✗ Sentetik video YANLIS olusturulmus!")
    print("Sorun: Video olusturma script'inde")

