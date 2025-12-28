import cv2
import numpy as np

fps = 30
duration_seconds = 5
total_frames = fps * duration_seconds

width = 640
height = 480

pixel_to_um = 1.832

known_speed_um_per_sec = 1000.0

speed_pixel_per_frame = (known_speed_um_per_sec / pixel_to_um) / fps

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('sentetik_test_video_v3.avi', fourcc, fps, (width, height), False)

vessel_center_y = height // 2
vessel_width = 80
centerline_y = vessel_center_y

num_particles = 12
particles = []
for i in range(num_particles):
    x = np.random.randint(-150, 100)
    y_offset = np.random.randint(-3, 4)
    size = np.random.randint(6, 12)
    brightness = np.random.randint(30, 70)
    particles.append([x, y_offset, size, brightness])

print(f"Sentetik Video Bilgileri (v3 - Merkez Cizgisi):")
print(f"FPS: {fps}")
print(f"Sure: {duration_seconds} saniye")
print(f"Toplam kare: {total_frames}")
print(f"Olcek: {pixel_to_um} um/pixel")
print(f"Hedef hiz: {known_speed_um_per_sec} um/s")
print(f"Pixel/kare: {speed_pixel_per_frame:.2f}")
print(f"Merkez cizgisi y: {centerline_y}")
print(f"Parcaciklar merkez cizgisi boyunca hareket ediyor")
print(f"\nVideo olusturuluyor...")

for frame_num in range(total_frames):
    frame = np.ones((height, width), dtype=np.uint8) * 200
    
    cv2.rectangle(frame, 
                  (0, vessel_center_y - vessel_width//2),
                  (width, vessel_center_y + vessel_width//2),
                  150, -1)
    
    cv2.line(frame, (0, centerline_y), (width, centerline_y), 140, 1)
    
    noise = np.random.normal(0, 2, (height, width)).astype(np.int16)
    frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    for particle in particles:
        particle[0] += speed_pixel_per_frame
        
        if particle[0] > width + 100:
            particle[0] = -100
            particle[1] = np.random.randint(-3, 4)
            particle[2] = np.random.randint(6, 12)
            particle[3] = np.random.randint(30, 70)
        
        x = int(particle[0])
        y = centerline_y + particle[1]
        size = particle[2]
        brightness = particle[3]
        
        if -size <= x < width + size and vessel_center_y - vessel_width//2 <= y <= vessel_center_y + vessel_width//2:
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    px, py = x + dx, y + dy
                    if 0 <= px < width and 0 <= py < height:
                        distance = np.sqrt(dx*dx + dy*dy)
                        if distance <= size/2:
                            intensity = int(brightness * (1 - distance/(size/2)))
                            current_val = frame[py, px]
                            frame[py, px] = max(0, current_val - intensity)
    
    if frame_num < 30:
        cv2.putText(frame, f"Hedef: {known_speed_um_per_sec:.0f} um/s", 
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 50, 2)
        cv2.putText(frame, f"Merkez Cizgisi", 
                    (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 50, 1)
    
    out.write(frame)
    
    if (frame_num + 1) % 30 == 0:
        print(f"  {frame_num + 1}/{total_frames} kare olusturuldu")

out.release()

print(f"\nVideo basariyla olusturuldu: sentetik_test_video_v3.avi")
print(f"\nOzellikler:")
print(f"✓ Parcaciklar damar merkez cizgisi boyunca hareket ediyor")
print(f"✓ Y pozisyonu sabit (+- 3 pixel salınım)")
print(f"✓ Sadece X yonunde hareket")
print(f"✓ UZD'de duz diagonal cizgiler oluşacak")
print(f"✓ Hough Transform kolayca tespit edebilecek")
print(f"\nKullanim:")
print(f"1. sentetik_test_video_v3.avi yukleyin")
print(f"2. Damar ortasini kapsayacak sekilde ROI secin")
print(f"3. Mavi merkez cizgisi damar ortasinda olmali")
print(f"4. Tam Analiz calistirin")
print(f"5. Beklenen: ~{known_speed_um_per_sec} um/s")
print(f"6. UZD'de duz diagonal cizgiler gormeli siniz")

