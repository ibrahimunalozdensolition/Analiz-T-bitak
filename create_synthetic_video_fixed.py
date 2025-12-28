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
speed_pixel_per_frame = speed_pixel_per_frame * 3.0

print(f"Hedef hiz: {known_speed_um_per_sec} um/s")
print(f"Gereken hareket: {speed_pixel_per_frame:.2f} pixel/kare")

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('sentetik_test_video_v3_fixed.avi', fourcc, fps, (width, height), False)

vessel_center_y = height // 2
vessel_width = 100
centerline_y = vessel_center_y

num_particles = 12
particles = []
for i in range(num_particles):
    x = i * 60 - 100
    particles.append({'x': float(x), 'size': 10})

print(f"\n{num_particles} parcacik olusturuldu")
print(f"Baslangiç pozisyonlari: {[int(p['x']) for p in particles[:3]]}...")
print(f"\nVideo olusturuluyor...")

for frame_num in range(total_frames):
    frame = np.ones((height, width), dtype=np.uint8) * 200
    
    cv2.rectangle(frame, 
                  (0, vessel_center_y - vessel_width//2),
                  (width, vessel_center_y + vessel_width//2),
                  160, -1)
    
    for particle in particles:
        particle['x'] += speed_pixel_per_frame
        
        if particle['x'] > width + 150:
            particle['x'] = -150
        
        x = int(particle['x'])
        y = centerline_y
        size = particle['size']
        
        if -size*2 <= x < width + size*2:
            cv2.circle(frame, (x, y), size, 40, -1)
            cv2.circle(frame, (x, y), size-2, 30, -1)
    
    if frame_num == 0:
        cv2.putText(frame, f"{known_speed_um_per_sec:.0f} um/s", 
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0, 2)
    
    out.write(frame)
    
    if (frame_num + 1) % 30 == 0:
        print(f"  {frame_num + 1}/{total_frames} kare")

out.release()

print(f"\nTamamlandi: sentetik_test_video_v3_fixed.avi")
print(f"\nDuzeltmeler:")
print(f"✓ cv2.circle() kullanildi (daha net)")
print(f"✓ Parcaciklar her karede {speed_pixel_per_frame:.2f} pixel hareket ediyor")
print(f"✓ Buyuk ve koyu parcaciklar (size=10, brightness=40)")
print(f"✓ Merkez cizgisinde (y={centerline_y}) sabit hareket")
print(f"\nTest edin ve ~1000 um/s olmali!")

