import cv2
import numpy as np
from scipy.spatial import KDTree

class VesselProcessor:
    def __init__(self):
        # Optical Flow tabanlı takip için state
        self.frame_counter = 0
        self.tracking_confidence = 1.0

        # ROI reference
        self.original_roi = None
        self.reference_roi_image = None
        self.accumulated_offset = np.array([0.0, 0.0])  # Kümülatif offset

        # Reset mekanizması
        self.last_reset_frame = 0
        self.reset_interval = 60  # Artırıldı: 40 -> 60 (daha az sıklıkta reset)
        
    def reset_tracking_state(self):
        """Takip durumunu sıfırla"""
        self.frame_counter = 0
        self.tracking_confidence = 1.0
        self.original_roi = None
        self.reference_roi_image = None
        self.accumulated_offset = np.array([0.0, 0.0])
        self.last_reset_frame = 0
    
    def extract_centerline(self, binary_image):
        kernel = np.ones((3, 3), np.uint8)
        cleaned = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel, iterations=2)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)
        
        skeleton = self._simple_skeleton(cleaned)
        
        return skeleton
    
    def _simple_skeleton(self, binary_image):
        size = np.size(binary_image)
        skel = np.zeros(binary_image.shape, np.uint8)
        
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        
        img = binary_image.copy()
        
        while True:
            eroded = cv2.erode(img, element)
            temp = cv2.dilate(eroded, element)
            temp = cv2.subtract(img, temp)
            skel = cv2.bitwise_or(skel, temp)
            img = eroded.copy()
            
            if cv2.countNonZero(img) == 0:
                break
        
        return skel
    
    def auto_threshold_vessel(self, gray_image):
        blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
        
        adaptive_thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 15, 3
        )
        
        return adaptive_thresh
    
    def find_vessel_mask(self, gray_image, use_adaptive=True):
        if use_adaptive:
            binary = self.auto_threshold_vessel(gray_image)
        else:
            _, binary = cv2.threshold(
                gray_image, 0, 255,
                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )
        
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        
        return binary
    
    def detect_vessel_direction(self, gray_image):
        blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
        
        sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
        
        grad_x_sum = np.sum(np.abs(sobel_x))
        grad_y_sum = np.sum(np.abs(sobel_y))
        
        if grad_y_sum > grad_x_sum * 1.2:
            return 'horizontal'
        elif grad_x_sum > grad_y_sum * 1.2:
            return 'vertical'
        else:
            return 'diagonal'
    
    def find_vessel_center_column(self, gray_image):
        height, width = gray_image.shape
        
        col_variances = []
        for col in range(width):
            col_data = gray_image[:, col].astype(np.float64)
            variance = np.var(col_data)
            col_variances.append(variance)
        
        col_variances = np.array(col_variances)
        
        kernel_size = max(5, width // 10)
        if kernel_size % 2 == 0:
            kernel_size += 1
        smoothed = np.convolve(col_variances, np.ones(kernel_size)/kernel_size, mode='same')
        
        center_region = width // 4
        search_start = center_region
        search_end = width - center_region
        
        if search_end > search_start:
            center_idx = search_start + np.argmax(smoothed[search_start:search_end])
        else:
            center_idx = width // 2
        
        return center_idx
    
    def auto_detect_centerline(self, roi_image):
        if len(roi_image.shape) == 3:
            gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi_image
        
        height, width = gray.shape
        
        direction = self.detect_vessel_direction(gray)
        
        centerline_points = self._trace_vessel_centerline(gray, direction)
        
        if len(centerline_points) > 5:
            centerline_points = self._smooth_centerline_curve(centerline_points)
        
        return centerline_points, direction
    
    def _trace_vessel_centerline(self, gray, direction):
        height, width = gray.shape
        
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        intensity_profile = np.mean(blurred, axis=0) if direction != 'horizontal' else np.mean(blurred, axis=1)
        
        inverted = 255 - blurred
        
        if direction != 'horizontal':
            col_sums = np.sum(inverted, axis=0)
            
            kernel_size = max(5, width // 8)
            if kernel_size % 2 == 0:
                kernel_size += 1
            smoothed_sums = np.convolve(col_sums, np.ones(kernel_size)/kernel_size, mode='same')
            
            margin = width // 6
            search_region = smoothed_sums[margin:width-margin]
            if len(search_region) > 0:
                best_col = margin + np.argmax(search_region)
            else:
                best_col = width // 2
            
            centerline_points = []
            window = max(10, width // 8)
            
            for row in range(height):
                left = max(0, best_col - window)
                right = min(width, best_col + window)
                
                row_segment = inverted[row, left:right]
                if len(row_segment) > 0:
                    local_max = left + np.argmax(row_segment)
                    centerline_points.append([row, local_max])
                else:
                    centerline_points.append([row, best_col])
        else:
            row_sums = np.sum(inverted, axis=1)
            
            kernel_size = max(5, height // 8)
            if kernel_size % 2 == 0:
                kernel_size += 1
            smoothed_sums = np.convolve(row_sums, np.ones(kernel_size)/kernel_size, mode='same')
            
            margin = height // 6
            search_region = smoothed_sums[margin:height-margin]
            if len(search_region) > 0:
                best_row = margin + np.argmax(search_region)
            else:
                best_row = height // 2
            
            centerline_points = []
            window = max(10, height // 8)
            
            for col in range(width):
                top = max(0, best_row - window)
                bottom = min(height, best_row + window)
                
                col_segment = inverted[top:bottom, col]
                if len(col_segment) > 0:
                    local_max = top + np.argmax(col_segment)
                    centerline_points.append([local_max, col])
                else:
                    centerline_points.append([best_row, col])
        
        return np.array(centerline_points, dtype=np.int32)
    
    def _smooth_centerline_curve(self, points, window_size=7):
        if len(points) < window_size:
            return points
        
        smoothed = np.zeros_like(points, dtype=np.float64)
        half_window = window_size // 2
        
        for i in range(len(points)):
            start = max(0, i - half_window)
            end = min(len(points), i + half_window + 1)
            smoothed[i] = np.mean(points[start:end], axis=0)
        
        return smoothed.astype(np.int32)
    
    def get_centerline_points(self, skeleton):
        points = np.column_stack(np.where(skeleton > 0))
        
        if len(points) == 0:
            return np.array([])
        
        sorted_points = self._sort_points_fast(points)
        
        return sorted_points
    
    def _sort_points_fast(self, points):
        if len(points) == 0:
            return points
        
        if len(points) < 3:
            return points
        
        y_range = np.max(points[:, 0]) - np.min(points[:, 0])
        x_range = np.max(points[:, 1]) - np.min(points[:, 1])
        
        if y_range >= x_range:
            sort_axis = 0
        else:
            sort_axis = 1
        
        sorted_indices = np.argsort(points[:, sort_axis])
        coarse_sorted = points[sorted_indices]
        
        if len(points) > 50:
            tree = KDTree(coarse_sorted)
            
            visited = np.zeros(len(coarse_sorted), dtype=bool)
            result = []
            
            current_idx = 0
            visited[current_idx] = True
            result.append(coarse_sorted[current_idx])
            
            for _ in range(len(coarse_sorted) - 1):
                distances, indices = tree.query(coarse_sorted[current_idx], k=min(10, len(coarse_sorted)))
                
                next_idx = None
                for idx in indices:
                    if not visited[idx]:
                        next_idx = idx
                        break
                
                if next_idx is None:
                    for idx in range(len(visited)):
                        if not visited[idx]:
                            next_idx = idx
                            break
                
                if next_idx is None:
                    break
                
                visited[next_idx] = True
                result.append(coarse_sorted[next_idx])
                current_idx = next_idx
            
            return np.array(result, dtype=np.int32)
        else:
            return coarse_sorted
    
    def _sort_points_along_line(self, points):
        return self._sort_points_fast(points)
    
    def smooth_centerline(self, points, window_size=5):
        if len(points) < window_size:
            return points
        
        smoothed = np.zeros_like(points, dtype=np.float64)
        
        half_window = window_size // 2
        
        for i in range(len(points)):
            start = max(0, i - half_window)
            end = min(len(points), i + half_window + 1)
            smoothed[i] = np.mean(points[start:end], axis=0)
        
        return smoothed.astype(np.int32)
    
    def resample_centerline(self, points, num_samples=None, spacing=None):
        if len(points) < 2:
            return points
        
        total_length = 0
        for i in range(1, len(points)):
            total_length += np.linalg.norm(points[i] - points[i-1])
        
        if spacing is not None:
            num_samples = max(2, int(total_length / spacing))
        elif num_samples is None:
            num_samples = len(points)
        
        cumulative_dist = [0]
        for i in range(1, len(points)):
            cumulative_dist.append(
                cumulative_dist[-1] + np.linalg.norm(points[i] - points[i-1])
            )
        
        cumulative_dist = np.array(cumulative_dist)
        
        target_dists = np.linspace(0, total_length, num_samples)
        
        resampled = []
        for target in target_dists:
            idx = np.searchsorted(cumulative_dist, target)
            if idx == 0:
                resampled.append(points[0])
            elif idx >= len(points):
                resampled.append(points[-1])
            else:
                t = (target - cumulative_dist[idx-1]) / (cumulative_dist[idx] - cumulative_dist[idx-1] + 1e-10)
                interp = points[idx-1] + t * (points[idx] - points[idx-1])
                resampled.append(interp)
        
        return np.array(resampled, dtype=np.int32)
    
    def extract_centerline_from_roi(self, roi_image):
        if len(roi_image.shape) == 3:
            gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi_image
        
        points, direction = self.auto_detect_centerline(gray)
        
        skeleton = np.zeros_like(gray, dtype=np.uint8)
        for pt in points:
            y, x = pt
            if 0 <= y < skeleton.shape[0] and 0 <= x < skeleton.shape[1]:
                skeleton[y, x] = 255
        
        vessel_mask = self.find_vessel_mask(gray)
        
        return points, skeleton, vessel_mask
    
    def get_intensity_profile_along_centerline(self, image, centerline_points):
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        profile = []
        for point in centerline_points:
            y, x = int(point[0]), int(point[1])
            if 0 <= y < gray.shape[0] and 0 <= x < gray.shape[1]:
                profile.append(gray[y, x])
            else:
                profile.append(0)
        
        return np.array(profile, dtype=np.uint8)
    
    def track_centerline(self, prev_frame, curr_frame, centerline_points, 
                          frame_count=0, original_centerline=None, redetect_interval=30,
                          cumulative_displacement=(0, 0), original_intensity_profile=None,
                          roi_offset=(0, 0)):
        if len(curr_frame.shape) == 3:
            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        else:
            curr_gray = curr_frame
        
        # Orijinal centerline'ı baz al
        if original_centerline is not None and len(original_centerline) > 0:
            base_centerline = original_centerline.copy()
        else:
            base_centerline = centerline_points.copy()
        
        # ROI offset'i uygula (ROI hareket ettiyse centerline de hareket etmeli)
        offset_y, offset_x = roi_offset
        if offset_x != 0 or offset_y != 0:
            base_centerline = base_centerline.copy()
            # Offset sınırlandır
            base_centerline[:, 0] = np.clip(base_centerline[:, 0] + offset_y, 0, curr_gray.shape[0] - 1)
            base_centerline[:, 1] = np.clip(base_centerline[:, 1] + offset_x, 0, curr_gray.shape[1] - 1)
        
        # Sadece minimal düzeltme yap (damar merkezine)
        corrected_points = self._precise_vessel_center(curr_gray, base_centerline)
        
        # Yoğunluk profili doğrulaması (threshold artırıldı: 0.3 -> 0.55)
        if original_intensity_profile is not None:
            if not self._validate_centerline(curr_gray, corrected_points, original_intensity_profile, threshold=0.55):
                # Doğrulama başarısız - orijinale dön (offset ile)
                return base_centerline.copy(), True, (0, 0), (0, 0)
        
        return corrected_points, False, (0, 0), (0, 0)
    
    def _precise_vessel_center(self, gray_image, centerline_points, search_radius=2):
        # Dengeli düzeltme - gerektiğinde damar merkezine hizala
        corrected_points = []
        height, width = gray_image.shape
        
        inverted = 255 - gray_image
        blurred = cv2.GaussianBlur(inverted, (3, 3), 0)
        
        for pt in centerline_points:
            y, x = int(pt[0]), int(pt[1])
            
            if not (0 <= y < height and 0 <= x < width):
                corrected_points.append([y, x])
                continue
                
            original_val = blurred[y, x]
            
            best_x = x
            best_y = y
            best_val = original_val
            
            # %30 daha iyi olan noktaları kabul et
            min_improvement = 1.3
            
            for dy in range(-search_radius, search_radius + 1):
                for dx in range(-search_radius, search_radius + 1):
                    if dx == 0 and dy == 0:
                        continue
                    new_x = x + dx
                    new_y = y + dy
                    if 0 <= new_x < width and 0 <= new_y < height:
                        val = blurred[new_y, new_x]
                        if val > best_val * min_improvement:
                            best_val = val
                            best_x = new_x
                            best_y = new_y
            
            corrected_points.append([best_y, best_x])
        
        corrected = np.array(corrected_points, dtype=np.int32)
        
        # Smooth uygula
        if len(corrected) > 5:
            corrected = self._smooth_centerline_curve(corrected, window_size=3)
        
        return corrected
    
    def _validate_centerline(self, gray_image, centerline_points, original_profile=None, threshold=0.5):
        current_profile = self.get_intensity_profile_along_centerline(gray_image, centerline_points)
        
        if original_profile is None:
            mean_intensity = np.mean(current_profile)
            return mean_intensity < 180
        
        if len(current_profile) != len(original_profile):
            min_len = min(len(current_profile), len(original_profile))
            current_profile = current_profile[:min_len]
            original_profile = original_profile[:min_len]
        
        if len(current_profile) == 0:
            return False
        
        correlation = np.corrcoef(current_profile.astype(float), original_profile.astype(float))[0, 1]
        
        if np.isnan(correlation):
            return True
        
        return correlation > threshold
    
    def get_intensity_profile_for_validation(self, gray_image, centerline_points):
        return self.get_intensity_profile_along_centerline(gray_image, centerline_points)
    
    def template_match_vessel(self, gray_image, template, centerline_points, search_range=20):
        if template is None or len(template) == 0:
            return centerline_points, 1.0
        
        height, width = gray_image.shape
        
        best_offset = 0
        best_score = -1
        
        for offset in range(-search_range, search_range + 1):
            test_points = centerline_points.copy()
            test_points[:, 1] += offset
            test_points[:, 1] = np.clip(test_points[:, 1], 0, width - 1)
            
            profile = self.get_intensity_profile_along_centerline(gray_image, test_points)
            
            if len(profile) == len(template):
                correlation = np.corrcoef(profile.astype(float), template.astype(float))[0, 1]
                if not np.isnan(correlation) and correlation > best_score:
                    best_score = correlation
                    best_offset = offset
        
        if best_score > 0.3 and best_offset != 0:
            matched_points = centerline_points.copy()
            matched_points[:, 1] += best_offset
            matched_points[:, 1] = np.clip(matched_points[:, 1], 0, width - 1)
            return matched_points, best_score
        
        return centerline_points, best_score if best_score > 0 else 0.0
    
    def track_roi(self, prev_frame, curr_frame, roi, original_roi=None, max_drift=35,
                   vessel_direction='vertical', centerline_points=None):
        """
        OPTICAL FLOW MOTION COMPENSATION SİSTEMİ

        Damarlar için optimize edilmiş takip:
        1. Dense optical flow hesapla
        2. Global motion'ı çıkar (kamera/doku hareketi)
        3. ROI çevresindeki dominant motion'a göre takip et
        4. Periyodik reset ile drift'i engelle

        Bu yöntem damar içindeki kan akışından etkilenmez!
        """
        x, y, w, h = roi

        # Frame'leri gray'e çevir
        if len(prev_frame.shape) == 3:
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        else:
            prev_gray = prev_frame

        if len(curr_frame.shape) == 3:
            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        else:
            curr_gray = curr_frame

        frame_h, frame_w = curr_gray.shape
        self.frame_counter += 1

        # İLK ÇAĞRI: Reference kaydet
        if self.original_roi is None:
            self.original_roi = original_roi if original_roi else roi
            self.reference_roi_image = prev_gray[y:y+h, x:x+w].copy()

        # İlk frame'de tracking yapma (prev == curr)
        if self.frame_counter <= 1:
            return roi, (0, 0)

        # PERİYODİK RESET: Drift'i sıfırla
        if (self.frame_counter - self.last_reset_frame) >= self.reset_interval:
            self.accumulated_offset *= 0.7  # %30 azalt
            self.last_reset_frame = self.frame_counter

        # ==================== OPTICAL FLOW HESAPLAMA ====================
        # ROI çevresinde daha geniş alan al (daha fazla context)
        margin = 25  # Artırıldı: 15 -> 25
        y1 = max(0, y - margin)
        x1 = max(0, x - margin)
        y2 = min(frame_h, y + h + margin)
        x2 = min(frame_w, x + w + margin)

        prev_region = prev_gray[y1:y2, x1:x2]
        curr_region = curr_gray[y1:y2, x1:x2]

        if prev_region.shape[0] < 30 or prev_region.shape[1] < 30:
            return roi, (0, 0)

        # Dense optical flow (Farneback) - Optimize edilmiş parametreler
        flow = cv2.calcOpticalFlowFarneback(
            prev_region, curr_region, None,
            pyr_scale=0.5,          # Pyramid scale
            levels=4,               # Artırıldı: 3 -> 4 (daha multi-scale)
            winsize=21,             # Artırıldı: 15 -> 21 (daha stabil)
            iterations=5,           # Artırıldı: 3 -> 5 (daha doğru)
            poly_n=7,              # Artırıldı: 5 -> 7 (daha smooth)
            poly_sigma=1.5,         # Polynomial sigma
            flags=0
        )

        # ==================== GLOBAL MOTION REMOVAL ====================
        # Tüm bölgenin median flow'u = kamera/doku hareketi
        median_flow_x = np.median(flow[..., 0])
        median_flow_y = np.median(flow[..., 1])

        # Global motion'ı çıkar
        compensated_flow = flow.copy()
        compensated_flow[..., 0] -= median_flow_x
        compensated_flow[..., 1] -= median_flow_y

        # ==================== ROI MOTION ESTIMATION ====================
        # ROI'nin merkezinde dominant motion'a bak
        roi_center_y = margin
        roi_center_x = margin
        roi_h_in_region = min(h, compensated_flow.shape[0] - roi_center_y)
        roi_w_in_region = min(w, compensated_flow.shape[1] - roi_center_x)

        if roi_h_in_region > 0 and roi_w_in_region > 0:
            roi_flow = compensated_flow[
                roi_center_y:roi_center_y+roi_h_in_region,
                roi_center_x:roi_center_x+roi_w_in_region
            ]

            # ROI içindeki median flow = ROI'nin hareketi
            roi_motion_x = np.median(roi_flow[..., 0])
            roi_motion_y = np.median(roi_flow[..., 1])
        else:
            roi_motion_x = 0
            roi_motion_y = 0

        # ==================== MOTION FILTERING ====================
        # Çok küçük hareketleri yoksay (noise)
        if abs(roi_motion_x) < 0.3:  # Threshold düşürüldü: 0.5 -> 0.3
            roi_motion_x = 0
        if abs(roi_motion_y) < 0.3:
            roi_motion_y = 0

        # Çok büyük ani hareketleri sınırla
        max_motion = 8  # Artırıldı: 5 -> 8 (daha hızlı hareket için)
        roi_motion_x = np.clip(roi_motion_x, -max_motion, max_motion)
        roi_motion_y = np.clip(roi_motion_y, -max_motion, max_motion)

        # Accumulated offset güncelle
        self.accumulated_offset[0] += roi_motion_x
        self.accumulated_offset[1] += roi_motion_y

        # ==================== YENİ ROI HESAPLAMA ====================
        new_x = x + int(roi_motion_x)
        new_y = y + int(roi_motion_y)

        # Sınırları kontrol et
        new_x = max(0, min(new_x, frame_w - w))
        new_y = max(0, min(new_y, frame_h - h))

        # ==================== DRIFT KONTROLÜ ====================
        if original_roi is not None or self.original_roi is not None:
            ref_roi = original_roi if original_roi else self.original_roi
            orig_x, orig_y, _, _ = ref_roi

            # Toplam drift
            total_drift_x = abs(new_x - orig_x)
            total_drift_y = abs(new_y - orig_y)

            # Max drift aşıldı mı?
            if total_drift_x > max_drift or total_drift_y > max_drift:
                # Orijinale doğru smooth correction (daha yumuşak)
                correction_strength = 0.3  # Azaltıldı: 0.5 -> 0.3
                new_x = int(new_x - (new_x - orig_x) * correction_strength)
                new_y = int(new_y - (new_y - orig_y) * correction_strength)

                # Accumulated offset'i de düzelt
                self.accumulated_offset *= 0.6  # Daha yumuşak: 0.5 -> 0.6
                self.tracking_confidence *= 0.95  # Daha yumuşak: 0.9 -> 0.95
            else:
                # İyi gidiyor
                self.tracking_confidence = min(1.0, self.tracking_confidence * 1.02)

        dx = new_x - x
        dy = new_y - y

        return (new_x, new_y, w, h), (dx, dy)
    
    def _correct_to_vessel_center(self, gray_image, centerline_points, search_radius=5):
        corrected_points = []
        inverted = 255 - gray_image
        height, width = gray_image.shape
        
        for pt in centerline_points:
            y, x = int(pt[0]), int(pt[1])
            
            best_x = x
            best_val = 0
            
            for offset in range(-search_radius, search_radius + 1):
                new_x = x + offset
                if 0 <= new_x < width and 0 <= y < height:
                    val = inverted[y, new_x]
                    if val > best_val:
                        best_val = val
                        best_x = new_x
            
            corrected_points.append([y, best_x])
        
        corrected = np.array(corrected_points, dtype=np.int32)
        
        if len(corrected) > 5:
            corrected = self._smooth_centerline_curve(corrected, window_size=5)
        
        return corrected

