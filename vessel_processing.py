import cv2
import numpy as np
from scipy.spatial import KDTree

class VesselProcessor:
    def __init__(self):
        # Kalman Filter için state
        self.kalman = None
        self.kalman_initialized = False
        self.prev_position = None
        self.tracking_confidence = 1.0
        
    def _init_kalman_filter(self, initial_x, initial_y):
        """Kalman Filter başlat - pozisyon ve hız takibi için"""
        self.kalman = cv2.KalmanFilter(4, 2)  # 4 state (x, y, vx, vy), 2 measurement (x, y)
        
        # Transition matrix (pozisyon + hız modeli)
        self.kalman.transitionMatrix = np.array([
            [1, 0, 1, 0],  # x = x + vx
            [0, 1, 0, 1],  # y = y + vy
            [0, 0, 1, 0],  # vx = vx
            [0, 0, 0, 1]   # vy = vy
        ], dtype=np.float32)
        
        # Measurement matrix
        self.kalman.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)
        
        # Process noise
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        
        # Measurement noise
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5
        
        # Initial state
        self.kalman.statePre = np.array([[initial_x], [initial_y], [0], [0]], dtype=np.float32)
        self.kalman.statePost = np.array([[initial_x], [initial_y], [0], [0]], dtype=np.float32)
        
        self.kalman_initialized = True
        
    def _kalman_predict(self):
        """Kalman ile sonraki pozisyonu tahmin et"""
        if not self.kalman_initialized:
            return None
        prediction = self.kalman.predict()
        return prediction[0, 0], prediction[1, 0]
    
    def _kalman_correct(self, measured_x, measured_y):
        """Kalman'ı ölçümle düzelt"""
        if not self.kalman_initialized:
            return measured_x, measured_y
        measurement = np.array([[measured_x], [measured_y]], dtype=np.float32)
        corrected = self.kalman.correct(measurement)
        return corrected[0, 0], corrected[1, 0]
    
    def reset_tracking_state(self):
        """Takip durumunu sıfırla"""
        self.kalman = None
        self.kalman_initialized = False
        self.prev_position = None
        self.tracking_confidence = 1.0
    
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
        
        # Yoğunluk profili doğrulaması
        if original_intensity_profile is not None:
            if not self._validate_centerline(curr_gray, corrected_points, original_intensity_profile, threshold=0.3):
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
    
    def track_roi(self, prev_frame, curr_frame, roi, original_roi=None, max_drift=15, 
                   vessel_direction='vertical', centerline_points=None):
        """
        GELİŞMİŞ HYBRİD TAKİP SİSTEMİ
        
        3 yöntem kombinasyonu:
        1. Template Matching - ana takip
        2. ECC Alignment - sub-pixel doğruluk  
        3. Kalman Filter - tahmin ve düzeltme
        """
        x, y, w, h = roi
        
        if len(prev_frame.shape) == 3:
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        else:
            prev_gray = prev_frame
        
        if len(curr_frame.shape) == 3:
            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        else:
            curr_gray = curr_frame
        
        frame_h, frame_w = curr_gray.shape
        
        # Kalman Filter'ı başlat (ilk çağrıda)
        if not self.kalman_initialized:
            self._init_kalman_filter(float(x), float(y))
        
        # Damar yönünü centerline'dan hesapla
        if centerline_points is not None and len(centerline_points) > 1:
            first_pt = centerline_points[0]
            last_pt = centerline_points[-1]
            delta_y = abs(last_pt[0] - first_pt[0])
            delta_x = abs(last_pt[1] - first_pt[1])
            vessel_direction = 'vertical' if delta_y > delta_x else 'horizontal'
        
        # ==================== YÖNTEM 1: TEMPLATE MATCHING ====================
        prev_roi = prev_gray[y:y+h, x:x+w]
        
        # Tüm yönlerde arama yap (vektör yönü sonra kontrol edilecek)
        search_margin = 5
        search_x1 = max(0, x - search_margin)
        search_y1 = max(0, y - search_margin)
        search_x2 = min(frame_w, x + w + search_margin)
        search_y2 = min(frame_h, y + h + search_margin)
        
        search_area = curr_gray[search_y1:search_y2, search_x1:search_x2]
        
        if search_area.shape[0] < h or search_area.shape[1] < w:
            return roi, (0, 0)
        
        result = cv2.matchTemplate(search_area, prev_roi, cv2.TM_CCOEFF_NORMED)
        _, template_confidence, _, max_loc = cv2.minMaxLoc(result)
        
        template_x = float(search_x1 + max_loc[0])
        template_y = float(search_y1 + max_loc[1])
        template_dx = template_x - x
        template_dy = template_y - y
        
        # ==================== YÖNTEM 2: ECC ALIGNMENT ====================
        ecc_dx, ecc_dy = 0.0, 0.0
        ecc_confidence = 0.0
        
        try:
            warp_matrix = np.eye(2, 3, dtype=np.float32)
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 1e-3)
            curr_roi = curr_gray[y:y+h, x:x+w]
            
            if prev_roi.shape == curr_roi.shape and prev_roi.size > 0:
                _, warp_matrix = cv2.findTransformECC(
                    prev_roi.astype(np.float32), 
                    curr_roi.astype(np.float32),
                    warp_matrix, 
                    cv2.MOTION_TRANSLATION,
                    criteria
                )
                ecc_dx = -warp_matrix[0, 2]  # ECC ters yönde
                ecc_dy = -warp_matrix[1, 2]
                ecc_confidence = 0.8
        except:
            ecc_confidence = 0.0
        
        # ==================== YÖNTEM 3: KALMAN TAHMİN ====================
        kalman_pred = self._kalman_predict()
        kalman_dx, kalman_dy = 0.0, 0.0
        kalman_confidence = 0.0
        
        if kalman_pred and self.prev_position is not None:
            pred_x, pred_y = kalman_pred
            kalman_dx = pred_x - self.prev_position[0]
            kalman_dy = pred_y - self.prev_position[1]
            kalman_confidence = 0.6
        
        # ==================== AĞIRLIKLI ORTALAMA ====================
        total_weight = template_confidence + ecc_confidence + kalman_confidence
        
        if total_weight < 0.3:
            return roi, (0, 0)
        
        # Ağırlıklı hareket
        weighted_dx = (template_dx * template_confidence + 
                       ecc_dx * ecc_confidence + 
                       kalman_dx * kalman_confidence) / total_weight
        weighted_dy = (template_dy * template_confidence + 
                       ecc_dy * ecc_confidence + 
                       kalman_dy * kalman_confidence) / total_weight
        
        # Kan akışı yönündeki hareketi AZALT (tamamen yoksayma)
        if vessel_direction == 'vertical':
            weighted_dy *= 0.2  # Y hareketi %20'ye düşür
        else:
            weighted_dx *= 0.2  # X hareketi %20'ye düşür
        
        # Hareket çok küçükse
        if abs(weighted_dx) < 0.3 and abs(weighted_dy) < 0.3:
            return roi, (0, 0)
        
        # Yeni pozisyon
        new_x = x + weighted_dx
        new_y = y + weighted_dy
        
        # ==================== KALMAN DÜZELTME ====================
        corrected_x, corrected_y = self._kalman_correct(new_x, new_y)
        self.prev_position = (corrected_x, corrected_y)
        
        # Tek frame'de maksimum hareket (3 piksel)
        final_dx = corrected_x - x
        final_dy = corrected_y - y
        max_frame_displacement = 3
        
        if abs(final_dx) > max_frame_displacement:
            final_dx = max_frame_displacement if final_dx > 0 else -max_frame_displacement
            corrected_x = x + final_dx
        if abs(final_dy) > max_frame_displacement:
            final_dy = max_frame_displacement if final_dy > 0 else -max_frame_displacement
            corrected_y = y + final_dy
        
        # Orijinal ROI'den maksimum sapma kontrolü
        if original_roi is not None:
            orig_x, orig_y, _, _ = original_roi
            total_drift_x = abs(corrected_x - orig_x)
            total_drift_y = abs(corrected_y - orig_y)
            
            if total_drift_x > max_drift or total_drift_y > max_drift:
                self.reset_tracking_state()
                self._init_kalman_filter(float(orig_x), float(orig_y))
                self.prev_position = (float(orig_x), float(orig_y))
                return original_roi, (0, 0)
        
        final_x = int(np.clip(corrected_x, 0, frame_w - w))
        final_y = int(np.clip(corrected_y, 0, frame_h - h))
        
        # Güven skorunu güncelle
        self.tracking_confidence = min(1.0, total_weight / 2.0)
        
        return (final_x, final_y, w, h), (final_dx, final_dy)
    
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

