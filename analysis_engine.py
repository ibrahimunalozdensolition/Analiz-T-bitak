import cv2
import numpy as np
from preprocessing import VideoPreprocessor
from vessel_processing import VesselProcessor
from raft_stabilizer import RAFTStabilizer, RAFTVesselTracker

class SpaceTimeDiagramAnalyzer:
    def __init__(self, fps, pixel_to_um=1.832):
        self.fps = fps
        self.pixel_to_um = pixel_to_um
        self.preprocessor = VideoPreprocessor()
        self.vessel_processor = VesselProcessor()
        self.raft_tracker = None
        self.std_image = None
        self.centerline_points = None
        self.analysis_info = {}
        
    def create_space_time_diagram(self, frames, centerline_points, track_vessel=True, roi_offsets=None):
        if len(frames) == 0 or len(centerline_points) == 0:
            return None
        
        num_frames = len(frames)
        num_points = len(centerline_points)
        
        std_image = np.zeros((num_points, num_frames), dtype=np.uint8)
        
        current_centerline = centerline_points.copy()
        original_centerline = centerline_points.copy()
        prev_frame = None
        cumulative_displacement = (0, 0)
        original_intensity_profile = None
        
        for t, frame in enumerate(frames):
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame
            
            if t == 0:
                original_intensity_profile = self.vessel_processor.get_intensity_profile_for_validation(
                    gray, original_centerline
                )
            
            # ROI offset'ini al
            roi_offset = roi_offsets[t] if roi_offsets and t < len(roi_offsets) else (0, 0)
            
            if track_vessel and prev_frame is not None and t > 0:
                current_centerline, was_redetected, displacement, cumulative_displacement = self.vessel_processor.track_centerline(
                    prev_frame, gray, current_centerline,
                    frame_count=t,
                    original_centerline=original_centerline,
                    redetect_interval=30,
                    cumulative_displacement=cumulative_displacement,
                    original_intensity_profile=original_intensity_profile,
                    roi_offset=roi_offset
                )
                
                if was_redetected:
                    cumulative_displacement = (0, 0)
            
            profile = self.vessel_processor.get_intensity_profile_along_centerline(
                gray, current_centerline
            )
            
            if len(profile) == num_points:
                std_image[:, t] = profile
            
            prev_frame = gray.copy()
        
        self.std_image = std_image
        return std_image
    
    def remove_global_motion_from_std(self, std_image):
        if std_image is None:
            return std_image
        
        compensated = std_image.astype(np.float32).copy()
        height, width = compensated.shape
        
        if width < 3:
            return std_image
        
        col_means = np.mean(compensated, axis=0)
        
        global_mean = np.mean(col_means)
        
        for col in range(width):
            compensated[:, col] = compensated[:, col] - (col_means[col] - global_mean)
        
        compensated = np.clip(compensated, 0, 255).astype(np.uint8)
        
        row_shifts = []
        window_size = 5
        
        for col in range(1, width):
            prev_col = compensated[:, col-1].astype(np.float32)
            curr_col = compensated[:, col].astype(np.float32)
            
            best_shift = 0
            best_corr = -1
            
            for shift in range(-3, 4):
                if shift == 0:
                    shifted = curr_col
                elif shift > 0:
                    shifted = np.concatenate([np.zeros(shift), curr_col[:-shift]])
                else:
                    shifted = np.concatenate([curr_col[-shift:], np.zeros(-shift)])
                
                corr = np.corrcoef(prev_col, shifted)[0, 1]
                if not np.isnan(corr) and corr > best_corr:
                    best_corr = corr
                    best_shift = shift
            
            row_shifts.append(best_shift)
        
        if len(row_shifts) > 0:
            median_shift = np.median(row_shifts)
            if abs(median_shift) > 0.5:
                self.analysis_info['global_motion_detected'] = True
                self.analysis_info['median_row_shift'] = median_shift
        
        return compensated
    
    def detect_lines_hough(self, std_image, min_line_length=10, max_line_gap=8, 
                           hough_threshold=15):
        if std_image is None:
            return []
        
        compensated_std = self.remove_global_motion_from_std(std_image)
        
        blurred = cv2.GaussianBlur(compensated_std, (3, 3), 0)
        
        median_val = np.median(blurred)
        canny_low = int(max(10, median_val * 0.3))
        canny_high = int(min(200, median_val * 1.5))
        
        edges = cv2.Canny(blurred, canny_low, canny_high)
        
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=hough_threshold,
            minLineLength=min_line_length,
            maxLineGap=max_line_gap
        )
        
        if lines is None:
            return []
        
        return lines.reshape(-1, 4)
    
    def calculate_speed_from_line(self, x1, y1, x2, y2, return_direction=False):
        delta_time_frames = x2 - x1
        delta_space_pixels = y2 - y1
        
        if abs(delta_time_frames) == 0:
            if return_direction:
                return 0.0, 0
            return 0.0
        
        delta_time_seconds = abs(delta_time_frames) / self.fps
        delta_space_um = delta_space_pixels * self.pixel_to_um
        
        speed_um_per_second = delta_space_um / delta_time_seconds
        
        if return_direction:
            direction = 1 if delta_space_pixels > 0 else -1
            return abs(speed_um_per_second), direction
        
        return abs(speed_um_per_second)
    
    def analyze_all_lines(self, lines, min_slope=0.05, max_slope=20.0, min_speed=150.0):
        speeds = []
        valid_lines = []
        alias_lines = []
        directions = []
        
        positive_count = 0
        negative_count = 0
        
        for line in lines:
            x1, y1, x2, y2 = line
            
            delta_x = abs(x2 - x1)
            delta_y = abs(y2 - y1)
            
            if delta_x == 0:
                continue
            
            slope = delta_y / delta_x
            
            if slope < min_slope or slope > max_slope:
                continue
            
            speed, direction = self.calculate_speed_from_line(x1, y1, x2, y2, return_direction=True)
            
            if speed >= min_speed:
                speeds.append(speed)
                valid_lines.append(line)
                directions.append(direction)
                
                if direction > 0:
                    positive_count += 1
                else:
                    negative_count += 1
        
        if len(speeds) == 0:
            return speeds, valid_lines, alias_lines
        
        dominant_direction = 1 if positive_count >= negative_count else -1
        
        filtered_speeds = []
        filtered_lines = []
        
        for i, (speed, line, direction) in enumerate(zip(speeds, valid_lines, directions)):
            if direction == dominant_direction:
                filtered_speeds.append(speed)
                filtered_lines.append(line)
        
        self.analysis_info['dominant_direction'] = dominant_direction
        self.analysis_info['positive_count'] = positive_count
        self.analysis_info['negative_count'] = negative_count
        self.analysis_info['direction_ratio'] = max(positive_count, negative_count) / (positive_count + negative_count) if (positive_count + negative_count) > 0 else 0
        
        return filtered_speeds, filtered_lines, alias_lines
    
    def check_aliasing(self, speed, roi_height):
        max_displacement_per_frame = roi_height / 2
        max_speed = max_displacement_per_frame * self.pixel_to_um * self.fps
        
        if speed > max_speed:
            return True, max_speed
        return False, max_speed
    
    def analyze_video(self, video_path, roi, progress_callback=None, centerline_points=None):
        x, y, w, h = roi
        original_roi = roi
        current_roi = list(roi)
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        
        frames = []
        roi_offsets = []
        frame_idx = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        prev_frame = None
        prev_full_frame = None
        prev_pts = None
        
        if progress_callback:
            progress_callback(0, 100, "RAFT stabilizer baslatiliyor...")
        
        self.raft_tracker = RAFTVesselTracker()
        if centerline_points is not None:
            self.raft_tracker.set_centerline(centerline_points)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if prev_frame is not None:
                stabilized, prev_pts = self.preprocessor.stabilize_frame(
                    frame, prev_frame, prev_pts
                )
            else:
                stabilized = frame
            
            prev_frame = frame.copy()
            
            if len(stabilized.shape) == 3:
                curr_gray = cv2.cvtColor(stabilized, cv2.COLOR_BGR2GRAY)
            else:
                curr_gray = stabilized
            
            if prev_full_frame is not None:
                if len(prev_full_frame.shape) == 3:
                    prev_gray = cv2.cvtColor(prev_full_frame, cv2.COLOR_BGR2GRAY)
                else:
                    prev_gray = prev_full_frame
                
                new_roi, displacement, updated_centerline = self.raft_tracker.track(
                    prev_gray, curr_gray, tuple(current_roi)
                )
                current_roi = list(new_roi)
            
            prev_full_frame = stabilized.copy()
            
            offset_x = current_roi[0] - original_roi[0]
            offset_y = current_roi[1] - original_roi[1]
            roi_offsets.append((offset_y, offset_x))
            
            x, y, w, h = current_roi
            roi_frame = stabilized[y:y+h, x:x+w]
            
            frames.append(roi_frame)
            
            frame_idx += 1
            if progress_callback and frame_idx % 10 == 0:
                pct = int((frame_idx / total_frames) * 30)
                progress_callback(pct, 100, f"RAFT ile kareler okunuyor... {frame_idx}/{total_frames}")
        
        cap.release()
        
        if len(frames) == 0:
            return None
        
        if progress_callback:
            progress_callback(35, 100, "Frame stabilizasyonu yapiliyor...")
        
        frames = self.preprocessor.stabilize_roi_frames(frames)
        
        if progress_callback:
            progress_callback(45, 100, "Arka plan modu tespit ediliyor...")
        
        self.preprocessor.detect_background_mode(frames)
        self.analysis_info['background_mode'] = self.preprocessor.background_mode
        
        if self.preprocessor.background_mode == 'mean_clahe':
            self.preprocessor.compute_mean_image(frames)
        
        if progress_callback:
            progress_callback(50, 100, "Kareler isleniyor...")
        
        processed_frames = []
        for i, frame in enumerate(frames):
            processed = self.preprocessor.preprocess_frame(frame)
            processed_frames.append(processed)
            
            if progress_callback and i % 20 == 0:
                pct = 50 + int((i / len(frames)) * 20)
                progress_callback(pct, 100, f"Kareler isleniyor... {i+1}/{len(frames)}")
        
        if progress_callback:
            progress_callback(75, 100, "Damar merkez cizgisi cikariliyor...")
        
        sample_frame = processed_frames[len(processed_frames)//2]
        centerline_points, skeleton, vessel_mask = self.vessel_processor.extract_centerline_from_roi(sample_frame)
        
        if len(centerline_points) < 10:
            centerline_points = self._create_vertical_centerline(sample_frame.shape)
        
        self.centerline_points = centerline_points
        
        if progress_callback:
            progress_callback(85, 100, "STD olusturuluyor...")
        
        std_image = self.create_space_time_diagram(processed_frames, centerline_points, 
                                                    track_vessel=True, roi_offsets=roi_offsets)
        
        if std_image is None:
            return None
        
        if progress_callback:
            progress_callback(95, 100, "Hough Transform cizgi tespiti...")
        
        lines = self.detect_lines_hough(std_image)
        
        speeds, valid_lines, alias_lines = self.analyze_all_lines(lines)
        
        n_valid = 0
        n_alias = 0
        valid_speeds = []
        
        for speed in speeds:
            is_alias, nyquist_limit = self.check_aliasing(speed, h)
            if is_alias:
                n_alias += 1
            else:
                n_valid += 1
                valid_speeds.append(speed)
        
        if n_valid < 8:
            results = {
                'mean': 0,
                'median': 0,
                'std': 0,
                'min': 0,
                'max': 0,
                'p25': 0,
                'p50': 0,
                'p75': 0,
                'p90': 0,
                'p95': 0,
                'n_valid': n_valid,
                'n_alias': n_alias,
                'n_total': len(speeds),
                'all_speeds': [],
                'std_image': std_image,
                'valid_lines': valid_lines,
                'centerline': centerline_points,
                'analysis_info': self.analysis_info,
                'insufficient_data': True
            }
            return results
        
        results = self._compute_statistics(valid_speeds)
        
        results['n_valid'] = n_valid
        results['n_alias'] = n_alias
        results['n_total'] = len(speeds)
        results['all_speeds'] = valid_speeds
        results['std_image'] = std_image
        results['valid_lines'] = valid_lines
        results['centerline'] = centerline_points
        results['analysis_info'] = self.analysis_info
        results['insufficient_data'] = False
        
        return results
    
    def _create_vertical_centerline(self, frame_shape):
        height, width = frame_shape[:2]
        center_x = width // 2
        
        points = np.array([[y, center_x] for y in range(height)])
        return points
    
    def _compute_statistics(self, speeds):
        if len(speeds) == 0:
            return {
                'mean': 0,
                'median': 0,
                'std': 0,
                'min': 0,
                'max': 0,
                'p25': 0,
                'p50': 0,
                'p75': 0,
                'p90': 0,
                'p95': 0
            }
        
        speeds_arr = np.array(speeds)
        
        return {
            'mean': float(np.mean(speeds_arr)),
            'median': float(np.median(speeds_arr)),
            'std': float(np.std(speeds_arr)),
            'min': float(np.min(speeds_arr)),
            'max': float(np.max(speeds_arr)),
            'p25': float(np.percentile(speeds_arr, 25)),
            'p50': float(np.percentile(speeds_arr, 50)),
            'p75': float(np.percentile(speeds_arr, 75)),
            'p90': float(np.percentile(speeds_arr, 90)),
            'p95': float(np.percentile(speeds_arr, 95))
        }
    
    def visualize_std_with_lines(self, std_image, lines):
        if std_image is None:
            return None
        
        vis = cv2.cvtColor(std_image, cv2.COLOR_GRAY2BGR)
        
        for line in lines:
            x1, y1, x2, y2 = line
            cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 255), 1)
        
        return vis

