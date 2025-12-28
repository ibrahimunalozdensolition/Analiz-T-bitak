import cv2
import numpy as np

class VideoPreprocessor:
    def __init__(self):
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.background_mode = None
        self.mean_image = None
        self.reference_frame = None
        self.cumulative_transform = np.eye(2, 3, dtype=np.float64)
        
    def stabilize_frame(self, curr_frame, prev_frame, prev_pts=None):
        if prev_frame is None:
            return curr_frame, None
            
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY) if len(prev_frame.shape) == 3 else prev_frame
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY) if len(curr_frame.shape) == 3 else curr_frame
        
        if prev_pts is None:
            prev_pts = cv2.goodFeaturesToTrack(
                prev_gray,
                maxCorners=300,
                qualityLevel=0.005,
                minDistance=20,
                blockSize=5
            )
        
        if prev_pts is None or len(prev_pts) < 4:
            return curr_frame, None
        
        lk_params = dict(
            winSize=(21, 21),
            maxLevel=4,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.01)
        )
        
        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, prev_pts, None, **lk_params
        )
        
        valid_prev = prev_pts[status == 1]
        valid_curr = curr_pts[status == 1]
        
        if len(valid_prev) < 4:
            return curr_frame, None
        
        transform_matrix, inliers = cv2.estimateAffinePartial2D(
            valid_prev, valid_curr,
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0
        )
        
        if transform_matrix is None:
            return curr_frame, None
        
        h, w = curr_frame.shape[:2]
        stabilized = cv2.warpAffine(curr_frame, transform_matrix, (w, h),
                                     flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_REPLICATE)
        
        new_pts = cv2.goodFeaturesToTrack(
            curr_gray,
            maxCorners=300,
            qualityLevel=0.005,
            minDistance=20,
            blockSize=5
        )
        
        return stabilized, new_pts
    
    def stabilize_roi_frames(self, frames):
        if len(frames) == 0:
            return frames
        
        stabilized_frames = [frames[0]]
        
        ref_frame = frames[0]
        if len(ref_frame.shape) == 3:
            ref_gray = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2GRAY)
        else:
            ref_gray = ref_frame
        
        for i in range(1, len(frames)):
            curr_frame = frames[i]
            if len(curr_frame.shape) == 3:
                curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
            else:
                curr_gray = curr_frame
            
            try:
                warp_matrix = np.eye(2, 3, dtype=np.float32)
                criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-6)
                
                _, warp_matrix = cv2.findTransformECC(
                    ref_gray, curr_gray, warp_matrix,
                    cv2.MOTION_EUCLIDEAN, criteria,
                    inputMask=None, gaussFiltSize=5
                )
                
                h, w = curr_frame.shape[:2]
                stabilized = cv2.warpAffine(
                    curr_frame, warp_matrix, (w, h),
                    flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                    borderMode=cv2.BORDER_REPLICATE
                )
                stabilized_frames.append(stabilized)
            except cv2.error:
                stabilized_frames.append(curr_frame)
        
        return stabilized_frames
    
    def measure_quality(self, frame):
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        contrast = np.std(gray)
        
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        noise = np.std(laplacian)
        
        snr = contrast / (noise + 1e-10)
        
        return {
            'contrast': contrast,
            'noise': noise,
            'snr': snr
        }
    
    def detect_background_mode(self, frames, threshold_snr=2.0, min_frames=30):
        if len(frames) < min_frames:
            return 'frame_clahe'
        
        sample_indices = np.linspace(0, len(frames)-1, min(10, len(frames)), dtype=int)
        
        qualities = []
        for idx in sample_indices:
            q = self.measure_quality(frames[idx])
            qualities.append(q)
        
        avg_snr = np.mean([q['snr'] for q in qualities])
        
        if avg_snr > threshold_snr:
            self.background_mode = 'mean_clahe'
        else:
            self.background_mode = 'frame_clahe'
        
        return self.background_mode
    
    def compute_mean_image(self, frames):
        if len(frames) == 0:
            return None
        
        first_frame = frames[0]
        if len(first_frame.shape) == 3:
            h, w = first_frame.shape[:2]
        else:
            h, w = first_frame.shape
        
        accumulator = np.zeros((h, w), dtype=np.float64)
        
        for frame in frames:
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame
            accumulator += gray.astype(np.float64)
        
        self.mean_image = (accumulator / len(frames)).astype(np.uint8)
        return self.mean_image
    
    def apply_background_removal_mode1(self, frame):
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        if self.mean_image is not None:
            diff = cv2.absdiff(gray, self.mean_image)
            enhanced = self.clahe.apply(diff)
        else:
            enhanced = self.clahe.apply(gray)
        
        return enhanced
    
    def apply_background_removal_mode2(self, frame):
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        enhanced = self.clahe.apply(gray)
        return enhanced
    
    def apply_background_removal(self, frame):
        if self.background_mode == 'mean_clahe':
            return self.apply_background_removal_mode1(frame)
        else:
            return self.apply_background_removal_mode2(frame)
    
    def convert_to_optical_density(self, frame, I0=None):
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        if I0 is None:
            I0 = np.max(gray)
        
        if I0 == 0:
            I0 = 255
        
        normalized = gray.astype(np.float64) / I0
        normalized = np.clip(normalized, 1e-10, 1.0)
        
        od = -np.log(normalized)
        
        od_min = np.min(od)
        od_max = np.max(od)
        
        if od_max - od_min > 0:
            od_normalized = (od - od_min) / (od_max - od_min) * 255
        else:
            od_normalized = od * 255
        
        return od_normalized.astype(np.uint8)
    
    def check_and_apply_od(self, frame, contrast_threshold=30.0):
        quality = self.measure_quality(frame)
        
        if quality['contrast'] < contrast_threshold:
            return self.convert_to_optical_density(frame), True
        else:
            if len(frame.shape) == 3:
                return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), False
            return frame, False
    
    def preprocess_frame(self, frame, use_od_if_needed=True):
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        enhanced = self.apply_background_removal(gray)
        
        if use_od_if_needed:
            enhanced, od_used = self.check_and_apply_od(enhanced)
        
        return enhanced

