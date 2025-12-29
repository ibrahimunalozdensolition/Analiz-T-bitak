import cv2
import numpy as np
import os

class SAMVesselSegmenter:
    def __init__(self, model_type="vit_b"):
        self.model_type = model_type
        self.sam = None
        self.predictor = None
        self.device = "cuda" if self._check_cuda() else "cpu"
        self.model_loaded = False
        
    def _check_cuda(self):
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False
    
    def load_model(self, checkpoint_path=None):
        try:
            import torch
            from segment_anything import sam_model_registry, SamPredictor
            
            if checkpoint_path is None:
                checkpoint_path = self._download_checkpoint()
            
            if checkpoint_path is None or not os.path.exists(checkpoint_path):
                print("SAM checkpoint bulunamadi")
                return False
            
            self.sam = sam_model_registry[self.model_type](checkpoint=checkpoint_path)
            self.sam.to(device=self.device)
            self.predictor = SamPredictor(self.sam)
            self.model_loaded = True
            print(f"SAM yuklendi: {self.model_type} on {self.device}")
            return True
            
        except ImportError as e:
            print(f"SAM yuklenemedi: {e}")
            print("Lutfen 'pip install segment-anything torch torchvision' calistirin")
            return False
        except Exception as e:
            print(f"SAM yukleme hatasi: {e}")
            return False
    
    def _download_checkpoint(self):
        checkpoints = {
            "vit_h": "sam_vit_h_4b8939.pth",
            "vit_l": "sam_vit_l_0b3195.pth",
            "vit_b": "sam_vit_b_01ec64.pth"
        }
        
        checkpoint_name = checkpoints.get(self.model_type, "sam_vit_b_01ec64.pth")
        
        possible_paths = [
            os.path.join(os.path.dirname(__file__), checkpoint_name),
            os.path.join(os.path.dirname(__file__), "models", checkpoint_name),
            os.path.join(os.path.expanduser("~"), ".cache", "sam", checkpoint_name),
            checkpoint_name
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        print(f"SAM checkpoint bulunamadi. Lutfen {checkpoint_name} dosyasini indirin:")
        print(f"https://dl.fbaipublicfiles.com/segment_anything/{checkpoint_name}")
        return None
    
    def segment_vessel_with_points(self, image, points, point_labels=None):
        if not self.model_loaded:
            if not self.load_model():
                return None, None
        
        if len(image.shape) == 2:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        self.predictor.set_image(image_rgb)
        
        points_np = np.array(points)
        
        if point_labels is None:
            point_labels = np.ones(len(points), dtype=np.int32)
        else:
            point_labels = np.array(point_labels, dtype=np.int32)
        
        masks, scores, logits = self.predictor.predict(
            point_coords=points_np,
            point_labels=point_labels,
            multimask_output=True
        )
        
        best_idx = np.argmax(scores)
        best_mask = masks[best_idx]
        best_score = scores[best_idx]
        
        return best_mask.astype(np.uint8) * 255, best_score
    
    def segment_vessel_with_box(self, image, box):
        if not self.model_loaded:
            if not self.load_model():
                return None, None
        
        if len(image.shape) == 2:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        self.predictor.set_image(image_rgb)
        
        box_np = np.array(box)
        
        masks, scores, logits = self.predictor.predict(
            box=box_np,
            multimask_output=True
        )
        
        best_idx = np.argmax(scores)
        best_mask = masks[best_idx]
        best_score = scores[best_idx]
        
        return best_mask.astype(np.uint8) * 255, best_score
    
    def segment_vessel_auto(self, image, roi=None):
        if not self.model_loaded:
            if not self.load_model():
                return None, None
        
        if len(image.shape) == 2:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if roi is not None:
            x, y, w, h = roi
            roi_image = image_rgb[y:y+h, x:x+w]
        else:
            roi_image = image_rgb
            x, y = 0, 0
        
        self.predictor.set_image(roi_image)
        
        h, w = roi_image.shape[:2]
        center_point = np.array([[w // 2, h // 2]])
        point_labels = np.array([1])
        
        masks, scores, logits = self.predictor.predict(
            point_coords=center_point,
            point_labels=point_labels,
            multimask_output=True
        )
        
        best_idx = np.argmax(scores)
        best_mask = masks[best_idx]
        best_score = scores[best_idx]
        
        return best_mask.astype(np.uint8) * 255, best_score
    
    def extract_centerline_from_mask(self, mask):
        kernel = np.ones((3, 3), np.uint8)
        cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)
        
        skeleton = self._skeletonize(cleaned)
        
        points = np.column_stack(np.where(skeleton > 0))
        
        if len(points) == 0:
            return np.array([])
        
        sorted_points = self._sort_skeleton_points(points)
        
        if len(sorted_points) > 5:
            sorted_points = self._smooth_points(sorted_points)
        
        return sorted_points
    
    def _skeletonize(self, binary_image):
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
    
    def _sort_skeleton_points(self, points):
        if len(points) < 2:
            return points
        
        y_range = np.max(points[:, 0]) - np.min(points[:, 0])
        x_range = np.max(points[:, 1]) - np.min(points[:, 1])
        
        if y_range >= x_range:
            sort_axis = 0
        else:
            sort_axis = 1
        
        sorted_indices = np.argsort(points[:, sort_axis])
        return points[sorted_indices]
    
    def _smooth_points(self, points, window_size=5):
        if len(points) < window_size:
            return points
        
        smoothed = np.zeros_like(points, dtype=np.float64)
        half_window = window_size // 2
        
        for i in range(len(points)):
            start = max(0, i - half_window)
            end = min(len(points), i + half_window + 1)
            smoothed[i] = np.mean(points[start:end], axis=0)
        
        return smoothed.astype(np.int32)


class LightweightVesselSegmenter:
    def __init__(self):
        pass
    
    def segment_vessel_adaptive(self, image, roi=None):
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        if roi is not None:
            x, y, w, h = roi
            gray = gray[y:y+h, x:x+w]
        
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(blurred)
        
        inverted = 255 - enhanced
        
        thresh = cv2.adaptiveThreshold(
            inverted, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 15, 3
        )
        
        kernel = np.ones((3, 3), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)
        
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            return None
        
        largest_contour = max(contours, key=cv2.contourArea)
        
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [largest_contour], -1, 255, -1)
        
        return mask
    
    def segment_vessel_frangi(self, image, roi=None):
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        if roi is not None:
            x, y, w, h = roi
            gray = gray[y:y+h, x:x+w]
        
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        scales = [1, 2, 3]
        vesselness = np.zeros_like(gray, dtype=np.float64)
        
        for scale in scales:
            sigma = scale
            
            Ixx = cv2.Sobel(blurred, cv2.CV_64F, 2, 0, ksize=3)
            Iyy = cv2.Sobel(blurred, cv2.CV_64F, 0, 2, ksize=3)
            Ixy = cv2.Sobel(blurred, cv2.CV_64F, 1, 1, ksize=3)
            
            Ixx = cv2.GaussianBlur(Ixx, (0, 0), sigma)
            Iyy = cv2.GaussianBlur(Iyy, (0, 0), sigma)
            Ixy = cv2.GaussianBlur(Ixy, (0, 0), sigma)
            
            trace = Ixx + Iyy
            det = Ixx * Iyy - Ixy * Ixy
            
            discriminant = np.sqrt(np.maximum(trace**2 - 4*det, 0))
            lambda1 = (trace + discriminant) / 2
            lambda2 = (trace - discriminant) / 2
            
            Rb = np.abs(lambda2) / (np.abs(lambda1) + 1e-10)
            S = np.sqrt(lambda1**2 + lambda2**2)
            
            beta = 0.5
            c = 0.5 * np.max(S)
            
            vessel_response = np.exp(-Rb**2 / (2 * beta**2)) * (1 - np.exp(-S**2 / (2 * c**2 + 1e-10)))
            vessel_response[lambda1 > 0] = 0
            
            vesselness = np.maximum(vesselness, vessel_response)
        
        vesselness = (vesselness / (np.max(vesselness) + 1e-10) * 255).astype(np.uint8)
        
        _, mask = cv2.threshold(vesselness, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return mask
    
    def extract_centerline(self, mask):
        kernel = np.ones((3, 3), np.uint8)
        cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        size = np.size(cleaned)
        skel = np.zeros(cleaned.shape, np.uint8)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        img = cleaned.copy()
        
        while True:
            eroded = cv2.erode(img, element)
            temp = cv2.dilate(eroded, element)
            temp = cv2.subtract(img, temp)
            skel = cv2.bitwise_or(skel, temp)
            img = eroded.copy()
            
            if cv2.countNonZero(img) == 0:
                break
        
        points = np.column_stack(np.where(skel > 0))
        
        if len(points) == 0:
            return np.array([])
        
        y_range = np.max(points[:, 0]) - np.min(points[:, 0])
        x_range = np.max(points[:, 1]) - np.min(points[:, 1])
        
        if y_range >= x_range:
            sorted_indices = np.argsort(points[:, 0])
        else:
            sorted_indices = np.argsort(points[:, 1])
        
        sorted_points = points[sorted_indices]
        
        if len(sorted_points) > 5:
            smoothed = np.zeros_like(sorted_points, dtype=np.float64)
            window = 5
            half = window // 2
            for i in range(len(sorted_points)):
                start = max(0, i - half)
                end = min(len(sorted_points), i + half + 1)
                smoothed[i] = np.mean(sorted_points[start:end], axis=0)
            sorted_points = smoothed.astype(np.int32)
        
        return sorted_points


class SAMLiveTracker:
    def __init__(self, model_type="vit_b", redetect_interval=5):
        self.segmenter = SAMVesselSegmenter(model_type=model_type)
        self.redetect_interval = redetect_interval
        self.frame_count = 0
        self.current_mask = None
        self.current_centerline = None
        self.reference_points = None
        self.model_loaded = False
        self.last_logits = None
        self.tracking_active = False
        
    def initialize(self, image, initial_points=None, initial_mask=None):
        if not self.segmenter.model_loaded:
            if not self.segmenter.load_model():
                return False
        
        self.model_loaded = True
        self.frame_count = 0
        
        if initial_mask is not None:
            self.current_mask = initial_mask
            self.current_centerline = self.segmenter.extract_centerline_from_mask(initial_mask)
            self.reference_points = self._get_points_from_mask(initial_mask)
        elif initial_points is not None:
            self.reference_points = initial_points
            mask, score = self.segmenter.segment_vessel_with_points(image, initial_points)
            if mask is not None:
                self.current_mask = mask
                self.current_centerline = self.segmenter.extract_centerline_from_mask(mask)
        else:
            return False
        
        self.tracking_active = True
        return True
    
    def _get_points_from_mask(self, mask, num_points=5):
        points = np.column_stack(np.where(mask > 0))
        if len(points) == 0:
            return None
        
        indices = np.linspace(0, len(points) - 1, num_points, dtype=int)
        selected = points[indices]
        
        return [[int(p[1]), int(p[0])] for p in selected]
    
    def track_frame(self, image):
        if not self.tracking_active or not self.model_loaded:
            return None, None, False
        
        self.frame_count += 1
        
        do_full_inference = (self.frame_count % self.redetect_interval == 0)
        
        if do_full_inference:
            return self._full_sam_inference(image)
        else:
            return self._propagate_mask(image)
    
    def _full_sam_inference(self, image):
        if self.reference_points is None and self.current_mask is not None:
            self.reference_points = self._get_points_from_mask(self.current_mask)
        
        if self.reference_points is None:
            return self.current_mask, self.current_centerline, False
        
        if len(image.shape) == 2:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        self.segmenter.predictor.set_image(image_rgb)
        
        points_np = np.array(self.reference_points)
        point_labels = np.ones(len(self.reference_points), dtype=np.int32)
        
        if self.current_mask is not None:
            mask_input = cv2.resize(self.current_mask, (256, 256), interpolation=cv2.INTER_NEAREST)
            mask_input = (mask_input > 127).astype(np.float32)[np.newaxis, :, :]
        else:
            mask_input = None
        
        try:
            if mask_input is not None:
                masks, scores, logits = self.segmenter.predictor.predict(
                    point_coords=points_np,
                    point_labels=point_labels,
                    mask_input=mask_input,
                    multimask_output=False
                )
            else:
                masks, scores, logits = self.segmenter.predictor.predict(
                    point_coords=points_np,
                    point_labels=point_labels,
                    multimask_output=True
                )
                best_idx = np.argmax(scores)
                masks = masks[best_idx:best_idx+1]
                logits = logits[best_idx:best_idx+1]
            
            self.current_mask = masks[0].astype(np.uint8) * 255
            self.last_logits = logits
            self.current_centerline = self.segmenter.extract_centerline_from_mask(self.current_mask)
            
            self.reference_points = self._get_points_from_mask(self.current_mask)
            
            return self.current_mask, self.current_centerline, True
            
        except Exception as e:
            print(f"SAM inference hatasi: {e}")
            return self.current_mask, self.current_centerline, False
    
    def _propagate_mask(self, image):
        if self.current_mask is None:
            return self._full_sam_inference(image)
        
        if len(image.shape) == 2:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        self.segmenter.predictor.set_image(image_rgb)
        
        mask_input = cv2.resize(self.current_mask, (256, 256), interpolation=cv2.INTER_NEAREST)
        mask_input = (mask_input > 127).astype(np.float32)[np.newaxis, :, :]
        
        try:
            masks, scores, logits = self.segmenter.predictor.predict(
                point_coords=None,
                point_labels=None,
                mask_input=mask_input,
                multimask_output=False
            )
            
            self.current_mask = masks[0].astype(np.uint8) * 255
            self.last_logits = logits
            self.current_centerline = self.segmenter.extract_centerline_from_mask(self.current_mask)
            
            return self.current_mask, self.current_centerline, False
            
        except Exception as e:
            print(f"Mask propagation hatasi: {e}")
            return self.current_mask, self.current_centerline, False
    
    def update_reference_points(self, points):
        self.reference_points = points
    
    def reset(self):
        self.frame_count = 0
        self.current_mask = None
        self.current_centerline = None
        self.reference_points = None
        self.last_logits = None
        self.tracking_active = False
    
    def get_current_state(self):
        return {
            'mask': self.current_mask,
            'centerline': self.current_centerline,
            'frame_count': self.frame_count,
            'tracking_active': self.tracking_active
        }


class HybridVesselTracker:
    def __init__(self, use_sam=True, sam_interval=10):
        self.use_sam = use_sam
        self.sam_interval = sam_interval
        self.sam_tracker = None
        self.frame_count = 0
        self.current_mask = None
        self.current_centerline = None
        self.prev_frame = None
        self.initialized = False
        
    def initialize(self, image, initial_points=None, initial_mask=None):
        self.frame_count = 0
        self.prev_frame = image.copy()
        
        if self.use_sam:
            self.sam_tracker = SAMLiveTracker(redetect_interval=self.sam_interval)
            success = self.sam_tracker.initialize(image, initial_points, initial_mask)
            if success:
                self.current_mask = self.sam_tracker.current_mask
                self.current_centerline = self.sam_tracker.current_centerline
                self.initialized = True
                return True
        
        if initial_mask is not None:
            self.current_mask = initial_mask
            segmenter = LightweightVesselSegmenter()
            self.current_centerline = segmenter.extract_centerline(initial_mask)
            self.initialized = True
            return True
        
        return False
    
    def track_frame(self, image):
        if not self.initialized:
            return None, None, False
        
        self.frame_count += 1
        
        if self.use_sam and self.sam_tracker is not None:
            mask, centerline, was_full_inference = self.sam_tracker.track_frame(image)
            if mask is not None:
                self.current_mask = mask
                self.current_centerline = centerline
                self.prev_frame = image.copy()
                return mask, centerline, was_full_inference
        
        if self.prev_frame is not None and self.current_mask is not None:
            mask, centerline = self._optical_flow_propagate(image)
            self.prev_frame = image.copy()
            return mask, centerline, False
        
        self.prev_frame = image.copy()
        return self.current_mask, self.current_centerline, False
    
    def _optical_flow_propagate(self, image):
        if len(self.prev_frame.shape) == 3:
            prev_gray = cv2.cvtColor(self.prev_frame, cv2.COLOR_BGR2GRAY)
        else:
            prev_gray = self.prev_frame
        
        if len(image.shape) == 3:
            curr_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            curr_gray = image
        
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        
        median_dx = np.median(flow[..., 0])
        median_dy = np.median(flow[..., 1])
        
        if abs(median_dx) > 0.5 or abs(median_dy) > 0.5:
            M = np.float32([[1, 0, median_dx], [0, 1, median_dy]])
            h, w = self.current_mask.shape
            self.current_mask = cv2.warpAffine(self.current_mask, M, (w, h))
            
            if self.current_centerline is not None and len(self.current_centerline) > 0:
                self.current_centerline = self.current_centerline.copy()
                self.current_centerline[:, 0] = np.clip(
                    self.current_centerline[:, 0] + int(median_dy), 0, h - 1
                )
                self.current_centerline[:, 1] = np.clip(
                    self.current_centerline[:, 1] + int(median_dx), 0, w - 1
                )
        
        return self.current_mask, self.current_centerline
    
    def reset(self):
        self.frame_count = 0
        self.current_mask = None
        self.current_centerline = None
        self.prev_frame = None
        self.initialized = False
        if self.sam_tracker:
            self.sam_tracker.reset()


def create_vessel_segmenter(use_sam=False, sam_checkpoint=None):
    if use_sam:
        segmenter = SAMVesselSegmenter(model_type="vit_b")
        if sam_checkpoint:
            segmenter.load_model(sam_checkpoint)
        return segmenter
    else:
        return LightweightVesselSegmenter()


def create_vessel_tracker(use_sam=True, sam_interval=10):
    return HybridVesselTracker(use_sam=use_sam, sam_interval=sam_interval)

