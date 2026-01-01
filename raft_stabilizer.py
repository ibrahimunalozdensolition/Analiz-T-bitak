import cv2
import numpy as np
import torch
from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
from torchvision.transforms import functional as F


class RAFTStabilizer:
    def __init__(self, device=None):
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
        
        weights = Raft_Small_Weights.DEFAULT
        self.model = raft_small(weights=weights).to(self.device)
        self.model.eval()
        
        self.cumulative_dx = 0.0
        self.cumulative_dy = 0.0
        self.frame_count = 0
        self.original_roi = None
        self.reset_interval = 120
        self.max_drift = 50
        
        self.reference_template = None
        self.template_update_interval = 30
        self.template_match_threshold = 0.6
        self.last_good_roi = None
    
    def reset(self):
        self.cumulative_dx = 0.0
        self.cumulative_dy = 0.0
        self.frame_count = 0
        self.original_roi = None
        self.reference_template = None
        self.last_good_roi = None
    
    def _create_template(self, frame, roi):
        x, y, w, h = roi
        template = frame[y:y+h, x:x+w].copy()
        if len(template.shape) == 3:
            template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        return template
    
    def _template_match_correction(self, frame, roi, search_range=15):
        if self.reference_template is None:
            return roi, 1.0
        
        x, y, w, h = roi
        frame_h, frame_w = frame.shape[:2]
        
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        search_y1 = max(0, y - search_range)
        search_x1 = max(0, x - search_range)
        search_y2 = min(frame_h, y + h + search_range)
        search_x2 = min(frame_w, x + w + search_range)
        
        search_region = gray[search_y1:search_y2, search_x1:search_x2]
        
        if search_region.shape[0] < self.reference_template.shape[0] or \
           search_region.shape[1] < self.reference_template.shape[1]:
            return roi, 0.0
        
        result = cv2.matchTemplate(search_region, self.reference_template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        if max_val > self.template_match_threshold:
            new_x = search_x1 + max_loc[0]
            new_y = search_y1 + max_loc[1]
            
            new_x = max(0, min(new_x, frame_w - w))
            new_y = max(0, min(new_y, frame_h - h))
            
            return (new_x, new_y, w, h), max_val
        
        return roi, max_val
    
    def _verify_vessel_position(self, frame, roi, centerline_points=None):
        x, y, w, h = roi
        
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        roi_region = gray[y:y+h, x:x+w]
        
        inverted = 255 - roi_region
        col_sums = np.sum(inverted, axis=0)
        
        if len(col_sums) > 10:
            kernel_size = max(5, len(col_sums) // 8)
            if kernel_size % 2 == 0:
                kernel_size += 1
            smoothed = np.convolve(col_sums, np.ones(kernel_size)/kernel_size, mode='same')
            
            margin = len(smoothed) // 6
            search_region = smoothed[margin:-margin] if margin > 0 else smoothed
            if len(search_region) > 0:
                vessel_center = margin + np.argmax(search_region)
                expected_center = w // 2
                
                offset = vessel_center - expected_center
                
                if abs(offset) > 3:
                    frame_h, frame_w = frame.shape[:2]
                    new_x = x + offset
                    new_x = max(0, min(new_x, frame_w - w))
                    return (new_x, y, w, h), offset
        
        return roi, 0
    
    def _pad_to_multiple_of_8(self, frame):
        h, w = frame.shape[:2]
        new_h = ((h + 7) // 8) * 8
        new_w = ((w + 7) // 8) * 8
        
        if new_h == h and new_w == w:
            return frame, (0, 0)
        
        pad_h = new_h - h
        pad_w = new_w - w
        
        if len(frame.shape) == 2:
            padded = np.pad(frame, ((0, pad_h), (0, pad_w)), mode='edge')
        else:
            padded = np.pad(frame, ((0, pad_h), (0, pad_w), (0, 0)), mode='edge')
        
        return padded, (pad_h, pad_w)
    
    def _preprocess_frame(self, frame):
        if len(frame.shape) == 2:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        else:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float()
        tensor = tensor.unsqueeze(0)
        
        return tensor
    
    def compute_flow(self, prev_frame, curr_frame):
        orig_h, orig_w = prev_frame.shape[:2]
        
        prev_padded, padding = self._pad_to_multiple_of_8(prev_frame)
        curr_padded, _ = self._pad_to_multiple_of_8(curr_frame)
        
        prev_tensor = self._preprocess_frame(prev_padded).to(self.device)
        curr_tensor = self._preprocess_frame(curr_padded).to(self.device)
        
        with torch.no_grad():
            flow_predictions = self.model(prev_tensor, curr_tensor)
            flow = flow_predictions[-1]
        
        flow_np = flow.squeeze(0).permute(1, 2, 0).cpu().numpy()
        
        flow_np = flow_np[:orig_h, :orig_w, :]
        
        return flow_np
    
    def get_global_motion(self, flow):
        flow_x = flow[..., 0]
        flow_y = flow[..., 1]
        
        global_dx = np.median(flow_x)
        global_dy = np.median(flow_y)
        
        return global_dx, global_dy
    
    def stabilize_roi(self, prev_frame, curr_frame, roi):
        self.frame_count += 1
        
        if self.original_roi is None:
            self.original_roi = roi
            self.reference_template = self._create_template(curr_frame, roi)
            self.last_good_roi = roi
        
        x, y, w, h = roi
        frame_h, frame_w = curr_frame.shape[:2]
        
        margin = 30
        y1 = max(0, y - margin)
        x1 = max(0, x - margin)
        y2 = min(frame_h, y + h + margin)
        x2 = min(frame_w, x + w + margin)
        
        prev_region = prev_frame[y1:y2, x1:x2]
        curr_region = curr_frame[y1:y2, x1:x2]
        
        if prev_region.shape[0] < 32 or prev_region.shape[1] < 32:
            return roi, (0, 0), self.cumulative_dx, self.cumulative_dy
        
        flow = self.compute_flow(prev_region, curr_region)
        
        global_dx, global_dy = self.get_global_motion(flow)
        
        if abs(global_dx) < 0.3:
            global_dx = 0
        if abs(global_dy) < 0.3:
            global_dy = 0
        
        max_motion = 15
        global_dx = np.clip(global_dx, -max_motion, max_motion)
        global_dy = np.clip(global_dy, -max_motion, max_motion)
        
        self.cumulative_dx += global_dx
        self.cumulative_dy += global_dy
        
        new_x = int(x + global_dx)
        new_y = int(y + global_dy)
        
        new_x = max(0, min(new_x, frame_w - w))
        new_y = max(0, min(new_y, frame_h - h))
        
        raft_roi = (new_x, new_y, w, h)
        
        if self.frame_count % 5 == 0:
            verified_roi, vessel_offset = self._verify_vessel_position(curr_frame, raft_roi)
            if abs(vessel_offset) > 2:
                new_x = verified_roi[0]
                self.cumulative_dx += vessel_offset * 0.5
        
        if self.frame_count % 10 == 0 and self.reference_template is not None:
            template_roi, match_score = self._template_match_correction(curr_frame, (new_x, new_y, w, h), search_range=20)
            
            if match_score > self.template_match_threshold:
                blend = 0.3
                new_x = int(new_x * (1 - blend) + template_roi[0] * blend)
                new_y = int(new_y * (1 - blend) + template_roi[1] * blend)
                self.last_good_roi = (new_x, new_y, w, h)
            elif match_score < 0.4 and self.last_good_roi is not None:
                new_x = self.last_good_roi[0]
                new_y = self.last_good_roi[1]
        
        orig_x, orig_y = self.original_roi[0], self.original_roi[1]
        drift_x = abs(new_x - orig_x)
        drift_y = abs(new_y - orig_y)
        
        if drift_x > self.max_drift or drift_y > self.max_drift:
            correction = 0.4
            new_x = int(new_x - (new_x - orig_x) * correction)
            new_y = int(new_y - (new_y - orig_y) * correction)
            self.cumulative_dx *= 0.6
            self.cumulative_dy *= 0.6
        
        if self.frame_count % self.reset_interval == 0:
            self.cumulative_dx *= 0.8
            self.cumulative_dy *= 0.8
        
        if self.frame_count % self.template_update_interval == 0:
            current_template = self._create_template(curr_frame, (new_x, new_y, w, h))
            if self.reference_template is not None:
                result = cv2.matchTemplate(current_template, self.reference_template, cv2.TM_CCOEFF_NORMED)
                _, match_val, _, _ = cv2.minMaxLoc(result)
                if match_val > 0.7:
                    self.reference_template = cv2.addWeighted(
                        self.reference_template, 0.7, 
                        current_template, 0.3, 0
                    ).astype(np.uint8)
        
        return (new_x, new_y, w, h), (global_dx, global_dy), self.cumulative_dx, self.cumulative_dy
    
    def get_flow_visualization(self, flow, scale=1.0):
        flow_x = flow[..., 0]
        flow_y = flow[..., 1]
        
        magnitude = np.sqrt(flow_x**2 + flow_y**2)
        angle = np.arctan2(flow_y, flow_x)
        
        hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
        hsv[..., 0] = (angle * 180 / np.pi / 2 + 90).astype(np.uint8)
        hsv[..., 1] = 255
        hsv[..., 2] = np.clip(magnitude * scale * 10, 0, 255).astype(np.uint8)
        
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        return rgb


class RAFTVesselTracker:
    def __init__(self, device=None):
        self.stabilizer = RAFTStabilizer(device)
        self.centerline_points = None
        self.original_centerline = None
    
    def set_centerline(self, centerline_points):
        self.centerline_points = centerline_points.copy()
        self.original_centerline = centerline_points.copy()
    
    def reset(self):
        self.stabilizer.reset()
        if self.original_centerline is not None:
            self.centerline_points = self.original_centerline.copy()
    
    def track(self, prev_frame, curr_frame, roi):
        new_roi, displacement, cum_dx, cum_dy = self.stabilizer.stabilize_roi(
            prev_frame, curr_frame, roi
        )
        
        if self.centerline_points is not None and self.original_centerline is not None:
            roi_offset_x = new_roi[0] - self.stabilizer.original_roi[0]
            roi_offset_y = new_roi[1] - self.stabilizer.original_roi[1]
            
            updated_centerline = self.original_centerline.copy()
            updated_centerline[:, 0] = np.clip(
                updated_centerline[:, 0] + roi_offset_y, 
                0, 
                curr_frame.shape[0] - 1
            )
            updated_centerline[:, 1] = np.clip(
                updated_centerline[:, 1] + roi_offset_x, 
                0, 
                curr_frame.shape[1] - 1
            )
            self.centerline_points = updated_centerline
        
        return new_roi, displacement, self.centerline_points

