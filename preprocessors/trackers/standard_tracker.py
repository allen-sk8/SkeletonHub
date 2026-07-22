import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from preprocessors.trackers.base_tracker import BaseTracker

class StandardTracker(BaseTracker):
    """
    StandardTracker: Standard YOLO (YOLOv8 or YOLOv11) or RTMDet tracker.
    Runs frame-by-frame detection/tracking and extracts bounding boxes for the largest detected person.
    """
    def __init__(self, det_type='yolov8', weights=None, device='cuda:0'):
        super().__init__(device)
        self.det_type = det_type.lower()
        self.weights = weights
        self.model = None
        self.rtmdet_model = None

        if self.det_type in ['yolov8', 'yolov11']:
            if not self.weights:
                if self.det_type == 'yolov8':
                    self.weights = "/home/allen/SkeletonHub/external/gvhmr/inputs/checkpoints/yolo/yolov8x.pt"
                else:
                    self.weights = "/home/allen/yolo/tracking_upgrade/weights/yolo11n.pt"
                    
            if not os.path.exists(self.weights):
                print(f"⚠️ YOLO weights {self.weights} not found, falling back to download model: {self.det_type}x")
                self.weights = "yolov8x.pt" if self.det_type == 'yolov8' else "yolo11n.pt"

            print(f"⚙️ Initializing {self.det_type.upper()} with weights: {self.weights}")
            self.model = YOLO(self.weights)
            self.model.to(self.device)
            
        elif self.det_type == 'rtmdet':
            print("⚙️ Initializing RTMDet Detector...")
            try:
                from mmdet.apis import init_detector
                from mmpose.utils import adapt_mmdet_pipeline
                rtmpose3d_path = "/home/allen/SkeletonHub/external/mmpose/projects/rtmpose3d"
                det_config = os.path.join(rtmpose3d_path, 'demo', 'rtmdet_m_640-8xb32_coco-person.py')
                det_checkpoint = 'https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth'
                
                self.rtmdet_model = init_detector(det_config, det_checkpoint, device=self.device.lower())
                self.rtmdet_model.cfg = adapt_mmdet_pipeline(self.rtmdet_model.cfg)
            except Exception as e:
                raise ImportError(f"❌ Cannot import or initialize mmdet/mmpose for RTMDet: {e}. Please run inside 'skeleton_env'.")
        else:
            raise ValueError(f"❌ Unsupported standard detector type: {self.det_type}")

    def track(self, video_path: str) -> np.ndarray:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"❌ Cannot open video: {video_path}")
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        out_boxes = []

        for frame_idx in range(total_frames):
            ret, frame = cap.read()
            if not ret or frame is None:
                break
                
            bbox = [-1.0, -1.0, -1.0, -1.0]

            if self.det_type in ['yolov8', 'yolov11']:
                # Run YOLO prediction/tracking on the single frame
                # By default, use predict class 0 (person)
                results = self.model.predict(frame, classes=[0], conf=0.1, verbose=False)[0]
                if len(results.boxes) > 0:
                    xyxy = results.boxes.xyxy.cpu().numpy()
                    confs = results.boxes.conf.cpu().numpy()
                    
                    # Choose the person with the largest box area
                    areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
                    max_idx = np.argmax(areas)
                    bbox = xyxy[max_idx].tolist()
                    
            elif self.det_type == 'rtmdet':
                # Run RTMDet prediction
                from mmdet.apis import inference_detector
                det_result = inference_detector(self.rtmdet_model, frame)
                pred_instances = det_result.pred_instances.cpu().numpy()
                
                bboxes = pred_instances.bboxes
                scores = pred_instances.scores
                labels = pred_instances.labels
                
                # Person class label in COCO is 0, filter by score > 0.3
                valid_idx = np.logical_and(labels == 0, scores > 0.3)
                bboxes = bboxes[valid_idx]
                
                if len(bboxes) > 0:
                    areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
                    max_idx = np.argmax(areas)
                    bbox = bboxes[max_idx].tolist()

            out_boxes.append(bbox)

        cap.release()
        return np.array(out_boxes, dtype=np.float32)
