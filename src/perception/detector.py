import torch
import os
from ultralytics import YOLO
import logging
from typing import List, Optional
import time

class PerceptionEngine:
    """Thread-safe wrapper around YOLO"""

    def __init__(self, config):
        self.cfg = config
        self.device = config.device if torch.cuda.is_available() else 'cpu'
        
        model_dir = os.path.dirname(config.model_path)
        if model_dir and not os.path.exists(model_dir):
            os.makedirs(model_dir)

        if not os.path.exists(config.model_path):
            logging.warning(f"Model not found at {config.model_path}. Attempting to download...")
            self._download_model(config.model_name, config.model_path)
        
        if not os.path.exists(config.model_path):
            raise FileNotFoundError(f"Model path not found: {config.model_path}")

        logging.info(f"Loading YOLO model from {config.model_path}...")
        
        self.model = YOLO(config.model_path)
    
        if self.device != 'cpu':
            self.model.to(self.device)
            logging.info(f"Model moved to {self.device}")
        
        self._warmup()
        
    def _download_model(self, model_name: str, save_path: str):
        """Helper method to download model"""
        try:
            logging.info(f"Downloading {model_name}...")
            temp_model = YOLO(model_name) 
            temp_model.save(save_path)
            logging.info(f"Model saved to {save_path}")
        except Exception as e:
            logging.error(f"Failed to download model: {e}")
    
    def _warmup(self):
        """Warm up GPU with dummy inference"""
        logging.info("Warming up the model...")
        dummy_input = torch.zeros(1, 3, self.cfg.input_size, self.cfg.input_size).to(self.device)

        if self.device != 'cpu':
            dummy_input = dummy_input.to(self.device)
        
        self.model(dummy_input, verbose=False)
        logging.info("Model warm-up complete.")
    
    def detect(self, frames: torch.Tensor) -> List:
        """Run object detection on input frames"""
        results = self.model(
            source=frames,
            conf=self.cfg.confidence,
            iou=self.cfg.iou_threshold,
            imgsz=self.cfg.imgsz,
            device=self.device,
            verbose=False,
            classes=self.cfg.classes,
            half=self.cfg.half
        )

        return results

    def track(self, frames: torch.Tensor) -> List:
        """Run object tracking on input frames"""
        results = self.model.track(
            source=frames,
            conf=self.cfg.confidence,
            iou=self.cfg.iou_threshold,
            imgsz=self.cfg.imgsz,
            device=self.device,
            persist=True,
            tracker=self.cfg.tracker_config,
            verbose=False,
            classes=self.cfg.classes,
            half=self.cfg.half
        )

        return results

    def export_tensorrt(self, output_path: str):
        """Export model to TensorRT for production."""
        logging.info("Exporting to TensorRT...")
        self.model.export(
            format="engine",
            save_path=output_path,
            imgsz=self.cfg.imgsz,
            device=self.device,
            half=self.cfg.half,
            dynamic=True
        )