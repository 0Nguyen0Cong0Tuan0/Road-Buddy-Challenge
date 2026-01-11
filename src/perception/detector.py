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
        
        def _download_model(self, model_name: str, save_path: str):
            """Helper method to download model"""
            try:
                logging.info(f"Downloading {model_name}...")
                temp_model = YOLO(model_name) 
                temp_model.save(save_path)
                logging.info(f"Model saved to {save_path}")
            except Exception as e:
                logging.error(f"Failed to download model: {e}")