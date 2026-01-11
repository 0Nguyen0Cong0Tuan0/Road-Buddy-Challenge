import torch
import os
from utralytics import YOLO
import logging
from typing import List, Optional
import time

class PerceptionEngine:
    """Thread-safe wrapper around YOLO11"""

    def __init__(self, config):
        self.cfg = config
        self.device = config.device if torch.cuda.is_available() else 'cpu'

        if not os.path.exists(config.model_path):
            raise FileNotFoundError(f"Model path does not found: {config.model_path}")

        logging.info(f"Loading YOLO model from {config.model_path} on {self.device}")

        
        
        
