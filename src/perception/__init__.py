"""
Perception Module for Object Detection and Tracking.

This module provides YOLO-based object detection and tracking using
finetuned models for traffic and lane detection tasks.

Components:
    - PerceptionEngine: Main detection/tracking interface
    - ModelRegistry: Registry of available finetuned models
    - Detection, FrameDetections: Result dataclasses
    - parse_yolo_results: Result parsing utility

Usage:
    from src.perception import PerceptionEngine, ModelRegistry, get_model
    
    # Using the registry
    model = get_model("yolo11n_bdd100k")
    
    # Using the engine
    from src.configs import YOLOConfig
    config = YOLOConfig(model_path="models/finetune/yolo11n_bdd100k/weights/best.pt")
    engine = PerceptionEngine(config)
    detections = engine.detect_and_parse(frames)
"""

from .detector import PerceptionEngine
from .model_registry import (
    ModelRegistry,
    ModelInfo,
    get_model,
    get_best_model_for_task,
)
from .results import (
    Detection,
    FrameDetections,
    parse_yolo_results,
    aggregate_detections,
    detections_to_annotations,
)

__all__ = [
    # Main engine
    "PerceptionEngine",
    # Model registry
    "ModelRegistry",
    "ModelInfo", 
    "get_model",
    "get_best_model_for_task",
    # Results
    "Detection",
    "FrameDetections",
    "parse_yolo_results",
    "aggregate_detections",
    "detections_to_annotations",
]
