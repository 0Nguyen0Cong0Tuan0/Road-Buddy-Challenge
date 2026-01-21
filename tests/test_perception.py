"""
Comprehensive CLI Test Script for Perception Module.

This script tests all components of the perception module:
- ModelRegistry: Model discovery and factory
- PerceptionEngine: Detection and tracking
- Results: Detection dataclasses and parsing utilities

Usage:
    # Run all tests
    python tests/test_perception.py --test all
    
    # Test specific components
    python tests/test_perception.py --test registry
    python tests/test_perception.py --test results
    python tests/test_perception.py --test detector
    
    # Verbose output
    python tests/test_perception.py --test all -v
    
    # Skip model loading (faster, for testing utilities only)
    python tests/test_perception.py --test all --skip-models
"""

import sys
import argparse
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np


class Colors:
    """ANSI color codes for terminal output."""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    END = '\033[0m'


def print_header(text: str):
    """Print a section header."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}\n")


def print_subheader(text: str):
    """Print a subsection header."""
    print(f"\n{Colors.CYAN}--- {text} ---{Colors.END}")


def print_success(text: str):
    """Print success message."""
    print(f"{Colors.GREEN}[PASS]{Colors.END} {text}")


def print_fail(text: str):
    """Print failure message."""
    print(f"{Colors.RED}[FAIL]{Colors.END} {text}")


def print_info(text: str):
    """Print info message."""
    print(f"{Colors.YELLOW}  INFO:{Colors.END} {text}")


def print_result(name: str, value: Any):
    """Print a name-value result."""
    print(f"  {name}: {value}")


class PerceptionTester:
    """Comprehensive tester for perception module."""
    
    def __init__(
        self,
        verbose: bool = False,
        skip_models: bool = False
    ):
        self.verbose = verbose
        self.skip_models = skip_models
        self.results: Dict[str, bool] = {}
        self.errors: List[str] = []
    
    def run_all_tests(self) -> bool:
        """Run all test suites."""
        print_header("PERCEPTION MODULE TEST SUITE")
        print(f"Skip model loading: {self.skip_models}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        start_time = time.time()
        
        # Run test suites
        self.test_registry()
        self.test_results()
        
        if not self.skip_models:
            self.test_detector()
        else:
            print_info("Skipping detector tests (--skip-models flag)")
        
        elapsed = time.time() - start_time
        
        # Print summary
        self._print_summary(elapsed)
        
        return all(self.results.values())
    
    def test_registry(self) -> bool:
        """Test ModelRegistry."""
        print_header("TESTING: ModelRegistry (model_registry.py)")
        
        from src.perception.model_registry import (
            ModelRegistry,
            ModelInfo,
            PROJECT_ROOT,
            FINETUNE_DIR
        )
        
        all_passed = True
        
        # Test 1: Project root detection
        print_subheader("Test 1: Project root detection")
        try:
            if PROJECT_ROOT.exists() and (PROJECT_ROOT / 'src').exists():
                print_success(f"Project root detected: {PROJECT_ROOT}")
                self.results['registry.project_root'] = True
            else:
                print_fail(f"Invalid project root: {PROJECT_ROOT}")
                self.results['registry.project_root'] = False
                all_passed = False
        except Exception as e:
            print_fail(f"Exception: {e}")
            self.results['registry.project_root'] = False
            all_passed = False
        
        # Test 2: List available models
        print_subheader("Test 2: list_models()")
        try:
            models = ModelRegistry.list_models()
            
            expected = [
                "yolo11n_road_lane",
                "yolo11n_bdd100k", 
                "yolo11l_road_lane",
                "yolo11l_bdd100k"
            ]
            
            if all(m in models for m in expected):
                print_success(f"Found {len(models)} registered models")
                for m in models:
                    print_result("  Model", m)
                self.results['registry.list_models'] = True
            else:
                print_fail(f"Missing expected models")
                self.results['registry.list_models'] = False
                all_passed = False
        except Exception as e:
            print_fail(f"Exception: {e}")
            self.results['registry.list_models'] = False
            all_passed = False
        
        # Test 3: List available models (with weights)
        print_subheader("Test 3: list_available_models()")
        try:
            available = ModelRegistry.list_available_models()
            print_success(f"Found {len(available)} models with weights")
            for m in available:
                print_result("  Available", m)
            self.results['registry.list_available'] = True
        except Exception as e:
            print_fail(f"Exception: {e}")
            self.results['registry.list_available'] = False
            all_passed = False
        
        # Test 4: Get model info
        print_subheader("Test 4: get_model_info()")
        try:
            info = ModelRegistry.get_model_info("yolo11n_road_lane")
            
            if info and isinstance(info, ModelInfo):
                print_success("Got model info for yolo11n_road_lane")
                print_result("Name", info.name)
                print_result("Path", info.path)
                print_result("Base model", info.base_model)
                print_result("Dataset", info.dataset)
                print_result("Exists", info.exists)
                self.results['registry.get_model_info'] = True
            else:
                print_fail("Failed to get model info")
                self.results['registry.get_model_info'] = False
                all_passed = False
        except Exception as e:
            print_fail(f"Exception: {e}")
            self.results['registry.get_model_info'] = False
            all_passed = False
        
        # Test 5: Get model path
        print_subheader("Test 5: get_model_path()")
        try:
            path = ModelRegistry.get_model_path("yolo11n_bdd100k")
            
            if path is not None:
                print_success(f"Got model path")
                print_result("Path", path)
                print_result("Exists", path.exists())
                self.results['registry.get_model_path'] = True
            else:
                print_fail("Path is None")
                self.results['registry.get_model_path'] = False
        except Exception as e:
            print_fail(f"Exception: {e}")
            self.results['registry.get_model_path'] = False
            all_passed = False
        
        # Test 6: Filter by dataset
        print_subheader("Test 6: get_models_by_dataset()")
        try:
            road_lane_models = ModelRegistry.get_models_by_dataset("road_lane")
            bdd100k_models = ModelRegistry.get_models_by_dataset("bdd100k")
            
            if len(road_lane_models) == 2 and len(bdd100k_models) == 2:
                print_success("Dataset filtering works")
                print_result("Road Lane models", road_lane_models)
                print_result("BDD100K models", bdd100k_models)
                self.results['registry.filter_dataset'] = True
            else:
                print_fail("Incorrect filtering results")
                self.results['registry.filter_dataset'] = False
        except Exception as e:
            print_fail(f"Exception: {e}")
            self.results['registry.filter_dataset'] = False
            all_passed = False
        
        # Test 7: Filter by base model
        print_subheader("Test 7: get_models_by_base()")
        try:
            yolo11n_models = ModelRegistry.get_models_by_base("yolo11n")
            yolo11l_models = ModelRegistry.get_models_by_base("yolo11l")
            
            if len(yolo11n_models) == 2 and len(yolo11l_models) == 2:
                print_success("Base model filtering works")
                print_result("YOLO11n models", yolo11n_models)
                print_result("YOLO11l models", yolo11l_models)
                self.results['registry.filter_base'] = True
            else:
                print_fail("Incorrect filtering results")
                self.results['registry.filter_base'] = False
        except Exception as e:
            print_fail(f"Exception: {e}")
            self.results['registry.filter_base'] = False
            all_passed = False
        
        return all_passed
    
    def test_results(self) -> bool:
        """Test result dataclasses and utilities."""
        print_header("TESTING: Results (results.py)")
        
        from src.perception.results import (
            Detection,
            FrameDetections,
            parse_yolo_results,
            aggregate_detections,
            detections_to_annotations
        )
        
        all_passed = True
        
        # Test 1: Detection dataclass
        print_subheader("Test 1: Detection dataclass")
        try:
            det = Detection(
                bbox=(100.0, 100.0, 200.0, 200.0),
                confidence=0.85,
                class_id=0,
                class_name="car"
            )
            
            checks = [
                det.x1 == 100.0,
                det.y1 == 100.0,
                det.x2 == 200.0,
                det.y2 == 200.0,
                det.width == 100.0,
                det.height == 100.0,
                det.area == 10000.0,
                det.center == (150.0, 150.0)
            ]
            
            if all(checks):
                print_success("Detection properties work correctly")
                print_result("BBox", det.bbox)
                print_result("Center", det.center)
                print_result("Area", det.area)
                self.results['results.detection'] = True
            else:
                print_fail("Detection property mismatch")
                self.results['results.detection'] = False
                all_passed = False
        except Exception as e:
            print_fail(f"Exception: {e}")
            self.results['results.detection'] = False
            all_passed = False
        
        # Test 2: Detection to_dict
        print_subheader("Test 2: Detection.to_dict()")
        try:
            det_dict = det.to_dict()
            required_keys = ['bbox', 'confidence', 'class_id', 'class_name', 'width', 'height', 'area', 'center']
            
            if all(k in det_dict for k in required_keys):
                print_success("Detection.to_dict() works")
                self.results['results.detection_to_dict'] = True
            else:
                print_fail("Missing keys in dict")
                self.results['results.detection_to_dict'] = False
        except Exception as e:
            print_fail(f"Exception: {e}")
            self.results['results.detection_to_dict'] = False
            all_passed = False
        
        # Test 3: Detection to_yolo_format
        print_subheader("Test 3: Detection.to_yolo_format()")
        try:
            cls_id, cx, cy, w, h = det.to_yolo_format(640, 480)
            
            expected_cx = 150.0 / 640
            expected_cy = 150.0 / 480
            expected_w = 100.0 / 640
            expected_h = 100.0 / 480
            
            checks = [
                cls_id == 0,
                abs(cx - expected_cx) < 0.001,
                abs(cy - expected_cy) < 0.001,
                abs(w - expected_w) < 0.001,
                abs(h - expected_h) < 0.001
            ]
            
            if all(checks):
                print_success("YOLO format conversion works")
                print_result("Normalized", f"({cx:.4f}, {cy:.4f}, {w:.4f}, {h:.4f})")
                self.results['results.yolo_format'] = True
            else:
                print_fail("YOLO format mismatch")
                self.results['results.yolo_format'] = False
        except Exception as e:
            print_fail(f"Exception: {e}")
            self.results['results.yolo_format'] = False
            all_passed = False
        
        # Test 4: FrameDetections dataclass
        print_subheader("Test 4: FrameDetections dataclass")
        try:
            det1 = Detection(bbox=(10, 10, 50, 50), confidence=0.9, class_id=0, class_name="car")
            det2 = Detection(bbox=(100, 100, 150, 150), confidence=0.8, class_id=1, class_name="truck")
            det3 = Detection(bbox=(200, 200, 250, 250), confidence=0.7, class_id=0, class_name="car")
            
            frame = FrameDetections(
                frame_idx=0,
                detections=[det1, det2, det3],
                image_size=(640, 480)
            )
            
            checks = [
                frame.num_detections == 3,
                frame.class_counts == {"car": 2, "truck": 1},
                set(frame.unique_classes) == {"car", "truck"}
            ]
            
            if all(checks):
                print_success("FrameDetections works correctly")
                print_result("Num detections", frame.num_detections)
                print_result("Class counts", frame.class_counts)
                self.results['results.frame_detections'] = True
            else:
                print_fail("FrameDetections mismatch")
                self.results['results.frame_detections'] = False
        except Exception as e:
            print_fail(f"Exception: {e}")
            self.results['results.frame_detections'] = False
            all_passed = False
        
        # Test 5: Filter by class
        print_subheader("Test 5: FrameDetections.filter_by_class()")
        try:
            cars = frame.filter_by_class("car")
            trucks = frame.filter_by_class("truck")
            
            if len(cars) == 2 and len(trucks) == 1:
                print_success("Class filtering works")
                print_result("Cars found", len(cars))
                print_result("Trucks found", len(trucks))
                self.results['results.filter_class'] = True
            else:
                print_fail("Incorrect filter results")
                self.results['results.filter_class'] = False
        except Exception as e:
            print_fail(f"Exception: {e}")
            self.results['results.filter_class'] = False
            all_passed = False
        
        # Test 6: Filter by confidence
        print_subheader("Test 6: FrameDetections.filter_by_confidence()")
        try:
            high_conf = frame.filter_by_confidence(0.85)
            
            if len(high_conf) == 1 and high_conf[0].confidence == 0.9:
                print_success("Confidence filtering works")
                print_result("High confidence detections", len(high_conf))
                self.results['results.filter_confidence'] = True
            else:
                print_fail("Incorrect filter results")
                self.results['results.filter_confidence'] = False
        except Exception as e:
            print_fail(f"Exception: {e}")
            self.results['results.filter_confidence'] = False
            all_passed = False
        
        # Test 7: Aggregate detections
        print_subheader("Test 7: aggregate_detections()")
        try:
            frame2 = FrameDetections(
                frame_idx=1,
                detections=[det1, det2],
                image_size=(640, 480)
            )
            
            stats = aggregate_detections([frame, frame2])
            
            checks = [
                stats['num_frames'] == 2,
                stats['total_detections'] == 5,
                'avg_detections_per_frame' in stats,
                'avg_confidence' in stats
            ]
            
            if all(checks):
                print_success("Aggregation works correctly")
                print_result("Total detections", stats['total_detections'])
                print_result("Avg per frame", f"{stats['avg_detections_per_frame']:.2f}")
                print_result("Avg confidence", f"{stats['avg_confidence']:.2f}")
                self.results['results.aggregate'] = True
            else:
                print_fail("Aggregation mismatch")
                self.results['results.aggregate'] = False
        except Exception as e:
            print_fail(f"Exception: {e}")
            self.results['results.aggregate'] = False
            all_passed = False
        
        # Test 8: Convert to annotations
        print_subheader("Test 8: detections_to_annotations()")
        try:
            annotations = detections_to_annotations(frame, 640, 480)
            
            if len(annotations) == 3:
                print_success("Annotation conversion works")
                for i, ann in enumerate(annotations[:2]):  # Show first 2
                    print_result(f"Annotation {i}", ann)
                self.results['results.annotations'] = True
            else:
                print_fail(f"Expected 3 annotations, got {len(annotations)}")
                self.results['results.annotations'] = False
        except Exception as e:
            print_fail(f"Exception: {e}")
            self.results['results.annotations'] = False
            all_passed = False
        
        return all_passed
    
    def test_detector(self) -> bool:
        """Test PerceptionEngine with actual models."""
        print_header("TESTING: PerceptionEngine (detector.py)")
        
        from src.perception.model_registry import ModelRegistry
        from src.perception.detector import PerceptionEngine
        from src.configs import YOLOConfig
        
        all_passed = True
        
        # Find an available model
        available = ModelRegistry.list_available_models()
        if not available:
            print_info("No models available, skipping detector tests")
            print_info("Download finetuned models to models/finetune/*/weights/best.pt")
            return True
        
        model_name = available[0]
        model_info = ModelRegistry.get_model_info(model_name)
        print_info(f"Using model: {model_name}")
        
        # Test 1: Initialize PerceptionEngine
        print_subheader("Test 1: PerceptionEngine initialization")
        try:
            config = YOLOConfig(
                model_path=str(model_info.path),
                device="cpu",  # Use CPU for testing
                confidence=0.25,
                iou_threshold=0.45,
                imgsz=640
            )
            
            engine = PerceptionEngine(config, warmup=True)
            
            print_success("PerceptionEngine initialized")
            print_result("Model", config.model_path)
            print_result("Device", engine.device)
            print_result("Classes", len(engine.class_names))
            self.results['detector.init'] = True
        except Exception as e:
            print_fail(f"Exception: {e}")
            self.results['detector.init'] = False
            return False
        
        # Test 2: Get class names
        print_subheader("Test 2: get_class_names()")
        try:
            names = engine.get_class_names()
            
            if isinstance(names, dict):
                print_success("Class names retrieved")
                print_result("Num classes", len(names))
                if self.verbose and names:
                    for idx, name in list(names.items())[:5]:
                        print_result(f"  Class {idx}", name)
                self.results['detector.class_names'] = True
            else:
                print_fail("Invalid class names format")
                self.results['detector.class_names'] = False
        except Exception as e:
            print_fail(f"Exception: {e}")
            self.results['detector.class_names'] = False
            all_passed = False
        
        # Test 3: Detection on dummy image
        print_subheader("Test 3: detect() with dummy image")
        try:
            # Create a dummy RGB image (numpy array, HWC format)
            dummy_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            
            results = engine.detect(dummy_img)
            
            if results is not None and len(results) == 1:
                print_success("Detection completed")
                print_result("Num results", len(results))
                result = results[0]
                num_detections = len(result.boxes) if result.boxes is not None else 0
                print_result("Detections", num_detections)
                self.results['detector.detect'] = True
            else:
                print_fail("Unexpected results format")
                self.results['detector.detect'] = False
        except Exception as e:
            print_fail(f"Exception: {e}")
            self.results['detector.detect'] = False
            all_passed = False
        
        # Test 4: Parse results
        print_subheader("Test 4: parse_results()")
        try:
            parsed = engine.parse_results(results)
            
            if isinstance(parsed, list) and len(parsed) == 1:
                print_success("Results parsed successfully")
                frame = parsed[0]
                print_result("Frame index", frame.frame_idx)
                print_result("Num detections", frame.num_detections)
                print_result("Image size", frame.image_size)
                self.results['detector.parse_results'] = True
            else:
                print_fail("Unexpected parsed format")
                self.results['detector.parse_results'] = False
        except Exception as e:
            print_fail(f"Exception: {e}")
            self.results['detector.parse_results'] = False
            all_passed = False
        
        # Test 5: detect_and_parse combined
        print_subheader("Test 5: detect_and_parse()")
        try:
            detections = engine.detect_and_parse(dummy_img, start_frame_idx=10)
            
            if isinstance(detections, list) and len(detections) == 1:
                print_success("detect_and_parse works")
                print_result("Frame index", detections[0].frame_idx)
                self.results['detector.detect_and_parse'] = True
            else:
                print_fail("Unexpected output")
                self.results['detector.detect_and_parse'] = False
        except Exception as e:
            print_fail(f"Exception: {e}")
            self.results['detector.detect_and_parse'] = False
            all_passed = False
        
        # Test 6: Batch detection
        print_subheader("Test 6: Batch detection")
        try:
            # Create batch of 3 images
            batch = [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(3)]
            
            results = engine.detect(batch)
            
            if len(results) == 3:
                print_success("Batch detection works")
                print_result("Batch size", len(results))
                total_dets = sum(len(r.boxes) if r.boxes is not None else 0 for r in results)
                print_result("Total detections", total_dets)
                self.results['detector.batch'] = True
            else:
                print_fail(f"Expected 3 results, got {len(results)}")
                self.results['detector.batch'] = False
        except Exception as e:
            print_fail(f"Exception: {e}")
            self.results['detector.batch'] = False
            all_passed = False
        
        # Test 7: __repr__
        print_subheader("Test 7: __repr__()")
        try:
            repr_str = repr(engine)
            
            if "PerceptionEngine" in repr_str and "model=" in repr_str:
                print_success("__repr__ works")
                print_result("Repr", repr_str)
                self.results['detector.repr'] = True
            else:
                print_fail("Invalid repr format")
                self.results['detector.repr'] = False
        except Exception as e:
            print_fail(f"Exception: {e}")
            self.results['detector.repr'] = False
            all_passed = False
        
        return all_passed
    
    def _print_summary(self, elapsed: float):
        """Print test summary."""
        print_header("TEST SUMMARY")
        
        passed = sum(1 for v in self.results.values() if v)
        failed = sum(1 for v in self.results.values() if not v)
        total = len(self.results)
        
        print(f"Time elapsed: {elapsed:.2f}s")
        print(f"Tests run: {total}")
        print(f"{Colors.GREEN}Passed: {passed}{Colors.END}")
        print(f"{Colors.RED}Failed: {failed}{Colors.END}")
        
        if failed > 0:
            print(f"\n{Colors.RED}Failed tests:{Colors.END}")
            for name, result in self.results.items():
                if not result:
                    print(f"  - {name}")
        
        print()
        if failed == 0:
            print(f"{Colors.GREEN}{Colors.BOLD}All tests passed!{Colors.END}")
        else:
            print(f"{Colors.RED}{Colors.BOLD}{failed} test(s) failed.{Colors.END}")


def get_model_path(name: str):
    """Helper to get model path from registry."""
    from src.perception.model_registry import ModelRegistry
    return ModelRegistry.get_model_path(name)


def main():
    parser = argparse.ArgumentParser(
        description='Test perception module',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tests/test_perception.py --test all
  python tests/test_perception.py --test registry
  python tests/test_perception.py --test results
  python tests/test_perception.py --test detector -v
  python tests/test_perception.py --test all --skip-models
        """
    )
    
    parser.add_argument(
        '--test', '-t',
        type=str,
        choices=['all', 'registry', 'results', 'detector'],
        default='all',
        help='Which component to test (default: all)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )
    parser.add_argument(
        '--skip-models',
        action='store_true',
        help='Skip tests that require loading YOLO models (faster)'
    )
    
    args = parser.parse_args()
    
    # Create tester
    tester = PerceptionTester(
        verbose=args.verbose,
        skip_models=args.skip_models
    )
    
    # Run tests
    if args.test == 'all':
        success = tester.run_all_tests()
    elif args.test == 'registry':
        tester.test_registry()
        tester._print_summary(0)
        success = all(tester.results.values())
    elif args.test == 'results':
        tester.test_results()
        tester._print_summary(0)
        success = all(tester.results.values())
    elif args.test == 'detector':
        if args.skip_models:
            print_info("Cannot test detector with --skip-models flag")
            success = True
        else:
            tester.test_detector()
            tester._print_summary(0)
            success = all(tester.results.values())
    else:
        success = False
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
