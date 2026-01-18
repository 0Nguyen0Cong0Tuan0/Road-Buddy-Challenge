# **Testing guide**

Comprehensive testing documentation for the ingestion module.

## **Test structure**

```
tests/
├── test_ingestion.py   # Comprehensive pytest suite (38 tests)
├── test_loader.py      # Standalone loader tests with output
└── test_utils.py       # Standalone utils tests with output
```

## **Running tests**

### **Pytest**

```bash
# Run all unit tests
pytest tests/test_ingestion.py -v

# Run specific test class
pytest tests/test_ingestion.py::TestFrameSampler -v
pytest tests/test_ingestion.py::TestUtilityFunctions -v

# Run with real video (integration tests)
pytest tests/test_ingestion.py --video_path data/raw/train/videos/0.mp4 -v
```

### **Standalone scripts**

Interactive tests with visual output

```bash
# Test video loader
python tests/test_loader.py data/raw/train/videos/0.mp4

# Test utilities
python tests/test_utils.py data/raw/train/videos/0.mp4

# With custom output directory
python tests/test_loader.py video.mp4 --output_dir ./debug_output

# Without saving images
python tests/test_loader.py video.mp4 --no_save
```

## **Test coverage**

### **TestFrameSampler (18 tests)**

| Test | Description |
|------|-------------|
| `test_sample_uniform_*` | Basic uniform sampling, edge cases |
| `test_sample_adaptive_*` | Short/medium/long videos, scaling |
| `test_sample_fps_*` | FPS-based sampling |
| `test_sample_temporal_chunks_*` | Chunk sampling, coverage |
| `test_calculate_frame_count` | Frame count calculation |
| `test_get_sampling_info` | Sampling info retrieval |

### **TestUtilityFunctions (12 tests)**

| Test | Description |
|------|-------------|
| `test_validate_video_*` | File validation |
| `test_estimate_memory` | Memory estimation |
| `test_convert_timestamp_*` | Time/frame conversion |
| `test_expand_bbox_*` | Bounding box expansion |
| `test_calculate_iou_*` | IoU calculation |

### **TestRoadVideoLoaderMocked (2 tests)**

| Test | Description |
|------|-------------|
| `test_loader_validates_video_path` | Path validation |
| `test_loader_validates_video_extension` | Extension validation |

### **TestRoadVideoLoaderReal (11 tests, requires video)**

| Test | Description |
|------|-------------|
| `test_loader_initialization` | Loader setup |
| `test_get_metadata` | Metadata extraction |
| `test_get_single_frame` | Single frame extraction |
| `test_sample_*` | All sampling methods |
| `test_stream_batches` | Batch streaming |

### **TestBatchVideoProcessor (1 test)**

| Test | Description |
|------|-------------|
| `test_processing_stats_to_dict` | Stats conversion |

### **TestModuleImports (3 tests)**

| Test | Description |
|------|-------------|
| `test_import_ingestion_module` | Main module imports |
| `test_import_sampler` | Sampler imports |
| `test_import_utils` | Utils imports |

## **Adding new tests**

### **For FrameSampler**

```python
class TestFrameSampler:
    def test_new_sampling_method(self, sampler):
        """Test description."""
        indices = sampler.your_new_method(...)
        assert len(indices) == expected_count
```

### **For RoadVideoLoader (with mock)**

```python
class TestRoadVideoLoaderMocked:
    def test_new_feature(self, mock_config):
        """Test with mocked config."""
        from src.ingestion.loader import RoadVideoLoader
        # Test logic
```

### **For utilities**

```python
class TestUtilityFunctions:
    def test_new_utility(self):
        """Test new utility function."""
        from src.ingestion.utils import new_function
        result = new_function(...)
        assert result == expected
```