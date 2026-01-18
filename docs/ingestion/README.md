# Video Ingestion Module

The ingestion module provides GPU-accelerated video loading, temporal sampling, and batch processing for the RoadBuddy traffic law QA system.

## Design Principle

> **NO resizing/cropping during ingestion** to preserve image quality.
> Only temporal sampling is performed - downstream models receive native resolution frames.

## Quick Start

```python
from src.ingestion import RoadVideoLoader, FrameSampler
from src.configs import DecordConfig

# Load video
config = DecordConfig(video_path='video.mp4', device='gpu')
loader = RoadVideoLoader(config)

# Adaptive sampling: longer videos get more frames
frames = loader.sample_adaptive(min_frames=8, max_frames=64)
print(f"Extracted {len(frames)} frames at native resolution")
```

## Module Structure

```
src/ingestion/
├── __init__.py       # Public API exports
├── loader.py         # RoadVideoLoader - Core video loading
├── sampler.py        # FrameSampler - Sampling strategies
├── processor.py      # BatchVideoProcessor - Batch processing
└── utils.py          # Utility functions
```

## Components

| Component | Description | Documentation |
|-----------|-------------|---------------|
| [RoadVideoLoader](loader.md) | GPU-accelerated video loading with Decord/OpenCV | Core class |
| [FrameSampler](sampler.md) | Frame sampling strategies | Sampling guide |
| [BatchVideoProcessor](processor.md) | Dataset batch processing | Batch operations |
| [Utilities](utils.md) | Helper functions | API reference |

## Sampling Strategies

| Strategy | Description | When to Use |
|----------|-------------|-------------|
| **Adaptive** | Frame count scales with duration | Default for variable-length videos |
| **Uniform** | Evenly spaced frames | Fixed frame count needed |
| **FPS-based** | Sample at target FPS | Time-based analysis |
| **Temporal Chunks** | Sample from video segments | Ensure temporal coverage |

### Adaptive Sampling Formula

```
num_frames = clip(duration × frames_per_second, min_frames, max_frames)
```

Example results with default settings (0.5 fps, min=8, max=64):
- 10s video → 8 frames (minimum)
- 30s video → 15 frames
- 60s video → 30 frames
- 120s video → 60 frames
- 200s video → 64 frames (maximum)

## Dependencies

- **decord**: GPU-accelerated video decoding (primary)
- **opencv-python**: Fallback video loading
- **torch**: Tensor operations
- **numpy**: Numerical computations

## Testing

```bash
# Run all unit tests
pytest tests/test_ingestion.py -v

# Run with real video file
pytest tests/test_ingestion.py --video_path data/raw/train/videos/0.mp4 -v

# Test standalone scripts
python tests/test_loader.py data/raw/train/videos/0.mp4
python tests/test_utils.py data/raw/train/videos/0.mp4
```

## Next Steps

- [Loader Guide](loader.md) - Detailed video loading documentation
- [Sampler Guide](sampler.md) - Sampling strategies explained
- [Processor Guide](processor.md) - Batch processing documentation
- [Utilities Reference](utils.md) - Helper functions API
