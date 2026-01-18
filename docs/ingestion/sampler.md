# **FrameSampler**

Intelligent frame sampling strategies for video processing.

## **Overview**

`FrameSampler` provides various strategies to select which frames to extract from a video. It focuses on **temporal sampling only** - no image transformations are applied to preserve quality.

## **Initialization**

```python
from src.ingestion import FrameSampler

# Basic usage
sampler = FrameSampler()

# With seed for reproducibility
sampler = FrameSampler(seed=42)
```

## **Sampling strategies**

### **Uniform sampling**

    Evenly spaced frames from start to end.

```python
indices = sampler.sample_uniform(total_frames=1000, num_frames=10)
# Returns: [0, 111, 222, 333, 444, 555, 666, 777, 888, 999]
```

### **Adaptive sampling**

    Frame count scales with video duration.
    

```python
indices = sampler.sample_adaptive(
    total_frames=900,      # 30 second video at 30fps
    fps=30.0,
    min_frames=8,          # Minimum frames
    max_frames=64,         # Maximum frames
    frames_per_second=0.5  # Target rate
)
# 30s × 0.5 = 15 frames

# Formula
# num_frames = clip(duration × frames_per_second, min_frames, max_frames)
```

### **FPS-based sampling**

    Sample at a target frame rate.

```python
indices = sampler.sample_fps(
    total_frames=900,    # 30 second video at 30fps
    video_fps=30.0,
    target_fps=1.0       # 1 frame per second
)
# Returns 30 frames, one every 30 original frames
```

### **Temporal chunk sampling**

    Divide video into segments and sample from each.

```python
indices = sampler.sample_temporal_chunks(
    total_frames=1000,
    num_chunks=4,         # 4 segments
    frames_per_chunk=2    # 2 frames per segment
)
# Returns 8 frames spread across 4 video segments
```

## **Utility methods**

### **Calculate frame count**

```python
count = sampler.calculate_frame_count(
    duration=60.0,
    frames_per_second=0.5,
    min_frames=8,
    max_frames=64
)
# Returns 30
```

### **Get sampling info**

```python
info = sampler.get_sampling_info(
    total_frames=1000,
    fps=30.0,
    indices=[0, 250, 500, 750, 999]
)
# Returns: {
#     'num_sampled': 5,
#     'total_frames': 1000,
#     'coverage_ratio': 0.005,
#     'avg_interval': 249.75,
#     'avg_interval_seconds': 8.325,
#     'sampling_fps': 0.15,
#     'first_frame': 0,
#     'last_frame': 999,
#     'video_duration': 33.33
# }
```

## **Convenience functions**

    For quick one-off sampling without creating a class instance:

```python
from src.ingestion import sample_video_adaptive, sample_video_uniform

# Adaptive
indices = sample_video_adaptive(total_frames=900, fps=30.0, min_frames=8, max_frames=64)

# Uniform
indices = sample_video_uniform(total_frames=1000, num_frames=10)
```

## **Strategy comparison**

| Strategy | Frame Count | Coverage | Best For |
|----------|-------------|----------|----------|
| Uniform | Fixed | Perfect | Fixed requirements |
| Adaptive | Variable | Good | Variable-length videos |
| FPS-based | Variable | Time-based | Time-sensitive analysis |
| Temporal Chunks | Fixed | Segmented | Ensuring representation |

## **Integration with RoadVideoLoader**

```python
from src.ingestion import RoadVideoLoader, FrameSampler
from src.configs import DecordConfig

# Method 1: Use loader's built-in methods
config = DecordConfig(video_path='video.mp4')
loader = RoadVideoLoader(config)
frames = loader.sample_adaptive(min_frames=8, max_frames=64)

# Method 2: Use FrameSampler for indices, then extract
sampler = FrameSampler()
indices = sampler.sample_adaptive(loader.total_frames, loader.fps)
frames = loader.sample_indices(indices)
```