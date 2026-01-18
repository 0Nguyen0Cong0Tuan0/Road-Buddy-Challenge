# **RoadVideoLoader**

GPU-accelerated video loader with Decord backend and OpenCV fallback.

## **Overview**

`RoadVideoLoader` is the core class for loading and extracting frames from video files. It automatically selects the best available backend (Decord GPU $\rightarrow$ Decord CPU $\rightarrow$ OpenCV).

## **Initialization**

```python
from src.ingestion import RoadVideoLoader
from src.configs import DecordConfig

config = DecordConfig(
    video_path='path/to/video.mp4',
    device='gpu',      # 'gpu' or 'cpu'
    ctx_id=0,          # GPU device ID
    batch_size=16,     # Batch size for streaming
    width=-1,          # -1 = native resolution
    height=-1,         # -1 = native resolution
    num_threads=0      # 0 = auto
)

loader = RoadVideoLoader(config)
```

## **Properties**

| Property | Type | Description |
|----------|------|-------------|
| `total_frames` | `int` | Total number of frames in video |
| `fps` | `float` | Frames per second |
| `duration` | `float` | Duration in seconds |
| `backend` | `str` | Active backend: `decord` or `opencv` |

## **Methods**

### **Frame extraction**


#### **`get_frame(frame_idx: int) -> torch.Tensor`**

    Extract a single frame.

```python
frame = loader.get_frame(0)  # First frame
# Returns: torch.Tensor shape (C, H, W), range [0, 1]
```

#### **`sample_indices(indices: List[int]) -> torch.Tensor`**

    Extract specific frames by indices.

```python
frames = loader.sample_indices([0, 50, 100, 150])
# Returns: torch.Tensor shape (N, C, H, W)
```

### **Sampling methods**

#### **`sample_uniform(num_frames: int) -> torch.Tensor`**

    Sample evenly spaced frames.

```python
frames = loader.sample_uniform(8)  # 8 evenly spaced frames
```

#### **`sample_fps(target_fps: float) -> torch.Tensor`**

    Sample at target FPS rate.

```python
frames = loader.sample_fps(1.0)  # 1 frame per second
```

#### **`sample_adaptive(min_frames, max_frames, frames_per_second) -> torch.Tensor`**

    Adaptive sampling based on video duration.

```python
# Longer videos get more frames
frames = loader.sample_adaptive(
    min_frames=8,           # Minimum frames
    max_frames=64,          # Maximum frames
    frames_per_second=0.5   # 0.5 frames per second of video
)
```

#### **`sample_temporal_chunks(num_chunks, frames_per_chunk) -> torch.Tensor`**

    Sample from temporal segments for better coverage.

```python
# 4 chunks × 2 frames/chunk = 8 frames spread across video
frames = loader.sample_temporal_chunks(num_chunks=4, frames_per_chunk=2)
```

### **Streaming**

#### **`stream_batches() -> Iterator[torch.Tensor]`**

    Stream video in batches for memory-efficient processing.

```python
for batch in loader.stream_batches():
    process(batch)  # Shape: (batch_size, C, H, W)
```


### **Utilities**

#### **`get_metadata() -> Dict[str, Any]`**

    Get comprehensive video metadata.

```python
metadata = loader.get_metadata()
# {'fps': 30.0, 'total_frames': 900, 'width': 1920, 'height': 1080, ...}
```

#### **`estimate_memory(num_frames, batch_size) -> float`**

    Estimate memory usage in MB.

```python
mem_mb = loader.estimate_memory(num_frames=8, batch_size=4)
```

## **Example: Complete pipeline**

```python
from src.ingestion import RoadVideoLoader
from src.configs import DecordConfig

# Initialize
config = DecordConfig(video_path='traffic_video.mp4', device='gpu')
loader = RoadVideoLoader(config)

# Get info
print(f"Video: {loader.duration:.1f}s, {loader.total_frames} frames @ {loader.fps:.1f} fps")

# Adaptive sampling (recommended)
frames = loader.sample_adaptive(min_frames=8, max_frames=64)
print(f"Sampled {len(frames)} frames at {frames.shape[2]}x{frames.shape[3]} resolution")

# Process frames with your model
# model(frames)
```
