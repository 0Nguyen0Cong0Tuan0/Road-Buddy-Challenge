# **Utility functions**

Helper functions for video validation, metadata extraction, and calculations.

## **Overview**

The `utils` module provides standalone functions that don't require class instantiation.

## **Functions**

#### **`validate_video(video_path: str) -> bool`**

    Check if a video file is valid.

```python
from src.ingestion import validate_video

if validate_video('video.mp4'):
    print("Valid video")
else:
    print("Invalid or missing video")
```

#### **`get_video_info(video_path: str, use_ffprobe: bool = True) -> Dict`**

    Extract video metadata without loading frames.

```python
from src.ingestion import get_video_info

info = get_video_info('video.mp4')
print(f"Resolution: {info['width']}x{info['height']}")
print(f"Duration: {info['duration']}s")
print(f"FPS: {info['fps']}")
print(f"Frames: {info['total_frames']}")
print(f"Size: {info['size_mb']:.2f} MB")

>>> {
    'width': 1920,
    'height': 1080,
    'fps': 30.0,
    'duration': 60.0,
    'total_frames': 1800,
    'codec': 'h264',
    'bitrate': 5000000,
    'size_mb': 45.2,
    'format': 'mp4'
}
```

#### **`estimate_memory(video_info, num_frames, batch_size, dtype) -> float`**

    Estimate memory usage for loading frames.

```python
from src.ingestion import get_video_info, estimate_memory

info = get_video_info('video.mp4')

# Estimate for 8 frames
mem = estimate_memory(info, num_frames=8, batch_size=4, dtype='float32')
print(f"Memory needed: {mem:.2f} MB")
```

#### **`convert_timestamp_to_frame(timestamp: float, fps: float) -> int`**

    Convert time in seconds to frame index.

```python
from src.ingestion import convert_timestamp_to_frame

frame = convert_timestamp_to_frame(2.5, fps=30.0)  # Returns 75
```

#### **`convert_frame_to_timestamp(frame_idx: int, fps: float) -> float`**

    Convert frame index to time in seconds.

```python
from src.ingestion import convert_frame_to_timestamp

time = convert_frame_to_timestamp(75, fps=30.0)  # Returns 2.5
```

#### **`expand_bbox(bbox, expansion_ratio, image_width, image_height) -> Tuple`**

    Expand a bounding box by a ratio while staying within image bounds.

```python
from src.ingestion import expand_bbox

bbox = (100, 100, 200, 200)  # (x1, y1, x2, y2)
expanded = expand_bbox(bbox, expansion_ratio=0.2, image_width=1920, image_height=1080)
# Returns larger box, clamped to image bounds
```

#### **`calculate_iou(bbox1, bbox2) -> float`**

    Calculate Intersection over Union between two bounding boxes.

```python
from src.ingestion import calculate_iou

bbox1 = (0, 0, 100, 100)
bbox2 = (50, 50, 150, 150)

iou = calculate_iou(bbox1, bbox2)
print(f"IoU: {iou:.3f}")  # 0.143 (14.3% overlap)
```


## **Example: Pre-processing check**

```python
from src.ingestion import validate_video, get_video_info, estimate_memory

video_path = 'data/videos/sample.mp4'

# Validate
if not validate_video(video_path):
    print("Invalid video!")
    exit(1)

# Get info
info = get_video_info(video_path)
print(f"Video: {info['width']}x{info['height']}, {info['duration']:.1f}s")

# Check memory
mem = estimate_memory(info, num_frames=32, batch_size=8)
print(f"Memory needed: {mem:.1f} MB")

# Safe to proceed
if mem < 1000:  # Less than 1GB
    print("Ready to process!")
```
