# **BatchVideoProcessor**

Batch processing for video datasets with progress tracking and resume capability.

## **Overview**

`BatchVideoProcessor` processes multiple videos from a CSV dataset, with features like:
- Progress tracking with tqdm
- Checkpoint/resume capability
- Error logging and recovery
- Flexible processing functions

## **Initialization**

```python
from src.ingestion import BatchVideoProcessor
from src.configs import DecordConfig

config = DecordConfig(device='gpu')
processor = BatchVideoProcessor(
    config=config,
    num_workers=4,                    # Parallel workers
    checkpoint_path='checkpoint.json' # For resume capability
)
```

## **Processing a dataset**

```python
from src.ingestion.processor import extract_keyframes

results = processor.process_dataset(
    csv_path='data/videos.csv',       # CSV with video paths
    video_col='video_path',           # Column with paths
    process_fn=extract_keyframes,     # Processing function
    id_col='video_id',                # For resume (optional)
    output_dir='outputs/',            # Save results
    save_interval=100                 # Checkpoint frequency
)

print(results['stats'])
# {'total_videos': 1000, 'processed': 950, 'failed': 30, 'skipped': 20, ...}
```

## **Custom processing functions**

    Create custom functions to process each video:

```python
from src.ingestion import RoadVideoLoader
from typing import Dict, Any

def my_processing_function(loader: RoadVideoLoader) -> Dict[str, Any]:
    """Custom processing function.
    
    Args:
        loader: Video loader instance
        
    Returns:
        dict: Processing results
    """
    # Extract frames adaptively
    frames = loader.sample_adaptive(min_frames=8, max_frames=32)
    
    # Run your model
    # predictions = model(frames)
    
    return {
        "num_frames": len(frames),
        "predictions": [],  # Your results
        "metadata": loader.get_metadata()
    }

# Use your function
results = processor.process_dataset(
    csv_path='data/videos.csv',
    video_col='video_path',
    process_fn=my_processing_function
)
```

## **Built-in processing functions**

### **Extract keyframes**

    Extract uniform keyframes from videos.

```python
from src.ingestion.processor import extract_keyframes

results = processor.process_dataset(
    csv_path='data/videos.csv',
    video_col='video_path',
    process_fn=lambda loader: extract_keyframes(loader, num_frames=16)
)
```

## **ProcessingStats**

    Track processing statistics

```python
from src.ingestion import ProcessingStats

stats = ProcessingStats(
    total_videos=100,
    processed=90,
    failed=5,
    skipped=5
)

print(stats.to_dict())
# {
#     'total_videos': 100,
#     'processed': 90,
#     'failed': 5,
#     'skipped': 5,
#     'success_rate': 0.9,
#     'total_time': 120.5,
#     'avg_time_per_video': 1.34,
#     'errors': [...]
# }
```

## **Resume capability**

    Processing can be resumed from checkpoints:

```python
# First run (interrupted)
processor = BatchVideoProcessor(config, checkpoint_path='checkpoint.json')
results = processor.process_dataset(...)  # Processes 500 videos, then crashes

# Resume
processor = BatchVideoProcessor(config, checkpoint_path='checkpoint.json')
results = processor.process_dataset(...)  # Continues from video 501
```

## **Output files**

When `output_dir` is specified

```
outputs/
├── processing_results.json   # All results
└── processing_stats.json     # Statistics
```

## **Example: Complete pipeline**

```python
from src.ingestion import BatchVideoProcessor
from src.configs import DecordConfig

# Setup
config = DecordConfig(video_path='', device='gpu')
processor = BatchVideoProcessor(
    config=config,
    num_workers=4,
    checkpoint_path='processing_checkpoint.json'
)

# Custom processing
def process_for_qa(loader):
    frames = loader.sample_adaptive(min_frames=8, max_frames=64)
    metadata = loader.get_metadata()
    return {
        'frames': frames,
        'duration': metadata['duration'],
        'resolution': f"{metadata['width']}x{metadata['height']}"
    }

# Process dataset
results = processor.process_dataset(
    csv_path='data/train.csv',
    video_col='video_path',
    id_col='video_id',
    process_fn=process_for_qa,
    output_dir='processed_data/',
    save_interval=50
)

# Report
stats = results['stats']
print(f"Processed {stats['processed']}/{stats['total_videos']} videos")
print(f"Success rate: {stats['success_rate']:.1%}")
print(f"Time: {stats['total_time']:.1f}s ({stats['avg_time_per_video']:.2f}s/video)")
```
