"""
Standalone test script for video loader module.

Usage:
    python test_loader.py <video_path> [--output_dir OUTPUT_DIR] [--save_images]
    
Example:
    python test_loader.py data\\raw\\train\\videos\\0b990034_129_clip_007_0046_0055_N.mp4 --save_images
    python test_loader.py data\\raw\\train\\videos\\0b990034_129_clip_007_0046_0055_N.mp4 --output_dir ./debug_output
"""

import sys
import argparse
import torchvision
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.loader import RoadVideoLoader
from src.configs import DecordConfig


def save_frame(frame, output_dir: Path, filename: str):
    """Save a single frame as PNG image."""
    filepath = output_dir / filename
    torchvision.utils.save_image(frame, str(filepath))
    return filepath


def save_batch_as_grid(frames, output_dir: Path, filename: str, nrow: int = 4):
    """Save a batch of frames as a grid image."""
    filepath = output_dir / filename
    torchvision.utils.save_image(frames, str(filepath), nrow=nrow, padding=2)
    return filepath


def save_batch_individual(frames, output_dir: Path, prefix: str):
    """Save each frame in a batch as individual images."""
    saved_paths = []
    for i, frame in enumerate(frames):
        filepath = output_dir / f"{prefix}_{i:03d}.png"
        torchvision.utils.save_image(frame, str(filepath))
        saved_paths.append(filepath)
    return saved_paths


def test_loader(video_path: str, output_dir: Path, save_images: bool = True):
    """Test video loader with a real video file."""
    print("=" * 70)
    print("TESTING VIDEO LOADER")
    print("=" * 70)
    print(f"\nVideo path: {video_path}")
    print(f"Output dir: {output_dir}")
    print(f"Save images: {save_images}\n")
    
    # Create output directory
    if save_images:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"✓ Output directory created: {output_dir}\n")
    
    # Get video name for file naming
    video_name = Path(video_path).stem
    
    # Test 1: Initialization
    print("Test 1: Initializing video loader...")
    try:
        config = DecordConfig(
            video_path=video_path,
            batch_size=16,
            device='cpu',  # Change to 'gpu' if you have CUDA
            ctx_id=0,
            width=-1,
            height=-1,
            num_threads=0
        )
        loader = RoadVideoLoader(config)
        print(f"  ✓ SUCCESS - Backend: {loader.backend}")
        print(f"    - Total frames: {loader.total_frames}")
        print(f"    - FPS: {loader.fps:.2f}")
        print(f"    - Duration: {loader.duration:.2f}s")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        return False
    
    # Test 2: Get metadata
    print("\nTest 2: Extracting metadata...")
    try:
        metadata = loader.get_metadata()
        print(f"  ✓ SUCCESS")
        print(f"    - Resolution: {metadata['width']}x{metadata['height']}")
        print(f"    - Size: {metadata['size_mb']:.2f} MB")
        print(f"    - Backend: {metadata['backend']}")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        return False
    
    # Test 3: Get single frame
    print("\nTest 3: Extracting single frame...")
    try:
        frame = loader.get_frame(0)
        print(f"  ✓ SUCCESS")
        print(f"    - Frame shape: {frame.shape}")
        print(f"    - Frame dtype: {frame.dtype}")
        print(f"    - Value range: [{frame.min():.3f}, {frame.max():.3f}]")
        
        if save_images:
            saved_path = save_frame(frame, output_dir, f"{video_name}_test3_single_frame.png")
            print(f"    - Saved: {saved_path}")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        return False
    
    # Test 4: Uniform sampling
    print("\nTest 4: Uniform sampling (8 frames)...")
    try:
        frames = loader.sample_uniform(8)
        print(f"  ✓ SUCCESS")
        print(f"    - Frames shape: {frames.shape}")
        print(f"    - Memory usage: ~{frames.numel() * 4 / (1024**2):.2f} MB")
        
        if save_images:
            # Save as grid
            grid_path = save_batch_as_grid(frames, output_dir, f"{video_name}_test4_uniform_grid.png", nrow=4)
            print(f"    - Saved grid: {grid_path}")
            
            # Save individual frames
            individual_dir = output_dir / f"{video_name}_test4_uniform_individual"
            individual_dir.mkdir(exist_ok=True)
            saved_paths = save_batch_individual(frames, individual_dir, "frame")
            print(f"    - Saved {len(saved_paths)} individual frames to: {individual_dir}")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        return False
    
    # Test 5: FPS sampling
    print("\nTest 5: FPS-based sampling (1 FPS)...")
    try:
        frames = loader.sample_fps(1.0)
        print(f"  ✓ SUCCESS")
        print(f"    - Frames extracted: {frames.shape[0]}")
        print(f"    - Expected frames: ~{int(loader.duration)}")
        
        if save_images:
            # Save as grid
            grid_path = save_batch_as_grid(frames, output_dir, f"{video_name}_test5_fps_grid.png", nrow=4)
            print(f"    - Saved grid: {grid_path}")
            
            # Save individual frames
            individual_dir = output_dir / f"{video_name}_test5_fps_individual"
            individual_dir.mkdir(exist_ok=True)
            saved_paths = save_batch_individual(frames, individual_dir, "frame")
            print(f"    - Saved {len(saved_paths)} individual frames to: {individual_dir}")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        return False
    
    # Test 6: Custom indices sampling
    print("\nTest 6: Custom indices sampling...")
    try:
        max_idx = min(loader.total_frames - 1, 50)
        indices = [0, max_idx // 4, max_idx // 2, 3 * max_idx // 4, max_idx]
        frames = loader.sample_indices(indices)
        print(f"  ✓ SUCCESS")
        print(f"    - Indices: {indices}")
        print(f"    - Frames extracted: {frames.shape[0]}")
        
        if save_images:
            # Save as grid
            grid_path = save_batch_as_grid(frames, output_dir, f"{video_name}_test6_indices_grid.png", nrow=5)
            print(f"    - Saved grid: {grid_path}")
            
            # Save individual frames with index in filename
            for i, (idx, frame) in enumerate(zip(indices, frames)):
                saved_path = save_frame(frame, output_dir, f"{video_name}_test6_frame_idx{idx:04d}.png")
            print(f"    - Saved {len(indices)} individual frames with index in filename")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        return False
    
    # Test 7: Preprocessing
    print("\nTest 7: Frame preprocessing...")
    try:
        frame = loader.get_frame(0)
        preprocessed = loader.preprocess_frame(
            frame,
            resize=(224, 224),
            normalize=True
        )
        print(f"  ✓ SUCCESS")
        print(f"    - Original shape: {frame.shape}")
        print(f"    - Preprocessed shape: {preprocessed.shape}")
        print(f"    - Normalized (not in [0,1]): {not (0 <= preprocessed.min() <= preprocessed.max() <= 1)}")
        
        if save_images:
            # Save original
            save_frame(frame, output_dir, f"{video_name}_test7_original.png")
            
            # Save resized (without normalization for visualization)
            resized = loader.preprocess_frame(frame, resize=(224, 224), normalize=False)
            save_frame(resized, output_dir, f"{video_name}_test7_resized_224x224.png")
            
            print(f"    - Saved original and resized frames")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        return False
    
    # Test 8: Batch streaming
    print("\nTest 8: Batch streaming...")
    try:
        batch_count = 0
        total_frames = 0
        first_batch = None
        
        for batch in loader.stream_batches():
            batch_count += 1
            total_frames += batch.shape[0]
            if batch_count == 1:
                first_batch = batch
                print(f"    - First batch shape: {batch.shape}")
        
        print(f"  ✓ SUCCESS")
        print(f"    - Total batches: {batch_count}")
        print(f"    - Total frames streamed: {total_frames}")
        
        if save_images and first_batch is not None:
            # Save first batch as grid
            grid_path = save_batch_as_grid(first_batch, output_dir, f"{video_name}_test8_first_batch_grid.png", nrow=4)
            print(f"    - Saved first batch grid: {grid_path}")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        return False
    
    # Test 9: Memory estimation
    print("\nTest 9: Memory estimation...")
    try:
        mem_8_frames = loader.estimate_memory(8, batch_size=4)
        mem_all_frames = loader.estimate_memory(loader.total_frames, batch_size=16)
        print(f"  ✓ SUCCESS")
        print(f"    - 8 frames (batch=4): {mem_8_frames:.2f} MB")
        print(f"    - All frames (batch=16): {mem_all_frames:.2f} MB")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        return False
    
    # Summary
    print("\n" + "=" * 70)
    print("ALL TESTS PASSED ✓")
    print("=" * 70)
    
    if save_images:
        print(f"\n📁 All saved images are in: {output_dir.absolute()}")
        
        # List saved files
        saved_files = list(output_dir.glob("*.png"))
        saved_dirs = [d for d in output_dir.iterdir() if d.is_dir()]
        
        print(f"   - {len(saved_files)} image files")
        print(f"   - {len(saved_dirs)} subdirectories with individual frames")
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Test video loader with real video')
    parser.add_argument('video_path', type=str, help='Path to video file')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save output images (default: tests/debug_output/<video_name>_<timestamp>)')
    parser.add_argument('--save_images', action='store_true', default=True,
                        help='Save extracted frames as images (default: True)')
    parser.add_argument('--no_save', action='store_true',
                        help='Disable saving images')
    
    args = parser.parse_args()
    
    # Validate video exists
    video_path = Path(args.video_path)
    if not video_path.exists():
        print(f"ERROR: Video file not found: {args.video_path}")
        sys.exit(1)
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(__file__).parent / "debug_output" / f"{video_path.stem}_{timestamp}"
    
    # Run tests
    save_images = not args.no_save
    success = test_loader(str(video_path), output_dir, save_images)
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
