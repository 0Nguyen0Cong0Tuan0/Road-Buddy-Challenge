# Video Loading with Decord

import logging
import os
from typing import Iterator, Dict, Any
import torch
import numpy as np

try:
    from decord import VideoReader, cpu, gpu
    from decord import bridge
    bridge.set_bridge('torch')
    DECORD_AVAILABLE = True
except ImportError:
    DECORD_AVAILABLE = False
    logging.warning("Decord not available. Install via: pip install decord")

class RoadVideoLoader:
    """GPU-accelerated video loader using Decord."""

    def __init__(self, config):
        if not DECORD_AVAILABLE:
            raise ImportError("Decord library is not installed.")

        self.cfg = config

        if not os.path.exists(config.video_path):
            raise FileNotFoundError(f"Video path does not found: {config.video_path}")
        
        self.ctx = self._get_context(config.device, config.ctx_id)

        logging.info(f"Initializing Decord VideoReader for {config.video_path} on {self.ctx}")
        self.reader = VideoReader(
            config.video_path,
            ctx=self.ctx,
            width=config.width,
            height=config.height,
            num_threads=config.num_threads
        )

        self.total_frames = len(self.reader)
        self.fps = self.reader.get_avg_fps()
        self.duration = self.total_frames / self.fps if self.fps else 0

        logging.info(f"Video loaded: {self.total_frames} frames at {self.fps} fps, duration {self.duration:.2f} seconds.")

    def _get_context(self, device_str: str, device_id: int):
        if 'gpu' in device_str.lower() or 'cuda' in device_str.lower():
            if torch.cuda.is_available():
                return gpu(device_id)
            else:
                logging.warning("CUDA not available, falling back to CPU.")
                return cpu(0)
        return cpu(0)

    def stream_batches(self) -> Iterator[torch.Tensor]:
        """Yields batches of video frames on target device."""
        batch_size = self.cfg.batch_size

        for i in range(0, self.total_frames, batch_size):
            end_idx = min(i + batch_size, self.total_frames)
            indices = list(range(i, end_idx))

            try:
                batch_tensor = self.reader.get_batch(indices)
                # convert from (B, H, W, C) to (B, C, H, W) and normalize to [0, 1]
                batch_tensor = batch_tensor.permute(0, 3, 1, 2).float() / 255.0
                yield batch_tensor
            except Exception as e:
                logging.error(f"Error decoding batch at frames {i}-{end_idx}: {e}")
                continue
    
    def get_frame(self, frame_idx: int) -> torch.Tensor:
        """Get a single frame by index."""
        frame = self.reader[frame_idx]
        return frame.permute(2, 0, 1).float() / 255.0


    def get_metadata(self) -> Dict[str, Any]:
        """Return video metadata."""
        return {
            "fps": self.fps,
            "total_frames": self.total_frames,
            "duration": self.duration,
            "width": self.reader[0].shape[1],
            "height": self.reader[0].shape[0],
        }