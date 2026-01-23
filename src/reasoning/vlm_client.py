"""
VLM Client for Road Buddy VQA.

Provides Gemini API integration for Vision-Language Model inference.

Usage:
    from src.reasoning.vlm_client import create_vlm_client, VLMConfig
    
    client = create_vlm_client(model_name="gemini-2.0-flash")
    response = client.generate(frames, prompt)
    print(response.text)
"""

import os
import logging
import base64
from io import BytesIO
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union
from abc import ABC, abstractmethod

import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class VLMConfig:
    """
    Configuration for VLM client.
    
    Attributes:
        model: Model name (e.g., "gemini-2.0-flash", "gemini-1.5-pro")
        api_key: API key
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        device: Device
    """
    model: str = "gemini-2.0-flash"
    api_key: Optional[str] = None
    max_tokens: int = 256
    temperature: float = 0.1
    device: str = "auto"
    use_quantization: bool = False
    
    def __post_init__(self):
        if self.api_key is None:
            self.api_key = os.environ.get("GOOGLE_API_KEY")

@dataclass
class VLMResponse:
    """
    Response from VLM generation.
    
    Attributes:
        text: Generated text response
        model: Model used for generation
        usage: Token usage statistics
        finish_reason: Reason for generation completion
    """
    text: str
    model: str = ""
    usage: Dict[str, int] = field(default_factory=dict)
    finish_reason: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "model": self.model,
            "usage": self.usage,
            "finish_reason": self.finish_reason,
        }

class BaseVLMClient(ABC):
    """Abstract base class for VLM clients."""
    
    @abstractmethod
    def generate(
        self,
        frames: List[np.ndarray],
        prompt: str,
        **kwargs
    ) -> VLMResponse:
        """Generate response from frames and prompt."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if client is ready."""
        pass

# Gemini Client
class GeminiVLMClient(BaseVLMClient):
    """
    Gemini Vision-Language Model client.
    
    Uses Google's Gemini API for multimodal inference.
    Supports: gemini-2.0-flash, gemini-1.5-flash, gemini-1.5-pro
    """
    
    SUPPORTED_MODELS = [
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
        "gemini-1.5-flash",
        "gemini-1.5-flash-8b",
        "gemini-1.5-pro",
    ]
    
    def __init__(self, config: Optional[VLMConfig] = None):
        """Initialize Gemini client."""
        self.config = config or VLMConfig()
        self._client = None
        self._model = None
        self._available = False
        self._initialize()
    
    def _initialize(self):
        """Initialize Google Generative AI client."""
        try:
            import google.generativeai as genai
            
            if not self.config.api_key:
                logger.warning("No API key provided. Set GOOGLE_API_KEY env var.")
                return
            
            genai.configure(api_key=self.config.api_key)
            
            # Get model
            model_name = self.config.model
            if not model_name.startswith("models/"):
                model_name = f"models/{model_name}"
            
            self._model = genai.GenerativeModel(
                model_name=model_name,
                generation_config=genai.GenerationConfig(
                    max_output_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                )
            )
            
            self._available = True
            logger.info(f"Gemini client initialized: {self.config.model}")
            
        except ImportError:
            logger.error("google-generativeai not installed.")
            logger.error("Install with: pip install google-generativeai")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {e}")
    
    def _encode_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """Encode numpy frame to Gemini-compatible format."""
        try:
            from PIL import Image
            
            # Ensure RGB format
            if frame.ndim == 2:  # Grayscale
                frame = np.stack([frame] * 3, axis=-1)
            elif frame.shape[-1] == 4:  # RGBA
                frame = frame[:, :, :3]
            
            # Convert to PIL Image
            image = Image.fromarray(frame.astype(np.uint8))
            
            # Encode to JPEG bytes
            buffer = BytesIO()
            image.save(buffer, format="JPEG", quality=85)
            image_bytes = buffer.getvalue()
            
            return {
                "mime_type": "image/jpeg",
                "data": image_bytes,
            }
            
        except Exception as e:
            logger.error(f"Failed to encode frame: {e}")
            raise
    
    def generate(self, frames: List[np.ndarray], prompt: str, **kwargs) -> VLMResponse:
        """Generate response from frames and prompt."""
        if not self._available:
            logger.warning("Gemini client not available, returning empty response")
            return VLMResponse(text="", model=self.config.model)
        
        try:
            # Build content parts
            parts = []
            
            # Add frames as images
            for i, frame in enumerate(frames):
                image_data = self._encode_frame(frame)
                parts.append(image_data)
            
            # Add text prompt
            parts.append(prompt)
            
            # Generate response
            response = self._model.generate_content(parts)
            
            # Extract text
            text = ""
            if response.parts:
                text = response.text
            
            # Build usage dict
            usage = {}
            if hasattr(response, 'usage_metadata'):
                usage = {
                    "prompt_tokens": getattr(response.usage_metadata, 'prompt_token_count', 0),
                    "completion_tokens": getattr(response.usage_metadata, 'candidates_token_count', 0),
                    "total_tokens": getattr(response.usage_metadata, 'total_token_count', 0),
                }
            
            # Get finish reason
            finish_reason = ""
            if response.candidates:
                finish_reason = str(response.candidates[0].finish_reason)
            
            return VLMResponse(
                text=text,
                model=self.config.model,
                usage=usage,
                finish_reason=finish_reason,
            )
            
        except Exception as e:
            logger.error(f"Gemini generation failed: {e}")
            return VLMResponse(text="", model=self.config.model)
    
    def is_available(self) -> bool:
        return self._available
    
    def __repr__(self) -> str:
        return f"GeminiVLMClient(model={self.config.model}, available={self._available})"

# Factory Function
def create_vlm_client(model_name: str = "gemini-2.0-flash", device: str = "auto", api_key: Optional[str] = None, max_tokens: int = 256, temperature: float = 0.1, use_quantization: bool = False, **kwargs) -> BaseVLMClient:
    """
    Create VLM client.
    
    Args:
        model_name: Model to use. Options:
            - "gemini-2.0-flash" (default, fast)
            - "gemini-1.5-flash" (fast, good quality)
            - "gemini-1.5-pro" (best quality)
        device: Device
        api_key: Google API key (or set GOOGLE_API_KEY env var)
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        use_quantization: Kept for compatibility
        **kwargs: Additional config options
    """
    config = VLMConfig(
        model=model_name,
        api_key=api_key,
        max_tokens=max_tokens,
        temperature=temperature,
        device=device,
        use_quantization=use_quantization,
    )
    
    if model_name == "mock":
        return MockVLMClient(config)
    elif model_name.startswith("gemini"):
        return GeminiVLMClient(config)
    else:
        logger.warning(f"Unknown model '{model_name}', using Gemini")
        return GeminiVLMClient(config)