"""
Reasoning Package for Road Buddy VQA.

Provides VLM integration (Gemini API), prompt building, and answer extraction.
"""

from .vlm_client import (
    VLMConfig,
    VLMResponse,
    GeminiVLMClient,
    MockVLMClient,
    create_vlm_client,
)

from .prompt_builder import (
    PromptStyle,
    PromptTemplate,
    build_mcq_prompt,
    build_prompt_from_sample,
    format_choices,
    format_context,
)

from .answer_extractor import (
    ExtractionResult,
    extract_answer,
    extract_answer_letter,
    batch_extract_answers,
)

__all__ = [
    # VLM Client (Gemini API)
    "VLMConfig",
    "VLMResponse",
    "GeminiVLMClient",
    "create_vlm_client",
    # Prompt Builder
    "PromptStyle",
    "PromptTemplate",
    "build_mcq_prompt",
    "build_prompt_from_sample",
    "format_choices",
    "format_context",
    # Answer Extractor
    "ExtractionResult",
    "extract_answer",
    "extract_answer_letter",
    "batch_extract_answers",
]
