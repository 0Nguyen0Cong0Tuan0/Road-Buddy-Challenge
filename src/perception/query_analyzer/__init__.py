"""
Query Analyzer Package for Vietnamese Traffic Questions.

This package provides tools to parse Vietnamese traffic-related questions
and extract:
1. Target objects (signs, lanes, traffic lights, etc.)
2. Question intent (existence, direction, speed, legality)
3. Temporal hints (first, current, etc.)

Strategies (configurable as plugins):
- KeywordExtractor: Fast rule-based extraction
- TranslationExtractor: Translate to English for CLIP
- SemanticExtractor: Use PhoBERT/embeddings

Usage:
    from src.perception.query_analyzer import QueryAnalyzer, QueryAnalysisResult
    
    analyzer = QueryAnalyzer(strategy="keyword")  # or "translation", "semantic"
    result = analyzer.analyze("Biển báo tốc độ tối đa là bao nhiêu?")
    print(result.target_objects)  # ['speed_limit_sign']
    print(result.question_intent)  # 'value_query'
"""

from .constants import (
    QuestionIntent,
    VIETNAMESE_TRAFFIC_KEYWORDS,
    INTENT_PATTERNS,
    INTENT_PATTERNS_ORDERED,
    TEMPORAL_KEYWORDS,
    SEMANTIC_OBJECT_DESCRIPTIONS,
)
from .models import QueryAnalysisResult
from .base import ExtractionStrategy
from .analyzer import QueryAnalyzer, get_available_strategies, get_yolo_class_mapping
from .strategies import (
    KeywordExtractionStrategy,
    TranslationExtractionStrategy,
    SemanticExtractionStrategy,
)

__all__ = [
    # Main classes
    "QueryAnalyzer",
    "QueryAnalysisResult",
    "QuestionIntent",
    "ExtractionStrategy",
    
    # Strategies
    "KeywordExtractionStrategy",
    "TranslationExtractionStrategy",
    "SemanticExtractionStrategy",
    
    # Utilities
    "get_available_strategies",
    "get_yolo_class_mapping",
    
    # Constants
    "VIETNAMESE_TRAFFIC_KEYWORDS",
    "INTENT_PATTERNS",
    "INTENT_PATTERNS_ORDERED",
    "TEMPORAL_KEYWORDS",
    "SEMANTIC_OBJECT_DESCRIPTIONS",
]
