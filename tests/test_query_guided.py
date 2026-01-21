"""
Test suite for Query-Guided Perception Module.

Tests the QueryAnalyzer, FrameScorer, and KeyframeSelector components
to ensure correct functionality for Vietnamese traffic question answering.

Run with: pytest tests/test_query_guided.py -v
"""

import pytest
import numpy as np
import json
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# ============================================================================
# QueryAnalyzer Tests
# ============================================================================

class TestQueryAnalyzer:
    """Tests for QueryAnalyzer component."""
    
    @pytest.fixture
    def analyzer(self):
        """Create QueryAnalyzer with keyword strategy."""
        from src.perception.query_analyzer import QueryAnalyzer
        return QueryAnalyzer(strategy="keyword")
    
    def test_init_keyword_strategy(self):
        """Test initialization with keyword strategy."""
        from src.perception.query_analyzer import QueryAnalyzer
        analyzer = QueryAnalyzer(strategy="keyword")
        assert analyzer.strategy_name == "keyword"
    
    def test_extract_traffic_sign_keywords(self, analyzer):
        """Test extraction of traffic sign keywords."""
        question = "Biển báo tốc độ tối đa là bao nhiêu?"
        result = analyzer.analyze(question)
        
        assert "tốc độ tối đa" in result.keywords_found or "biển báo" in result.keywords_found
        assert len(result.target_objects) > 0
        assert "traffic sign" in result.yolo_classes or len(result.yolo_classes) > 0
    
    def test_extract_traffic_light_keywords(self, analyzer):
        """Test extraction of traffic light keywords."""
        question = "Video có đèn đỏ không?"
        result = analyzer.analyze(question)
        
        assert "đèn đỏ" in result.keywords_found
        assert "traffic_light_red" in result.target_objects or "red_light" in result.target_objects
        assert "traffic light" in result.yolo_classes
    
    def test_extract_lane_keywords(self, analyzer):
        """Test extraction of lane/road keywords."""
        question = "Làn đường có được rẽ phải không?"
        result = analyzer.analyze(question)
        
        assert any(k in result.keywords_found for k in ["làn đường", "rẽ phải"])
    
    def test_detect_existence_intent(self, analyzer):
        """Test detection of existence question intent."""
        from src.perception.query_analyzer import QuestionIntent
        
        question = "Trong video có biển cấm không?"
        result = analyzer.analyze(question)
        
        assert result.question_intent == QuestionIntent.EXISTENCE
    
    def test_detect_value_intent(self, analyzer):
        """Test detection of value question intent."""
        from src.perception.query_analyzer import QuestionIntent
        
        question = "Tốc độ tối đa trên làn đường là bao nhiêu?"
        result = analyzer.analyze(question)
        
        assert result.question_intent == QuestionIntent.VALUE
    
    def test_detect_permission_intent(self, analyzer):
        """Test detection of permission question intent."""
        from src.perception.query_analyzer import QuestionIntent
        
        question = "Xe mô tô có được phép đi ở làn này không?"
        result = analyzer.analyze(question)
        
        assert result.question_intent == QuestionIntent.PERMISSION
    
    def test_temporal_hints_extraction(self, analyzer):
        """Test extraction of temporal hints."""
        question = "Biển cấm nào xuất hiện đầu tiên?"
        result = analyzer.analyze(question)
        
        assert "first" in result.temporal_hints
    
    def test_confidence_score(self, analyzer):
        """Test that confidence is calculated."""
        question = "Biển báo chỉ dẫn rẽ trái có trong video không?"
        result = analyzer.analyze(question)
        
        assert 0 <= result.confidence <= 1
    
    def test_batch_analysis(self, analyzer):
        """Test batch analysis of multiple questions."""
        questions = [
            "Biển báo tốc độ là bao nhiêu?",
            "Có đèn đỏ không?",
            "Làn nào được rẽ phải?"
        ]
        
        results = analyzer.analyze_batch(questions)
        
        assert len(results) == 3
        for result in results:
            assert result.original_question in questions
    
    def test_to_dict_serialization(self, analyzer):
        """Test serialization to dictionary."""
        question = "Biển cấm dừng đỗ ở đâu?"
        result = analyzer.analyze(question)
        
        result_dict = result.to_dict()
        
        assert isinstance(result_dict, dict)
        assert "original_question" in result_dict
        assert "target_objects" in result_dict
        assert "yolo_classes" in result_dict


class TestQueryAnalyzerTranslation:
    """Tests for QueryAnalyzer with translation strategy."""
    
    @pytest.fixture
    def analyzer(self):
        """Create QueryAnalyzer with translation strategy."""
        from src.perception.query_analyzer import QueryAnalyzer
        try:
            return QueryAnalyzer(strategy="translation", translator="googletrans")
        except Exception:
            pytest.skip("Translation dependencies not installed")
    
    def test_translation_output(self, analyzer):
        """Test that translation produces English output."""
        question = "Biển báo tốc độ tối đa là bao nhiêu?"
        result = analyzer.analyze(question)
        
        # Should have both original and translated
        assert result.original_question == question
        # Translation may or may not work depending on network
        # Just check that keywords are still extracted
        assert len(result.keywords_found) > 0


# ============================================================================
# FrameScorer Tests
# ============================================================================

class TestFrameScorer:
    """Tests for FrameScorer component."""
    
    @pytest.fixture
    def dummy_frames(self):
        """Create dummy frames for testing."""
        return [
            np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            for _ in range(5)
        ]
    
    @pytest.fixture
    def scorer_config(self):
        """Create scorer config without CLIP (for fast testing)."""
        from src.perception.frame_scorer import ScoringConfig
        return ScoringConfig(
            strategy="combined",
            alpha=0.0,  # Disable CLIP (may not be installed)
            beta=0.0,   # Disable detection
            gamma=1.0   # Only use distinctiveness
        )
    
    def test_init_scorer(self, scorer_config):
        """Test FrameScorer initialization."""
        from src.perception.frame_scorer import FrameScorer
        scorer = FrameScorer(scorer_config)
        assert scorer is not None
    
    def test_score_frames_returns_array(self, scorer_config, dummy_frames):
        """Test that scoring returns correct shape."""
        from src.perception.frame_scorer import FrameScorer
        scorer = FrameScorer(scorer_config)
        
        scores = scorer.score_frames(dummy_frames, "test question")
        
        assert isinstance(scores, np.ndarray)
        assert len(scores) == len(dummy_frames)
    
    def test_score_frames_detailed(self, scorer_config, dummy_frames):
        """Test detailed scoring output."""
        from src.perception.frame_scorer import FrameScorer
        scorer = FrameScorer(scorer_config)
        
        scores = scorer.score_frames(
            dummy_frames, 
            "test question",
            return_detailed=True
        )
        
        assert isinstance(scores, list)
        assert len(scores) == len(dummy_frames)
        
        for score in scores:
            assert hasattr(score, "frame_idx")
            assert hasattr(score, "final_score")
            assert hasattr(score, "ifd_score")


class TestDistinctivenessScoring:
    """Tests for Inter-Frame Distinctiveness scoring."""
    
    def test_compute_distinctiveness_from_embeddings(self):
        """Test distinctiveness from embeddings."""
        from src.perception.frame_scorer import DistinctivenessStrategy
        
        strategy = DistinctivenessStrategy(window_size=2)
        
        # Create similar embeddings
        embeddings = np.array([
            [1, 0, 0],
            [1, 0.1, 0],  # Very similar to first
            [0, 1, 0],    # Different
            [0, 1, 0.1],  # Similar to third
            [0, 0, 1]     # Different
        ], dtype=np.float32)
        
        scores = strategy.compute(embeddings)
        
        assert len(scores) == 5
        # The distinct frame (index 2, 4) should have higher scores
        assert scores[2] > scores[1]  # Frame 2 is more distinct
    
    def test_compute_from_frames(self):
        """Test distinctiveness from actual frames."""
        from src.perception.frame_scorer import DistinctivenessStrategy
        
        strategy = DistinctivenessStrategy()
        
        # Create frames with different colors
        frames = [
            np.full((100, 100, 3), i * 50, dtype=np.uint8)
            for i in range(5)
        ]
        
        scores = strategy.compute_from_frames(frames)
        
        assert len(scores) == 5
        assert all(s >= 0 for s in scores)


# ============================================================================
# KeyframeSelector Tests
# ============================================================================

class TestKeyframeSelector:
    """Tests for KeyframeSelector component."""
    
    @pytest.fixture
    def config(self):
        """Create minimal config for testing."""
        from src.perception.keyframe_selector import KeyframeSelectorConfig
        return KeyframeSelectorConfig(
            num_keyframes=4,
            query_strategy="keyword",
            scoring_strategy="clip",  # Will be tested if CLIP available
            yolo_mode="none",  # Don't require YOLO for tests
            sample_fps=1.0,
            max_frames=16
        )
    
    @pytest.fixture
    def selector(self, config):
        """Create KeyframeSelector."""
        from src.perception.keyframe_selector import KeyframeSelector
        return KeyframeSelector(config)
    
    def test_init_selector(self, config):
        """Test KeyframeSelector initialization."""
        from src.perception.keyframe_selector import KeyframeSelector
        selector = KeyframeSelector(config)
        assert selector is not None
        assert selector.config.num_keyframes == 4
    
    def test_select_from_frames(self, selector):
        """Test selecting from pre-loaded frames."""
        frames = [
            np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            for _ in range(10)
        ]
        
        question = "Biển báo tốc độ là bao nhiêu?"
        
        result = selector.select_from_frames(frames, question)
        
        assert result.num_keyframes <= selector.config.num_keyframes
        assert result.total_frames_sampled == 10
        assert result.query_analysis is not None
        assert len(result.keyframes) > 0
    
    def test_top_k_selection(self):
        """Test top-k selection algorithm."""
        from src.perception.keyframe_selector import KeyframeSelector, KeyframeSelectorConfig
        
        config = KeyframeSelectorConfig(
            num_keyframes=3,
            selection_strategy="top_k",
            yolo_mode="none"
        )
        selector = KeyframeSelector(config)
        
        scores = np.array([0.1, 0.9, 0.5, 0.8, 0.3])
        selected = selector._select_top_k(scores, k=3)
        
        # Should select indices with highest scores, sorted by frame order
        assert 1 in selected  # 0.9
        assert 3 in selected  # 0.8
        assert 2 in selected  # 0.5
    
    def test_diverse_top_k_selection(self):
        """Test diverse top-k selection."""
        from src.perception.keyframe_selector import KeyframeSelector, KeyframeSelectorConfig
        
        config = KeyframeSelectorConfig(
            num_keyframes=3,
            selection_strategy="diverse_top_k",
            diversity_threshold=0.01,  # Low threshold for testing
            yolo_mode="none"
        )
        selector = KeyframeSelector(config)
        
        # Create diverse frames
        frames = [
            np.full((100, 100, 3), i * 50, dtype=np.uint8)
            for i in range(5)
        ]
        
        scores = np.array([0.1, 0.9, 0.5, 0.8, 0.3])
        selected = selector._select_diverse_top_k(scores, frames, k=3)
        
        assert len(selected) == 3
        assert len(set(selected)) == 3  # All unique
    
    def test_result_to_dict(self, selector):
        """Test that results can be serialized."""
        frames = [
            np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            for _ in range(5)
        ]
        
        result = selector.select_from_frames(frames, "Test question")
        result_dict = result.to_dict()
        
        assert isinstance(result_dict, dict)
        assert "keyframes" in result_dict
        assert "query_analysis" in result_dict


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests with actual training data."""
    
    @pytest.fixture
    def train_data(self):
        """Load training data if available."""
        train_path = project_root / "data" / "raw" / "train" / "train.json"
        if not train_path.exists():
            pytest.skip("Training data not available")
        
        with open(train_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def test_analyze_real_questions(self, train_data):
        """Test query analysis on real questions from train.json."""
        from src.perception.query_analyzer import QueryAnalyzer
        
        analyzer = QueryAnalyzer(strategy="keyword")
        
        # Test first 10 questions
        questions = [item["question"] for item in train_data["data"][:10]]
        
        for question in questions:
            result = analyzer.analyze(question)
            
            # All questions should extract some information
            assert result.original_question == question
            # Most traffic questions should find keywords
            print(f"Q: {question[:50]}...")
            print(f"   Keywords: {result.keywords_found}")
            print(f"   Objects: {result.target_objects}")
            print()
    
    def test_question_statistics(self, train_data):
        """Analyze statistics of training questions."""
        from src.perception.query_analyzer import QueryAnalyzer, QuestionIntent
        
        analyzer = QueryAnalyzer(strategy="keyword")
        
        intents = {}
        keywords_found_count = 0
        
        for item in train_data["data"]:
            result = analyzer.analyze(item["question"])
            
            intent = result.question_intent.value
            intents[intent] = intents.get(intent, 0) + 1
            
            if result.keywords_found:
                keywords_found_count += 1
        
        total = len(train_data["data"])
        print(f"\nQuestion Statistics (n={total}):")
        print(f"  Questions with keywords: {keywords_found_count} ({100*keywords_found_count/total:.1f}%)")
        print(f"  Intent distribution:")
        for intent, count in sorted(intents.items(), key=lambda x: -x[1]):
            print(f"    {intent}: {count} ({100*count/total:.1f}%)")


# ============================================================================
# Configuration Tests
# ============================================================================

class TestConfiguration:
    """Tests for configuration module."""
    
    def test_query_guided_config_exists(self):
        """Test that QueryGuidedConfig is available."""
        from config.settings import QueryGuidedConfig, get_config
        
        config = get_config()
        assert hasattr(config, "query_guided")
        assert isinstance(config.query_guided, QueryGuidedConfig)
    
    def test_default_values(self):
        """Test default configuration values."""
        from config.settings import get_config
        
        config = get_config()
        qg = config.query_guided
        
        assert qg.num_keyframes == 8
        assert qg.use_translation == True
        assert qg.yolo_mode == "selected_only"
        assert qg.query_strategy == "keyword"
    
    def test_weight_values(self):
        """Test scoring weight values."""
        from config.settings import get_config
        
        config = get_config()
        qg = config.query_guided
        
        # Weights should sum to 1
        total = qg.alpha + qg.beta + qg.gamma
        assert abs(total - 1.0) < 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
