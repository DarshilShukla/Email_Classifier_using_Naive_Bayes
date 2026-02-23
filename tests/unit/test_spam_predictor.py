"""
Unit tests for SpamPredictor class.
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock
from src.spam_predictor import SpamPredictor


class TestSpamPredictorInit:
    """Test SpamPredictor initialization."""
    
    def test_init_with_valid_components(self):
        """Test initialization with valid model, vectorizer, and preprocessor."""
        model = Mock()
        vectorizer = Mock()
        preprocessor = Mock()
        
        predictor = SpamPredictor(model, vectorizer, preprocessor)
        
        assert predictor.model is model
        assert predictor.vectorizer is vectorizer
        assert predictor.preprocessor is preprocessor
    
    def test_init_with_none_model(self):
        """Test initialization fails when model is None."""
        vectorizer = Mock()
        preprocessor = Mock()
        
        with pytest.raises(ValueError, match="Model cannot be None"):
            SpamPredictor(None, vectorizer, preprocessor)
    
    def test_init_with_none_vectorizer(self):
        """Test initialization fails when vectorizer is None."""
        model = Mock()
        preprocessor = Mock()
        
        with pytest.raises(ValueError, match="Vectorizer cannot be None"):
            SpamPredictor(model, None, preprocessor)
    
    def test_init_with_none_preprocessor(self):
        """Test initialization fails when preprocessor is None."""
        model = Mock()
        vectorizer = Mock()
        
        with pytest.raises(ValueError, match="Preprocessor cannot be None"):
            SpamPredictor(model, vectorizer, None)


class TestSpamPredictorPredict:
    """Test SpamPredictor predict method."""
    
    def test_predict_ham_message(self):
        """Test predicting a ham message."""
        # Setup mocks
        model = Mock()
        model.predict.return_value = np.array([0])
        
        vectorizer = Mock()
        vectorizer.transform.return_value = Mock()
        
        preprocessor = Mock()
        preprocessor.preprocess_text.return_value = "hello world"
        
        predictor = SpamPredictor(model, vectorizer, preprocessor)
        
        # Test prediction
        result = predictor.predict("Hello World!")
        
        assert result == "ham"
        preprocessor.preprocess_text.assert_called_once_with("Hello World!")
        vectorizer.transform.assert_called_once_with(["hello world"])
    
    def test_predict_spam_message(self):
        """Test predicting a spam message."""
        # Setup mocks
        model = Mock()
        model.predict.return_value = np.array([1])
        
        vectorizer = Mock()
        vectorizer.transform.return_value = Mock()
        
        preprocessor = Mock()
        preprocessor.preprocess_text.return_value = "free prize win"
        
        predictor = SpamPredictor(model, vectorizer, preprocessor)
        
        # Test prediction
        result = predictor.predict("FREE PRIZE! WIN NOW!")
        
        assert result == "spam"
        preprocessor.preprocess_text.assert_called_once_with("FREE PRIZE! WIN NOW!")
        vectorizer.transform.assert_called_once_with(["free prize win"])
    
    def test_predict_with_non_string_input(self):
        """Test predict raises TypeError for non-string input."""
        model = Mock()
        vectorizer = Mock()
        preprocessor = Mock()
        
        predictor = SpamPredictor(model, vectorizer, preprocessor)
        
        with pytest.raises(TypeError, match="Message must be a string"):
            predictor.predict(123)
    
    def test_predict_with_empty_string(self):
        """Test predict handles empty string."""
        # Setup mocks
        model = Mock()
        model.predict.return_value = np.array([0])
        
        vectorizer = Mock()
        vectorizer.transform.return_value = Mock()
        
        preprocessor = Mock()
        preprocessor.preprocess_text.return_value = ""
        
        predictor = SpamPredictor(model, vectorizer, preprocessor)
        
        # Test prediction
        result = predictor.predict("")
        
        assert result == "ham"
        preprocessor.preprocess_text.assert_called_once_with("")
    
    def test_predict_before_training(self):
        """Test predict raises error when model is not trained."""
        # Setup mocks - model that raises error when predict is called
        model = Mock()
        model.predict.side_effect = AttributeError("Model not fitted")
        
        vectorizer = Mock()
        vectorizer.transform.return_value = Mock()
        
        preprocessor = Mock()
        preprocessor.preprocess_text.return_value = "test message"
        
        predictor = SpamPredictor(model, vectorizer, preprocessor)
        
        # Test prediction fails with untrained model
        with pytest.raises(AttributeError, match="Model not fitted"):
            predictor.predict("Test message")


class TestSpamPredictorPredictProba:
    """Test SpamPredictor predict_proba method."""
    
    def test_predict_proba_returns_confidence_scores(self):
        """Test predict_proba returns confidence scores for both classes."""
        # Setup mocks
        model = Mock()
        model.predict_proba.return_value = np.array([[0.8, 0.2]])
        
        vectorizer = Mock()
        vectorizer.transform.return_value = Mock()
        
        preprocessor = Mock()
        preprocessor.preprocess_text.return_value = "hello world"
        
        predictor = SpamPredictor(model, vectorizer, preprocessor)
        
        # Test prediction
        result = predictor.predict_proba("Hello World!")
        
        assert result == {'ham': 0.8, 'spam': 0.2}
        preprocessor.preprocess_text.assert_called_once_with("Hello World!")
        vectorizer.transform.assert_called_once_with(["hello world"])
    
    def test_predict_proba_spam_confident(self):
        """Test predict_proba with high spam confidence."""
        # Setup mocks
        model = Mock()
        model.predict_proba.return_value = np.array([[0.1, 0.9]])
        
        vectorizer = Mock()
        vectorizer.transform.return_value = Mock()
        
        preprocessor = Mock()
        preprocessor.preprocess_text.return_value = "free prize"
        
        predictor = SpamPredictor(model, vectorizer, preprocessor)
        
        # Test prediction
        result = predictor.predict_proba("FREE PRIZE!")
        
        assert result == {'ham': 0.1, 'spam': 0.9}
    
    def test_predict_proba_with_non_string_input(self):
        """Test predict_proba raises TypeError for non-string input."""
        model = Mock()
        vectorizer = Mock()
        preprocessor = Mock()
        
        predictor = SpamPredictor(model, vectorizer, preprocessor)
        
        with pytest.raises(TypeError, match="Message must be a string"):
            predictor.predict_proba(123)
    
    def test_predict_proba_probabilities_sum_to_one(self):
        """Test that returned probabilities sum to approximately 1.0."""
        # Setup mocks
        model = Mock()
        model.predict_proba.return_value = np.array([[0.65, 0.35]])
        
        vectorizer = Mock()
        vectorizer.transform.return_value = Mock()
        
        preprocessor = Mock()
        preprocessor.preprocess_text.return_value = "test message"
        
        predictor = SpamPredictor(model, vectorizer, preprocessor)
        
        # Test prediction
        result = predictor.predict_proba("Test message")
        
        total = result['ham'] + result['spam']
        assert abs(total - 1.0) < 1e-6  # Should sum to 1.0
