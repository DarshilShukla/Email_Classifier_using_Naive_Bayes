"""
Property-based tests for SpamPredictor class.

These tests verify correctness properties related to real-time prediction.
"""

import pytest
from hypothesis import given, strategies as st
from unittest.mock import Mock
import numpy as np
from src.spam_predictor import SpamPredictor
from src.text_preprocessor import TextPreprocessor


class TestPredictionProperties:
    """Property-based tests for prediction functionality."""
    
    @given(st.text(min_size=1))
    def test_property_19_prediction_output_validity(self, message):
        """
        Property 19: Prediction output validity
        
        For any valid text message, the prediction output should be 
        exactly one of two values: "ham" or "spam".
        
        **Validates: Requirements 6.3**
        """
        # Setup mocks
        model = Mock()
        # Randomly return 0 or 1
        model.predict.return_value = np.array([hash(message) % 2])
        
        vectorizer = Mock()
        vectorizer.transform.return_value = Mock()
        
        preprocessor = Mock()
        preprocessor.preprocess_text.return_value = message.lower()
        
        predictor = SpamPredictor(model, vectorizer, preprocessor)
        
        # Test prediction
        result = predictor.predict(message)
        
        # Verify result is exactly "ham" or "spam"
        assert result in ["ham", "spam"]
        assert isinstance(result, str)
    
    @given(st.text(min_size=1))
    def test_property_18_preprocessing_consistency(self, message):
        """
        Property 18: Preprocessing consistency
        
        For any message, the preprocessing applied during prediction should 
        apply the same transformations (in the same order) as used during training.
        
        This test verifies that the predictor calls the preprocessor's 
        preprocess_text method, which applies the full pipeline.
        
        **Validates: Requirements 6.1**
        """
        # Setup mocks
        model = Mock()
        model.predict.return_value = np.array([0])
        
        vectorizer = Mock()
        vectorizer.transform.return_value = Mock()
        
        # Use real preprocessor to verify full pipeline is applied
        preprocessor = TextPreprocessor()
        
        predictor = SpamPredictor(model, vectorizer, preprocessor)
        
        # Test prediction
        predictor.predict(message)
        
        # Verify vectorizer received preprocessed text
        call_args = vectorizer.transform.call_args[0][0]
        assert isinstance(call_args, list)
        assert len(call_args) == 1
        
        # Verify the preprocessed text has expected characteristics:
        # - Should be lowercase (or empty if all stopwords/special chars)
        # - Should not contain special characters
        preprocessed = call_args[0]
        if preprocessed:  # If not empty after preprocessing
            assert preprocessed == preprocessed.lower()
            # Should only contain alphanumeric and spaces
            assert all(c.isalnum() or c.isspace() for c in preprocessed)
    
    @given(
        st.floats(min_value=0.0, max_value=1.0),
        st.text(min_size=1)
    )
    def test_predict_proba_returns_valid_probabilities(self, ham_prob, message):
        """
        Test that predict_proba returns valid probability values.
        
        Probabilities should be between 0 and 1, and sum to approximately 1.
        
        **Validates: Requirements 6.2**
        """
        spam_prob = 1.0 - ham_prob
        
        # Setup mocks
        model = Mock()
        model.predict_proba.return_value = np.array([[ham_prob, spam_prob]])
        
        vectorizer = Mock()
        vectorizer.transform.return_value = Mock()
        
        preprocessor = Mock()
        preprocessor.preprocess_text.return_value = message.lower()
        
        predictor = SpamPredictor(model, vectorizer, preprocessor)
        
        # Test prediction
        result = predictor.predict_proba(message)
        
        # Verify structure
        assert isinstance(result, dict)
        assert 'ham' in result
        assert 'spam' in result
        
        # Verify probability values
        assert 0.0 <= result['ham'] <= 1.0
        assert 0.0 <= result['spam'] <= 1.0
        
        # Verify probabilities sum to approximately 1.0
        total = result['ham'] + result['spam']
        assert abs(total - 1.0) < 1e-6
    
    @given(st.text(min_size=1))
    def test_predict_and_predict_proba_consistency(self, message):
        """
        Test that predict() and predict_proba() are consistent.
        
        The class with higher probability should match the predicted label.
        """
        # Setup mocks with consistent predictions
        prediction_value = hash(message) % 2
        ham_prob = 0.7 if prediction_value == 0 else 0.3
        spam_prob = 1.0 - ham_prob
        
        model = Mock()
        model.predict.return_value = np.array([prediction_value])
        model.predict_proba.return_value = np.array([[ham_prob, spam_prob]])
        
        vectorizer = Mock()
        vectorizer.transform.return_value = Mock()
        
        preprocessor = Mock()
        preprocessor.preprocess_text.return_value = message.lower()
        
        predictor = SpamPredictor(model, vectorizer, preprocessor)
        
        # Get both predictions
        label = predictor.predict(message)
        probas = predictor.predict_proba(message)
        
        # Verify consistency
        if label == "ham":
            assert probas['ham'] >= probas['spam']
        else:
            assert probas['spam'] >= probas['ham']
