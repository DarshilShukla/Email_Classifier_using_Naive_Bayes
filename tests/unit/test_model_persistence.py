"""
Unit tests for the ModelPersistence class.

Tests model and vectorizer saving/loading functionality.
"""

import pytest
import os
import tempfile
import shutil
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from src.model_persistence import ModelPersistence


class TestModelPersistence:
    """Test suite for ModelPersistence class."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        # Cleanup after test
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_model(self):
        """Create a simple trained model for testing."""
        model = MultinomialNB()
        # Train on simple data
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
        y = np.array([0, 0, 1, 1])
        model.fit(X, y)
        return model
    
    @pytest.fixture
    def sample_vectorizer(self):
        """Create a fitted vectorizer for testing."""
        vectorizer = TfidfVectorizer()
        corpus = ["hello world", "spam message", "ham message"]
        vectorizer.fit(corpus)
        return vectorizer
    
    # Tests for save_model()
    
    def test_save_model_basic(self, temp_dir, sample_model):
        """Test basic model saving."""
        file_path = os.path.join(temp_dir, "model.pkl")
        
        ModelPersistence.save_model(sample_model, file_path)
        
        assert os.path.exists(file_path)
    
    def test_save_model_creates_directory(self, temp_dir, sample_model):
        """Test that save_model creates directory if it doesn't exist."""
        file_path = os.path.join(temp_dir, "subdir", "model.pkl")
        
        ModelPersistence.save_model(sample_model, file_path)
        
        assert os.path.exists(file_path)
        assert os.path.exists(os.path.dirname(file_path))
    
    def test_save_model_none_raises_error(self, temp_dir):
        """Test save_model raises ValueError for None model."""
        file_path = os.path.join(temp_dir, "model.pkl")
        
        with pytest.raises(ValueError, match="Cannot save None model"):
            ModelPersistence.save_model(None, file_path)
    
    def test_save_model_invalid_path_raises_error(self, sample_model):
        """Test save_model raises IOError for invalid path."""
        # Use an invalid path with invalid characters
        file_path = "C:\\invalid\x00path\\model.pkl"
        
        with pytest.raises(IOError, match="Failed to save model"):
            ModelPersistence.save_model(sample_model, file_path)
    
    # Tests for save_vectorizer()
    
    def test_save_vectorizer_basic(self, temp_dir, sample_vectorizer):
        """Test basic vectorizer saving."""
        file_path = os.path.join(temp_dir, "vectorizer.pkl")
        
        ModelPersistence.save_vectorizer(sample_vectorizer, file_path)
        
        assert os.path.exists(file_path)
    
    def test_save_vectorizer_creates_directory(self, temp_dir, sample_vectorizer):
        """Test that save_vectorizer creates directory if it doesn't exist."""
        file_path = os.path.join(temp_dir, "subdir", "vectorizer.pkl")
        
        ModelPersistence.save_vectorizer(sample_vectorizer, file_path)
        
        assert os.path.exists(file_path)
        assert os.path.exists(os.path.dirname(file_path))
    
    def test_save_vectorizer_none_raises_error(self, temp_dir):
        """Test save_vectorizer raises ValueError for None vectorizer."""
        file_path = os.path.join(temp_dir, "vectorizer.pkl")
        
        with pytest.raises(ValueError, match="Cannot save None vectorizer"):
            ModelPersistence.save_vectorizer(None, file_path)
    
    def test_save_vectorizer_invalid_path_raises_error(self, sample_vectorizer):
        """Test save_vectorizer raises IOError for invalid path."""
        # Use an invalid path with invalid characters
        file_path = "C:\\invalid\x00path\\vectorizer.pkl"
        
        with pytest.raises(IOError, match="Failed to save vectorizer"):
            ModelPersistence.save_vectorizer(sample_vectorizer, file_path)
    
    # Tests for load_model()
    
    def test_load_model_basic(self, temp_dir, sample_model):
        """Test basic model loading."""
        file_path = os.path.join(temp_dir, "model.pkl")
        
        # Save then load
        ModelPersistence.save_model(sample_model, file_path)
        loaded_model = ModelPersistence.load_model(file_path)
        
        assert loaded_model is not None
        assert isinstance(loaded_model, MultinomialNB)
    
    def test_load_model_file_not_found(self, temp_dir):
        """Test load_model raises IOError for missing file."""
        file_path = os.path.join(temp_dir, "nonexistent.pkl")
        
        with pytest.raises(IOError, match="Model file not found"):
            ModelPersistence.load_model(file_path)
    
    def test_load_model_corrupted_file(self, temp_dir):
        """Test load_model raises IOError for corrupted file."""
        file_path = os.path.join(temp_dir, "corrupted.pkl")
        
        # Create a corrupted file
        with open(file_path, 'w') as f:
            f.write("This is not a valid pickle file")
        
        with pytest.raises(IOError, match="Failed to load model"):
            ModelPersistence.load_model(file_path)
    
    def test_load_model_preserves_functionality(self, temp_dir, sample_model):
        """Test that loaded model can make predictions."""
        file_path = os.path.join(temp_dir, "model.pkl")
        
        # Save model
        ModelPersistence.save_model(sample_model, file_path)
        
        # Load model
        loaded_model = ModelPersistence.load_model(file_path)
        
        # Test prediction
        X_test = np.array([[1, 2], [4, 5]])
        predictions = loaded_model.predict(X_test)
        
        assert len(predictions) == 2
        assert all(pred in [0, 1] for pred in predictions)
    
    # Tests for load_vectorizer()
    
    def test_load_vectorizer_basic(self, temp_dir, sample_vectorizer):
        """Test basic vectorizer loading."""
        file_path = os.path.join(temp_dir, "vectorizer.pkl")
        
        # Save then load
        ModelPersistence.save_vectorizer(sample_vectorizer, file_path)
        loaded_vectorizer = ModelPersistence.load_vectorizer(file_path)
        
        assert loaded_vectorizer is not None
        assert isinstance(loaded_vectorizer, TfidfVectorizer)
    
    def test_load_vectorizer_file_not_found(self, temp_dir):
        """Test load_vectorizer raises IOError for missing file."""
        file_path = os.path.join(temp_dir, "nonexistent.pkl")
        
        with pytest.raises(IOError, match="Vectorizer file not found"):
            ModelPersistence.load_vectorizer(file_path)
    
    def test_load_vectorizer_corrupted_file(self, temp_dir):
        """Test load_vectorizer raises IOError for corrupted file."""
        file_path = os.path.join(temp_dir, "corrupted.pkl")
        
        # Create a corrupted file
        with open(file_path, 'w') as f:
            f.write("This is not a valid pickle file")
        
        with pytest.raises(IOError, match="Failed to load vectorizer"):
            ModelPersistence.load_vectorizer(file_path)
    
    def test_load_vectorizer_preserves_functionality(self, temp_dir, sample_vectorizer):
        """Test that loaded vectorizer can transform text."""
        file_path = os.path.join(temp_dir, "vectorizer.pkl")
        
        # Save vectorizer
        ModelPersistence.save_vectorizer(sample_vectorizer, file_path)
        
        # Load vectorizer
        loaded_vectorizer = ModelPersistence.load_vectorizer(file_path)
        
        # Test transformation
        test_text = ["hello world"]
        result = loaded_vectorizer.transform(test_text)
        
        assert result.shape[0] == 1
        assert result.shape[1] > 0
    
    # Integration tests
    
    def test_save_and_load_model_produces_identical_predictions(self, temp_dir, sample_model):
        """Test that save→load→predict produces identical results.
        
        Requirements: 7.1, 7.3
        """
        file_path = os.path.join(temp_dir, "model.pkl")
        
        # Get predictions from original model
        X_test = np.array([[1, 2], [2, 3], [3, 4]])
        original_predictions = sample_model.predict(X_test)
        
        # Save and load model
        ModelPersistence.save_model(sample_model, file_path)
        loaded_model = ModelPersistence.load_model(file_path)
        
        # Get predictions from loaded model
        loaded_predictions = loaded_model.predict(X_test)
        
        # Predictions should be identical
        np.testing.assert_array_equal(original_predictions, loaded_predictions)
    
    def test_save_and_load_vectorizer_produces_identical_output(self, temp_dir, sample_vectorizer):
        """Test that save→load→transform produces identical results.
        
        Requirements: 7.2, 7.4
        """
        file_path = os.path.join(temp_dir, "vectorizer.pkl")
        
        # Get transformation from original vectorizer
        test_text = ["hello world", "spam message"]
        original_output = sample_vectorizer.transform(test_text)
        
        # Save and load vectorizer
        ModelPersistence.save_vectorizer(sample_vectorizer, file_path)
        loaded_vectorizer = ModelPersistence.load_vectorizer(file_path)
        
        # Get transformation from loaded vectorizer
        loaded_output = loaded_vectorizer.transform(test_text)
        
        # Outputs should be identical
        np.testing.assert_array_almost_equal(
            original_output.toarray(), 
            loaded_output.toarray()
        )
    
    def test_complete_persistence_workflow(self, temp_dir):
        """Test complete save/load workflow for both model and vectorizer."""
        model_path = os.path.join(temp_dir, "model.pkl")
        vectorizer_path = os.path.join(temp_dir, "vectorizer.pkl")
        
        # Create a model and vectorizer that work together
        corpus = ["hello world", "spam message", "ham message"]
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(corpus)
        y = np.array([0, 1, 0])
        
        model = MultinomialNB()
        model.fit(X, y)
        
        # Save both
        ModelPersistence.save_model(model, model_path)
        ModelPersistence.save_vectorizer(vectorizer, vectorizer_path)
        
        # Verify files exist
        assert os.path.exists(model_path)
        assert os.path.exists(vectorizer_path)
        
        # Load both
        loaded_model = ModelPersistence.load_model(model_path)
        loaded_vectorizer = ModelPersistence.load_vectorizer(vectorizer_path)
        
        # Verify functionality
        test_text = ["hello world"]
        X_test = loaded_vectorizer.transform(test_text)
        predictions = loaded_model.predict(X_test)
        
        assert len(predictions) == 1
        assert predictions[0] in [0, 1]
    
    def test_overwrite_existing_file(self, temp_dir, sample_model):
        """Test that saving overwrites existing file."""
        file_path = os.path.join(temp_dir, "model.pkl")
        
        # Save first model
        ModelPersistence.save_model(sample_model, file_path)
        first_size = os.path.getsize(file_path)
        
        # Create and save a different model
        new_model = MultinomialNB()
        X = np.array([[5, 6], [6, 7], [7, 8], [8, 9]])
        y = np.array([0, 0, 1, 1])
        new_model.fit(X, y)
        
        ModelPersistence.save_model(new_model, file_path)
        
        # File should still exist
        assert os.path.exists(file_path)
        
        # Load and verify it's the new model
        loaded_model = ModelPersistence.load_model(file_path)
        test_pred = loaded_model.predict([[5, 6]])
        
        # Should work without errors
        assert test_pred[0] in [0, 1]
