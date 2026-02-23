"""
Unit tests for the ModelTrainer class.

Tests model training, data splitting, cross-validation, and model retrieval.
"""

import pytest
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.naive_bayes import MultinomialNB
from src.model_trainer import ModelTrainer


class TestModelTrainer:
    """Test suite for ModelTrainer class."""
    
    @pytest.fixture
    def trainer(self):
        """Create a ModelTrainer instance for testing."""
        return ModelTrainer()
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        # Create simple sparse feature matrix
        X = csr_matrix(np.random.rand(200, 10))
        y = np.random.randint(0, 2, 200)
        return X, y
    
    # Tests for __init__()
    
    def test_init_default(self):
        """Test default initialization."""
        trainer = ModelTrainer()
        assert trainer.model_type == "multinomial_nb"
        assert trainer.model is None
    
    def test_init_custom_model_type(self):
        """Test initialization with custom model type."""
        trainer = ModelTrainer(model_type="custom")
        assert trainer.model_type == "custom"
    
    # Tests for split_data()
    
    def test_split_data_basic(self, trainer, sample_data):
        """Test basic data splitting."""
        X, y = sample_data
        X_train, X_test, y_train, y_test = trainer.split_data(X, y)
        
        # Check shapes (use .shape[0] for sparse matrices)
        assert X_train.shape[0] + X_test.shape[0] == X.shape[0]
        assert len(y_train) + len(y_test) == len(y)
        assert X_train.shape[0] == len(y_train)
        assert X_test.shape[0] == len(y_test)
    
    def test_split_data_default_ratio(self, trainer, sample_data):
        """Test default 80/20 split ratio."""
        X, y = sample_data
        X_train, X_test, y_train, y_test = trainer.split_data(X, y)
        
        # Should be approximately 80/20 split
        train_ratio = X_train.shape[0] / X.shape[0]
        assert 0.75 < train_ratio < 0.85
    
    def test_split_data_custom_ratio(self, trainer, sample_data):
        """Test custom split ratio."""
        X, y = sample_data
        X_train, X_test, y_train, y_test = trainer.split_data(X, y, test_size=0.3)
        
        # Should be approximately 70/30 split
        train_ratio = X_train.shape[0] / X.shape[0]
        assert 0.65 < train_ratio < 0.75
    
    def test_split_data_reproducibility(self, trainer, sample_data):
        """Test that same random_state produces same split."""
        X, y = sample_data
        
        X_train1, X_test1, y_train1, y_test1 = trainer.split_data(X, y, random_state=42)
        X_train2, X_test2, y_train2, y_test2 = trainer.split_data(X, y, random_state=42)
        
        # Should produce identical splits
        np.testing.assert_array_equal(y_train1, y_train2)
        np.testing.assert_array_equal(y_test1, y_test2)
    
    def test_split_data_invalid_test_size_too_small(self, trainer, sample_data):
        """Test split_data raises ValueError for test_size <= 0."""
        X, y = sample_data
        
        with pytest.raises(ValueError, match="test_size must be in range"):
            trainer.split_data(X, y, test_size=0)
    
    def test_split_data_invalid_test_size_too_large(self, trainer, sample_data):
        """Test split_data raises ValueError for test_size >= 1."""
        X, y = sample_data
        
        with pytest.raises(ValueError, match="test_size must be in range"):
            trainer.split_data(X, y, test_size=1.0)
    
    def test_split_data_invalid_test_size_negative(self, trainer, sample_data):
        """Test split_data raises ValueError for negative test_size."""
        X, y = sample_data
        
        with pytest.raises(ValueError, match="test_size must be in range"):
            trainer.split_data(X, y, test_size=-0.2)
    
    # Tests for train()
    
    def test_train_basic(self, trainer, sample_data):
        """Test basic model training."""
        X, y = sample_data
        X_train, X_test, y_train, y_test = trainer.split_data(X, y)
        
        trainer.train(X_train, y_train)
        
        assert trainer.model is not None
        assert isinstance(trainer.model, MultinomialNB)
    
    def test_train_model_can_predict(self, trainer, sample_data):
        """Test that trained model can make predictions."""
        X, y = sample_data
        X_train, X_test, y_train, y_test = trainer.split_data(X, y)
        
        trainer.train(X_train, y_train)
        predictions = trainer.model.predict(X_test)
        
        assert len(predictions) == len(y_test)
        assert all(pred in [0, 1] for pred in predictions)
    
    def test_train_insufficient_data(self, trainer):
        """Test train raises ValueError for insufficient data."""
        # Create dataset with < 100 samples
        X = csr_matrix(np.random.rand(50, 10))
        y = np.random.randint(0, 2, 50)
        
        with pytest.raises(ValueError, match="Insufficient training data"):
            trainer.train(X, y)
    
    def test_train_exactly_100_samples(self, trainer):
        """Test train works with exactly 100 samples."""
        X = csr_matrix(np.random.rand(100, 10))
        y = np.random.randint(0, 2, 100)
        
        # Should not raise error
        trainer.train(X, y)
        assert trainer.model is not None
    
    def test_train_model_type_is_multinomial_nb(self, trainer, sample_data):
        """Test that trained model is MultinomialNB."""
        X, y = sample_data
        X_train, X_test, y_train, y_test = trainer.split_data(X, y)
        
        trainer.train(X_train, y_train)
        
        assert isinstance(trainer.model, MultinomialNB)
    
    # Tests for cross_validate()
    
    def test_cross_validate_basic(self, trainer, sample_data):
        """Test basic cross-validation."""
        X, y = sample_data
        
        result = trainer.cross_validate(X, y)
        
        assert 'scores' in result
        assert 'mean' in result
        assert 'std' in result
    
    def test_cross_validate_default_folds(self, trainer, sample_data):
        """Test cross-validation with default 5 folds."""
        X, y = sample_data
        
        result = trainer.cross_validate(X, y)
        
        # Should have 5 scores (one per fold)
        assert len(result['scores']) == 5
    
    def test_cross_validate_custom_folds(self, trainer, sample_data):
        """Test cross-validation with custom number of folds."""
        X, y = sample_data
        
        result = trainer.cross_validate(X, y, cv=3)
        
        # Should have 3 scores
        assert len(result['scores']) == 3
    
    def test_cross_validate_scores_in_valid_range(self, trainer, sample_data):
        """Test that cross-validation scores are between 0 and 1."""
        X, y = sample_data
        
        result = trainer.cross_validate(X, y)
        
        # All scores should be between 0 and 1
        assert all(0 <= score <= 1 for score in result['scores'])
    
    def test_cross_validate_mean_calculation(self, trainer, sample_data):
        """Test that mean is correctly calculated."""
        X, y = sample_data
        
        result = trainer.cross_validate(X, y)
        
        # Mean should equal numpy mean of scores
        expected_mean = np.mean(result['scores'])
        assert abs(result['mean'] - expected_mean) < 1e-10
    
    def test_cross_validate_std_calculation(self, trainer, sample_data):
        """Test that standard deviation is correctly calculated."""
        X, y = sample_data
        
        result = trainer.cross_validate(X, y)
        
        # Std should equal numpy std of scores
        expected_std = np.std(result['scores'])
        assert abs(result['std'] - expected_std) < 1e-10
    
    def test_cross_validate_without_training(self, trainer, sample_data):
        """Test cross-validation works without prior training."""
        X, y = sample_data
        
        # Should work even if model is None
        result = trainer.cross_validate(X, y)
        
        assert 'scores' in result
        assert len(result['scores']) == 5
    
    def test_cross_validate_after_training(self, trainer, sample_data):
        """Test cross-validation after training."""
        X, y = sample_data
        X_train, X_test, y_train, y_test = trainer.split_data(X, y)
        
        trainer.train(X_train, y_train)
        result = trainer.cross_validate(X, y)
        
        assert 'scores' in result
        assert len(result['scores']) == 5
    
    # Tests for get_model()
    
    def test_get_model_after_training(self, trainer, sample_data):
        """Test get_model returns trained model."""
        X, y = sample_data
        X_train, X_test, y_train, y_test = trainer.split_data(X, y)
        
        trainer.train(X_train, y_train)
        model = trainer.get_model()
        
        assert model is not None
        assert isinstance(model, MultinomialNB)
    
    def test_get_model_before_training(self, trainer):
        """Test get_model raises ValueError before training."""
        with pytest.raises(ValueError, match="Model has not been trained yet"):
            trainer.get_model()
    
    def test_get_model_returns_same_instance(self, trainer, sample_data):
        """Test get_model returns the same model instance."""
        X, y = sample_data
        X_train, X_test, y_train, y_test = trainer.split_data(X, y)
        
        trainer.train(X_train, y_train)
        model1 = trainer.get_model()
        model2 = trainer.get_model()
        
        assert model1 is model2
    
    def test_get_model_can_predict(self, trainer, sample_data):
        """Test that returned model can make predictions."""
        X, y = sample_data
        X_train, X_test, y_train, y_test = trainer.split_data(X, y)
        
        trainer.train(X_train, y_train)
        model = trainer.get_model()
        
        predictions = model.predict(X_test)
        assert len(predictions) == len(y_test)
    
    # Integration tests
    
    def test_complete_training_workflow(self, trainer, sample_data):
        """Test complete training workflow."""
        X, y = sample_data
        
        # Split data
        X_train, X_test, y_train, y_test = trainer.split_data(X, y)
        
        # Train model
        trainer.train(X_train, y_train)
        
        # Cross-validate
        cv_result = trainer.cross_validate(X, y)
        
        # Get model and predict
        model = trainer.get_model()
        predictions = model.predict(X_test)
        
        # Verify everything worked
        assert len(predictions) == len(y_test)
        assert 'mean' in cv_result
        assert cv_result['mean'] > 0
    
    def test_training_with_dense_array(self, trainer):
        """Test training with dense numpy array."""
        X = np.random.rand(200, 10)
        y = np.random.randint(0, 2, 200)
        
        X_train, X_test, y_train, y_test = trainer.split_data(X, y)
        trainer.train(X_train, y_train)
        
        assert trainer.model is not None
    
    def test_training_with_sparse_matrix(self, trainer, sample_data):
        """Test training with sparse matrix."""
        X, y = sample_data
        
        X_train, X_test, y_train, y_test = trainer.split_data(X, y)
        trainer.train(X_train, y_train)
        
        assert trainer.model is not None
    
    # Tests for accuracy requirements
    
    @pytest.mark.skipif(
        not pytest.importorskip("os").path.exists("data/spam.csv"),
        reason="UCI SMS spam dataset not available"
    )
    def test_model_achieves_97_percent_accuracy_on_uci_dataset(self, trainer):
        """Test model achieves ≥97% accuracy on UCI dataset.
        
        Requirements: 4.4
        """
        import os
        from src.data_cleaner import DataCleaner
        from src.text_preprocessor import TextPreprocessor
        
        # Load and clean the UCI dataset
        cleaner = DataCleaner()
        df = cleaner.clean_pipeline("data/spam.csv")
        
        # Preprocess text
        preprocessor = TextPreprocessor()
        preprocessed_messages = [preprocessor.preprocess_text(msg) for msg in df['message']]
        
        # Vectorize
        X = preprocessor.fit_transform(preprocessed_messages)
        y = df['label'].values
        
        # Split data
        X_train, X_test, y_train, y_test = trainer.split_data(X, y)
        
        # Train model
        trainer.train(X_train, y_train)
        
        # Get predictions
        model = trainer.get_model()
        predictions = model.predict(X_test)
        
        # Calculate accuracy
        accuracy = (predictions == y_test).sum() / len(y_test)
        
        # Assert accuracy is at least 97%
        assert accuracy >= 0.97, f"Model accuracy {accuracy:.2%} is below required 97%"
    
    def test_model_high_accuracy_on_synthetic_data(self, trainer):
        """Test model achieves high accuracy on synthetic separable data.
        
        This test uses synthetic data to verify the model can achieve high accuracy
        when the data is well-separated. This serves as a proxy test when the
        UCI dataset is not available.
        
        Requirements: 4.4
        """
        # Create synthetic data with clear separation
        np.random.seed(42)
        
        # Class 0: Low values in first 5 features, high in last 5
        X_class0 = np.random.rand(500, 10)
        X_class0[:, :5] *= 0.3  # Low values
        X_class0[:, 5:] *= 2.0  # High values
        y_class0 = np.zeros(500, dtype=int)
        
        # Class 1: High values in first 5 features, low in last 5
        X_class1 = np.random.rand(500, 10)
        X_class1[:, :5] *= 2.0  # High values
        X_class1[:, 5:] *= 0.3  # Low values
        y_class1 = np.ones(500, dtype=int)
        
        # Combine and convert to sparse matrix
        X = csr_matrix(np.vstack([X_class0, X_class1]))
        y = np.hstack([y_class0, y_class1])
        
        # Split data
        X_train, X_test, y_train, y_test = trainer.split_data(X, y)
        
        # Train model
        trainer.train(X_train, y_train)
        
        # Get predictions
        model = trainer.get_model()
        predictions = model.predict(X_test)
        
        # Calculate accuracy
        accuracy = (predictions == y_test).sum() / len(y_test)
        
        # With well-separated synthetic data, should achieve >90% accuracy
        assert accuracy > 0.90, f"Model accuracy {accuracy:.2%} is unexpectedly low on synthetic data"
