"""
Property-based tests for model persistence module.

These tests verify universal properties that should hold across all valid inputs
using Hypothesis for property-based testing.
"""

import pytest
import os
import tempfile
import shutil
import numpy as np
from hypothesis import given, strategies as st, settings
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from src.model_persistence import ModelPersistence


class TestPersistenceProperties:
    """Property-based tests for ModelPersistence class."""
    
    @given(
        st.integers(min_value=10, max_value=100),
        st.integers(min_value=2, max_value=20),
        st.integers(min_value=1, max_value=50)
    )
    @settings(max_examples=100)
    def test_property_model_persistence_roundtrip(self, n_samples, n_features, random_seed):
        """
        Property 20: Model persistence round-trip
        
        For any trained model and vectorizer, saving to disk then loading from disk
        then making predictions should produce identical results to the original
        model's predictions on the same input.
        
        **Validates: Requirements 7.5**
        """
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Generate random training data
            np.random.seed(random_seed)
            X_train = np.random.rand(n_samples, n_features)
            # Ensure non-negative values for MultinomialNB
            X_train = np.abs(X_train)
            y_train = np.random.randint(0, 2, n_samples)
            
            # Train a model
            model = MultinomialNB()
            model.fit(X_train, y_train)
            
            # Generate random test data
            n_test = max(1, n_samples // 5)
            X_test = np.abs(np.random.rand(n_test, n_features))
            
            # Get predictions from original model
            original_predictions = model.predict(X_test)
            original_probabilities = model.predict_proba(X_test)
            
            # Save model to disk
            model_path = os.path.join(temp_dir, f"model_{random_seed}.pkl")
            ModelPersistence.save_model(model, model_path)
            
            # Load model from disk
            loaded_model = ModelPersistence.load_model(model_path)
            
            # Get predictions from loaded model
            loaded_predictions = loaded_model.predict(X_test)
            loaded_probabilities = loaded_model.predict_proba(X_test)
            
            # Property: Predictions should be identical
            np.testing.assert_array_equal(
                original_predictions, 
                loaded_predictions,
                err_msg="Loaded model predictions differ from original model predictions"
            )
            
            # Property: Probabilities should be identical
            np.testing.assert_array_almost_equal(
                original_probabilities,
                loaded_probabilities,
                decimal=10,
                err_msg="Loaded model probabilities differ from original model probabilities"
            )
        finally:
            # Cleanup
            shutil.rmtree(temp_dir)
    
    @given(
        st.lists(st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=('L', 'N'))), min_size=5, max_size=50),
        st.integers(min_value=1, max_value=50)
    )
    @settings(max_examples=100)
    def test_property_vectorizer_persistence_roundtrip(self, corpus, random_seed):
        """
        Property 20: Vectorizer persistence round-trip
        
        For any fitted vectorizer, saving to disk then loading from disk then
        transforming text should produce identical results to the original
        vectorizer's transformation.
        
        **Validates: Requirements 7.5**
        """
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Ensure corpus has unique elements for fitting
            corpus = list(set(corpus))
            if len(corpus) < 2:
                # Skip if corpus is too small after deduplication
                return
            
            # Fit a vectorizer
            vectorizer = TfidfVectorizer()
            try:
                vectorizer.fit(corpus)
            except ValueError:
                # Skip if corpus is invalid for TF-IDF
                return
            
            # Create test data (subset of corpus)
            test_texts = corpus[:min(5, len(corpus))]
            
            # Get transformation from original vectorizer
            original_output = vectorizer.transform(test_texts)
            
            # Save vectorizer to disk
            vectorizer_path = os.path.join(temp_dir, f"vectorizer_{random_seed}.pkl")
            ModelPersistence.save_vectorizer(vectorizer, vectorizer_path)
            
            # Load vectorizer from disk
            loaded_vectorizer = ModelPersistence.load_vectorizer(vectorizer_path)
            
            # Get transformation from loaded vectorizer
            loaded_output = loaded_vectorizer.transform(test_texts)
            
            # Property: Transformations should be identical
            np.testing.assert_array_almost_equal(
                original_output.toarray(),
                loaded_output.toarray(),
                decimal=10,
                err_msg="Loaded vectorizer output differs from original vectorizer output"
            )
        finally:
            # Cleanup
            shutil.rmtree(temp_dir)
    
    @given(
        st.integers(min_value=10, max_value=100),
        st.lists(st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=('L', 'N'))), min_size=5, max_size=50),
        st.integers(min_value=1, max_value=50)
    )
    @settings(max_examples=100)
    def test_property_complete_pipeline_persistence_roundtrip(self, n_samples, corpus, random_seed):
        """
        Property 20: Complete pipeline persistence round-trip
        
        For any trained model and vectorizer working together, saving both to disk
        then loading both then making predictions should produce identical results
        to the original pipeline's predictions.
        
        **Validates: Requirements 7.5**
        """
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Ensure corpus has unique elements
            corpus = list(set(corpus))
            if len(corpus) < n_samples:
                # Extend corpus if needed
                corpus = corpus * ((n_samples // len(corpus)) + 1)
            corpus = corpus[:n_samples]
            
            # Fit vectorizer and transform corpus
            vectorizer = TfidfVectorizer()
            try:
                X_train = vectorizer.fit_transform(corpus)
            except ValueError:
                # Skip if corpus is invalid
                return
            
            # Generate random labels
            np.random.seed(random_seed)
            y_train = np.random.randint(0, 2, n_samples)
            
            # Train model
            model = MultinomialNB()
            model.fit(X_train, y_train)
            
            # Create test data
            test_texts = corpus[:min(5, len(corpus))]
            X_test = vectorizer.transform(test_texts)
            
            # Get predictions from original pipeline
            original_predictions = model.predict(X_test)
            original_probabilities = model.predict_proba(X_test)
            
            # Save both model and vectorizer
            model_path = os.path.join(temp_dir, f"model_{random_seed}.pkl")
            vectorizer_path = os.path.join(temp_dir, f"vectorizer_{random_seed}.pkl")
            
            ModelPersistence.save_model(model, model_path)
            ModelPersistence.save_vectorizer(vectorizer, vectorizer_path)
            
            # Load both model and vectorizer
            loaded_model = ModelPersistence.load_model(model_path)
            loaded_vectorizer = ModelPersistence.load_vectorizer(vectorizer_path)
            
            # Transform test data with loaded vectorizer and predict with loaded model
            X_test_loaded = loaded_vectorizer.transform(test_texts)
            loaded_predictions = loaded_model.predict(X_test_loaded)
            loaded_probabilities = loaded_model.predict_proba(X_test_loaded)
            
            # Property: Complete pipeline predictions should be identical
            np.testing.assert_array_equal(
                original_predictions,
                loaded_predictions,
                err_msg="Loaded pipeline predictions differ from original pipeline predictions"
            )
            
            # Property: Complete pipeline probabilities should be identical
            np.testing.assert_array_almost_equal(
                original_probabilities,
                loaded_probabilities,
                decimal=10,
                err_msg="Loaded pipeline probabilities differ from original pipeline probabilities"
            )
        finally:
            # Cleanup
            shutil.rmtree(temp_dir)
