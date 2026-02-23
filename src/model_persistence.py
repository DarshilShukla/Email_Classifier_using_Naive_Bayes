"""
Model persistence module for SMS spam classification.

This module provides the ModelPersistence class for saving and loading
trained models and vectorizers to/from disk.
"""

import os
from typing import Any
import joblib


class ModelPersistence:
    """
    Handles saving and loading of trained models and vectorizers.
    
    The ModelPersistence class provides methods to serialize and deserialize
    machine learning models and TF-IDF vectorizers using joblib.
    """
    
    @staticmethod
    def save_model(model: Any, file_path: str) -> None:
        """
        Save a trained model to disk.
        
        Args:
            model: Trained classifier instance to save
            file_path: Path where the model should be saved
            
        Raises:
            IOError: If unable to write to disk
            ValueError: If model is None
        """
        if model is None:
            raise ValueError("Cannot save None model")
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            joblib.dump(model, file_path)
        except Exception as e:
            raise IOError(f"Failed to save model to {file_path}: {str(e)}")
    
    @staticmethod
    def save_vectorizer(vectorizer: Any, file_path: str) -> None:
        """
        Save a fitted vectorizer to disk.
        
        Args:
            vectorizer: Fitted TF-IDF vectorizer instance to save
            file_path: Path where the vectorizer should be saved
            
        Raises:
            IOError: If unable to write to disk
            ValueError: If vectorizer is None
        """
        if vectorizer is None:
            raise ValueError("Cannot save None vectorizer")
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            joblib.dump(vectorizer, file_path)
        except Exception as e:
            raise IOError(f"Failed to save vectorizer to {file_path}: {str(e)}")
    
    @staticmethod
    def load_model(file_path: str) -> Any:
        """
        Load a trained model from disk.
        
        Args:
            file_path: Path to the saved model file
            
        Returns:
            Loaded classifier instance
            
        Raises:
            IOError: If file doesn't exist or is corrupted
        """
        if not os.path.exists(file_path):
            raise IOError(f"Model file not found: {file_path}")
        
        try:
            model = joblib.load(file_path)
            return model
        except Exception as e:
            raise IOError(f"Failed to load model from {file_path}: {str(e)}")
    
    @staticmethod
    def load_vectorizer(file_path: str) -> Any:
        """
        Load a fitted vectorizer from disk.
        
        Args:
            file_path: Path to the saved vectorizer file
            
        Returns:
            Loaded TF-IDF vectorizer instance
            
        Raises:
            IOError: If file doesn't exist or is corrupted
        """
        if not os.path.exists(file_path):
            raise IOError(f"Vectorizer file not found: {file_path}")
        
        try:
            vectorizer = joblib.load(file_path)
            return vectorizer
        except Exception as e:
            raise IOError(f"Failed to load vectorizer from {file_path}: {str(e)}")
