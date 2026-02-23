"""
Model training module for SMS spam classification.

This module provides the ModelTrainer class for training and validating
the spam classification model using Multinomial Naive Bayes.
"""

from typing import Tuple, Dict
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import MultinomialNB


class ModelTrainer:
    """
    Handles model training and validation for SMS spam classification.
    
    The ModelTrainer class provides methods to split data, train a
    Multinomial Naive Bayes classifier, perform cross-validation,
    and retrieve the trained model.
    """
    
    def __init__(self, model_type: str = "multinomial_nb"):
        """
        Initialize the ModelTrainer.
        
        Args:
            model_type: Type of model to use (default: "multinomial_nb")
        """
        self.model_type = model_type
        self.model = None
    
    def split_data(
        self, 
        X, 
        y, 
        test_size: float = 0.2, 
        random_state: int = 42
    ) -> Tuple:
        """
        Split data into training and testing sets.
        
        Args:
            X: Feature matrix (can be sparse or dense)
            y: Target labels array
            test_size: Proportion of data to use for testing (default: 0.2)
            random_state: Random seed for reproducibility (default: 42)
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
            
        Raises:
            ValueError: If test_size is not in range (0, 1)
        """
        if not 0 < test_size < 1:
            raise ValueError(f"test_size must be in range (0, 1), got {test_size}")
        
        return train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=random_state
        )
    
    def train(self, X_train, y_train) -> None:
        """
        Train the Multinomial Naive Bayes classifier.
        
        Args:
            X_train: Training feature matrix
            y_train: Training labels
            
        Raises:
            ValueError: If training data is insufficient (<100 samples)
        """
        if len(y_train) < 100:
            raise ValueError(
                f"Insufficient training data: {len(y_train)} samples. "
                "Minimum 100 samples required."
            )
        
        # Initialize and train the model
        self.model = MultinomialNB()
        self.model.fit(X_train, y_train)
    
    def cross_validate(self, X, y, cv: int = 5) -> Dict[str, float]:
        """
        Perform k-fold cross-validation on the dataset.
        
        Args:
            X: Feature matrix
            y: Target labels
            cv: Number of folds (default: 5)
            
        Returns:
            Dictionary containing cross-validation scores:
                - 'scores': Array of scores for each fold
                - 'mean': Mean score across all folds
                - 'std': Standard deviation of scores
                
        Raises:
            ValueError: If model has not been initialized
        """
        if self.model is None:
            # Initialize model for cross-validation
            model = MultinomialNB()
        else:
            model = self.model
        
        # Perform cross-validation
        scores = cross_val_score(model, X, y, cv=cv)
        
        return {
            'scores': scores,
            'mean': scores.mean(),
            'std': scores.std()
        }
    
    def get_model(self):
        """
        Return the trained model.
        
        Returns:
            Trained classifier instance
            
        Raises:
            ValueError: If model has not been trained yet
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call train() first.")
        
        return self.model
