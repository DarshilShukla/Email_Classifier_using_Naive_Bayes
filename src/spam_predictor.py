"""
Spam prediction module for SMS spam classification.

This module provides the SpamPredictor class for classifying
new SMS messages in real-time using a trained model.
"""

from typing import Dict


class SpamPredictor:
    """
    Handles real-time classification of SMS messages.
    
    The SpamPredictor class integrates the text preprocessor, TF-IDF vectorizer,
    and trained model to classify new messages as "ham" or "spam" with confidence scores.
    """
    
    def __init__(self, model, vectorizer, preprocessor):
        """
        Initialize the SpamPredictor.
        
        Args:
            model: Trained classifier (e.g., MultinomialNB)
            vectorizer: Fitted TF-IDF vectorizer
            preprocessor: TextPreprocessor instance for text transformation
            
        Raises:
            ValueError: If any component is None
        """
        if model is None:
            raise ValueError("Model cannot be None")
        if vectorizer is None:
            raise ValueError("Vectorizer cannot be None")
        if preprocessor is None:
            raise ValueError("Preprocessor cannot be None")
        
        self.model = model
        self.vectorizer = vectorizer
        self.preprocessor = preprocessor
    
    def predict(self, message: str) -> str:
        """
        Classify a single message as "ham" or "spam".
        
        Applies the same preprocessing pipeline used during training,
        transforms the message using the fitted vectorizer, and returns
        the classification result.
        
        Args:
            message: Raw text message to classify
            
        Returns:
            Classification result: "ham" or "spam"
            
        Raises:
            TypeError: If message is not a string
        """
        if not isinstance(message, str):
            raise TypeError(f"Message must be a string, got {type(message).__name__}")
        
        # Preprocess the message
        preprocessed = self.preprocessor.preprocess_text(message)
        
        # Transform to TF-IDF features
        features = self.vectorizer.transform([preprocessed])
        
        # Make prediction
        prediction = self.model.predict(features)[0]
        
        # Convert numeric prediction to label
        return "ham" if prediction == 0 else "spam"
    
    def predict_proba(self, message: str) -> Dict[str, float]:
        """
        Return confidence scores for both classes.
        
        Applies the same preprocessing pipeline used during training,
        transforms the message using the fitted vectorizer, and returns
        probability scores for both "ham" and "spam" classes.
        
        Args:
            message: Raw text message to classify
            
        Returns:
            Dictionary with confidence scores:
                - 'ham': Probability of being ham (0.0 to 1.0)
                - 'spam': Probability of being spam (0.0 to 1.0)
                
        Raises:
            TypeError: If message is not a string
        """
        if not isinstance(message, str):
            raise TypeError(f"Message must be a string, got {type(message).__name__}")
        
        # Preprocess the message
        preprocessed = self.preprocessor.preprocess_text(message)
        
        # Transform to TF-IDF features
        features = self.vectorizer.transform([preprocessed])
        
        # Get probability scores
        probabilities = self.model.predict_proba(features)[0]
        
        # Return as dictionary with class labels
        return {
            'ham': float(probabilities[0]),
            'spam': float(probabilities[1])
        }
