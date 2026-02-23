"""
Model evaluation module for SMS spam classifier.

This module provides the ModelEvaluator class for computing performance metrics
including accuracy, precision, recall, F1-score, and confusion matrix.
"""

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)
import numpy as np


class ModelEvaluator:
    """
    Evaluates classification model performance.
    
    This class computes various performance metrics for a trained classifier
    including accuracy, confusion matrix, precision, recall, and F1-score.
    """
    
    def __init__(self, model, X_test, y_test):
        """
        Initialize the evaluator with model and test data.
        
        Args:
            model: Trained classifier with predict() method
            X_test: Test features (sparse matrix or array)
            y_test: True labels for test data (array)
        """
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.y_pred = model.predict(X_test)
    
    def compute_accuracy(self):
        """
        Compute overall classification accuracy.
        
        Returns:
            float: Accuracy score (correct predictions / total predictions)
        """
        return accuracy_score(self.y_test, self.y_pred)
    
    def generate_confusion_matrix(self):
        """
        Generate confusion matrix for binary classification.
        
        Returns:
            numpy.ndarray: 2x2 confusion matrix where:
                - [0,0] = True Negatives (ham predicted as ham)
                - [0,1] = False Positives (ham predicted as spam)
                - [1,0] = False Negatives (spam predicted as ham)
                - [1,1] = True Positives (spam predicted as spam)
        """
        return confusion_matrix(self.y_test, self.y_pred, labels=[0, 1])
    
    def compute_precision(self):
        """
        Compute precision for both classes.
        
        Precision = True Positives / (True Positives + False Positives)
        
        Returns:
            dict: Precision scores with keys:
                - 'ham': Precision for class 0 (ham)
                - 'spam': Precision for class 1 (spam)
                - 'weighted': Weighted average precision
        """
        precision_per_class = precision_score(self.y_test, self.y_pred, average=None, labels=[0, 1], zero_division=0)
        precision_weighted = precision_score(self.y_test, self.y_pred, average='weighted', zero_division=0)
        
        return {
            'ham': precision_per_class[0],
            'spam': precision_per_class[1],
            'weighted': precision_weighted
        }
    
    def compute_recall(self):
        """
        Compute recall for both classes.
        
        Recall = True Positives / (True Positives + False Negatives)
        
        Returns:
            dict: Recall scores with keys:
                - 'ham': Recall for class 0 (ham)
                - 'spam': Recall for class 1 (spam)
                - 'weighted': Weighted average recall
        """
        recall_per_class = recall_score(self.y_test, self.y_pred, average=None, labels=[0, 1], zero_division=0)
        recall_weighted = recall_score(self.y_test, self.y_pred, average='weighted', zero_division=0)
        
        return {
            'ham': recall_per_class[0],
            'spam': recall_per_class[1],
            'weighted': recall_weighted
        }
    
    def compute_f1_score(self):
        """
        Compute F1-score for both classes.
        
        F1-score = 2 * (Precision * Recall) / (Precision + Recall)
        
        Returns:
            dict: F1-scores with keys:
                - 'ham': F1-score for class 0 (ham)
                - 'spam': F1-score for class 1 (spam)
                - 'weighted': Weighted average F1-score
        """
        f1_per_class = f1_score(self.y_test, self.y_pred, average=None, labels=[0, 1], zero_division=0)
        f1_weighted = f1_score(self.y_test, self.y_pred, average='weighted', zero_division=0)
        
        return {
            'ham': f1_per_class[0],
            'spam': f1_per_class[1],
            'weighted': f1_weighted
        }
    
    def generate_classification_report(self):
        """
        Generate detailed classification report.
        
        Returns:
            str: Formatted classification report showing precision, recall,
                 F1-score, and support for each class
        """
        target_names = ['ham', 'spam']
        return classification_report(self.y_test, self.y_pred, labels=[0, 1], target_names=target_names, zero_division=0)
    
    def evaluate_all(self):
        """
        Compute all evaluation metrics and return as structured dictionary.
        
        Returns:
            dict: Dictionary containing all metrics:
                - 'accuracy': Overall accuracy score
                - 'confusion_matrix': 2x2 numpy array
                - 'precision': Dict with ham, spam, and weighted precision
                - 'recall': Dict with ham, spam, and weighted recall
                - 'f1_score': Dict with ham, spam, and weighted F1-scores
                - 'classification_report': Formatted string report
        """
        return {
            'accuracy': self.compute_accuracy(),
            'confusion_matrix': self.generate_confusion_matrix(),
            'precision': self.compute_precision(),
            'recall': self.compute_recall(),
            'f1_score': self.compute_f1_score(),
            'classification_report': self.generate_classification_report()
        }
