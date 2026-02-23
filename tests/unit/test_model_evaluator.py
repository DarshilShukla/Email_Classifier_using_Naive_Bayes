"""
Unit tests for ModelEvaluator class.
"""

import pytest
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from src.model_evaluator import ModelEvaluator


class MockModel:
    """Mock model for testing."""
    
    def __init__(self, predictions):
        self.predictions = predictions
    
    def predict(self, X):
        return self.predictions


def test_compute_accuracy_perfect():
    """Test accuracy computation with perfect predictions."""
    y_test = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    
    model = MockModel(y_pred)
    X_test = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    
    evaluator = ModelEvaluator(model, X_test, y_test)
    accuracy = evaluator.compute_accuracy()
    
    assert accuracy == 1.0


def test_compute_accuracy_zero():
    """Test accuracy computation with all wrong predictions."""
    y_test = np.array([0, 0, 1, 1])
    y_pred = np.array([1, 1, 0, 0])
    
    model = MockModel(y_pred)
    X_test = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    
    evaluator = ModelEvaluator(model, X_test, y_test)
    accuracy = evaluator.compute_accuracy()
    
    assert accuracy == 0.0


def test_compute_accuracy_partial():
    """Test accuracy computation with partial correct predictions."""
    y_test = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 0])
    
    model = MockModel(y_pred)
    X_test = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    
    evaluator = ModelEvaluator(model, X_test, y_test)
    accuracy = evaluator.compute_accuracy()
    
    assert accuracy == 0.5


def test_generate_confusion_matrix():
    """Test confusion matrix generation."""
    y_test = np.array([0, 0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 0, 0, 1])
    
    model = MockModel(y_pred)
    X_test = np.zeros((6, 2))
    
    evaluator = ModelEvaluator(model, X_test, y_test)
    cm = evaluator.generate_confusion_matrix()
    
    # Expected confusion matrix:
    # [[2, 1],   # 2 TN (ham->ham), 1 FP (ham->spam)
    #  [1, 2]]   # 1 FN (spam->ham), 2 TP (spam->spam)
    assert cm.shape == (2, 2)
    assert cm[0, 0] == 2  # True Negatives
    assert cm[0, 1] == 1  # False Positives
    assert cm[1, 0] == 1  # False Negatives
    assert cm[1, 1] == 2  # True Positives
    assert cm.sum() == len(y_test)


def test_compute_precision():
    """Test precision computation for both classes."""
    y_test = np.array([0, 0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 0, 0, 1])
    
    model = MockModel(y_pred)
    X_test = np.zeros((6, 2))
    
    evaluator = ModelEvaluator(model, X_test, y_test)
    precision = evaluator.compute_precision()
    
    # Precision for ham (class 0): TP / (TP + FP) = 2 / (2 + 1) = 0.667
    # Precision for spam (class 1): TP / (TP + FP) = 2 / (2 + 1) = 0.667
    assert 'ham' in precision
    assert 'spam' in precision
    assert 'weighted' in precision
    assert 0 <= precision['ham'] <= 1
    assert 0 <= precision['spam'] <= 1
    assert 0 <= precision['weighted'] <= 1


def test_compute_recall():
    """Test recall computation for both classes."""
    y_test = np.array([0, 0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 0, 0, 1])
    
    model = MockModel(y_pred)
    X_test = np.zeros((6, 2))
    
    evaluator = ModelEvaluator(model, X_test, y_test)
    recall = evaluator.compute_recall()
    
    # Recall for ham (class 0): TP / (TP + FN) = 2 / (2 + 1) = 0.667
    # Recall for spam (class 1): TP / (TP + FN) = 2 / (2 + 1) = 0.667
    assert 'ham' in recall
    assert 'spam' in recall
    assert 'weighted' in recall
    assert 0 <= recall['ham'] <= 1
    assert 0 <= recall['spam'] <= 1
    assert 0 <= recall['weighted'] <= 1


def test_compute_f1_score():
    """Test F1-score computation for both classes."""
    y_test = np.array([0, 0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 0, 0, 1])
    
    model = MockModel(y_pred)
    X_test = np.zeros((6, 2))
    
    evaluator = ModelEvaluator(model, X_test, y_test)
    f1 = evaluator.compute_f1_score()
    
    # F1 = 2 * (precision * recall) / (precision + recall)
    assert 'ham' in f1
    assert 'spam' in f1
    assert 'weighted' in f1
    assert 0 <= f1['ham'] <= 1
    assert 0 <= f1['spam'] <= 1
    assert 0 <= f1['weighted'] <= 1


def test_f1_score_formula():
    """Test that F1-score follows the correct formula."""
    y_test = np.array([0, 0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 0, 0, 1])
    
    model = MockModel(y_pred)
    X_test = np.zeros((6, 2))
    
    evaluator = ModelEvaluator(model, X_test, y_test)
    precision = evaluator.compute_precision()
    recall = evaluator.compute_recall()
    f1 = evaluator.compute_f1_score()
    
    # Verify F1 = 2 * (P * R) / (P + R) for each class
    for cls in ['ham', 'spam']:
        p = precision[cls]
        r = recall[cls]
        expected_f1 = 2 * (p * r) / (p + r) if (p + r) > 0 else 0
        assert abs(f1[cls] - expected_f1) < 1e-6


def test_generate_classification_report():
    """Test classification report generation."""
    y_test = np.array([0, 0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 0, 0, 1])
    
    model = MockModel(y_pred)
    X_test = np.zeros((6, 2))
    
    evaluator = ModelEvaluator(model, X_test, y_test)
    report = evaluator.generate_classification_report()
    
    assert isinstance(report, str)
    assert 'ham' in report
    assert 'spam' in report
    assert 'precision' in report
    assert 'recall' in report
    assert 'f1-score' in report


def test_evaluate_all_structure():
    """Test that evaluate_all returns all required metrics."""
    y_test = np.array([0, 0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 0, 0, 1])
    
    model = MockModel(y_pred)
    X_test = np.zeros((6, 2))
    
    evaluator = ModelEvaluator(model, X_test, y_test)
    results = evaluator.evaluate_all()
    
    # Verify all required keys are present
    assert 'accuracy' in results
    assert 'confusion_matrix' in results
    assert 'precision' in results
    assert 'recall' in results
    assert 'f1_score' in results
    assert 'classification_report' in results
    
    # Verify types
    assert isinstance(results['accuracy'], (float, np.floating))
    assert isinstance(results['confusion_matrix'], np.ndarray)
    assert isinstance(results['precision'], dict)
    assert isinstance(results['recall'], dict)
    assert isinstance(results['f1_score'], dict)
    assert isinstance(results['classification_report'], str)


def test_evaluate_all_consistency():
    """Test that evaluate_all returns consistent results with individual methods."""
    y_test = np.array([0, 0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 0, 0, 1])
    
    model = MockModel(y_pred)
    X_test = np.zeros((6, 2))
    
    evaluator = ModelEvaluator(model, X_test, y_test)
    
    # Get results from individual methods
    accuracy = evaluator.compute_accuracy()
    cm = evaluator.generate_confusion_matrix()
    precision = evaluator.compute_precision()
    recall = evaluator.compute_recall()
    f1 = evaluator.compute_f1_score()
    
    # Get results from evaluate_all
    all_results = evaluator.evaluate_all()
    
    # Verify consistency
    assert all_results['accuracy'] == accuracy
    assert np.array_equal(all_results['confusion_matrix'], cm)
    assert all_results['precision'] == precision
    assert all_results['recall'] == recall
    assert all_results['f1_score'] == f1
