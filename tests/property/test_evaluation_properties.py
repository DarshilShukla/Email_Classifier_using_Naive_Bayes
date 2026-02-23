"""
Property-based tests for model evaluation.

These tests verify correctness properties for the ModelEvaluator class
using Hypothesis for property-based testing.
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, assume
from src.model_evaluator import ModelEvaluator


class MockModel:
    """Mock model for testing."""
    
    def __init__(self, predictions):
        self.predictions = predictions
    
    def predict(self, X):
        return self.predictions


# Strategy for generating binary labels
binary_labels = st.lists(
    st.integers(min_value=0, max_value=1),
    min_size=10,
    max_size=100
)


@given(y_true=binary_labels, y_pred=binary_labels)
def test_property_14_accuracy_computation(y_true, y_pred):
    """
    Property 14: Accuracy computation
    
    For any set of predictions and true labels, accuracy should equal
    the number of correct predictions divided by the total number of predictions.
    
    **Validates: Requirements 5.1**
    """
    # Ensure both arrays have the same length
    min_len = min(len(y_true), len(y_pred))
    assume(min_len >= 10)
    
    y_true = np.array(y_true[:min_len])
    y_pred = np.array(y_pred[:min_len])
    
    # Create mock model and evaluator
    model = MockModel(y_pred)
    X_test = np.zeros((len(y_true), 2))
    
    evaluator = ModelEvaluator(model, X_test, y_true)
    accuracy = evaluator.compute_accuracy()
    
    # Manually compute expected accuracy
    correct_predictions = np.sum(y_true == y_pred)
    total_predictions = len(y_true)
    expected_accuracy = correct_predictions / total_predictions
    
    # Verify accuracy formula
    assert abs(accuracy - expected_accuracy) < 1e-10
    assert 0 <= accuracy <= 1


@given(y_true=binary_labels, y_pred=binary_labels)
def test_property_15_confusion_matrix_dimensions(y_true, y_pred):
    """
    Property 15: Confusion matrix dimensions
    
    For any binary classification problem, the confusion matrix should be
    a 2×2 matrix where the sum of all elements equals the total number of predictions.
    
    **Validates: Requirements 5.2**
    """
    # Ensure both arrays have the same length
    min_len = min(len(y_true), len(y_pred))
    assume(min_len >= 10)
    
    y_true = np.array(y_true[:min_len])
    y_pred = np.array(y_pred[:min_len])
    
    # Create mock model and evaluator
    model = MockModel(y_pred)
    X_test = np.zeros((len(y_true), 2))
    
    evaluator = ModelEvaluator(model, X_test, y_true)
    cm = evaluator.generate_confusion_matrix()
    
    # Verify confusion matrix is 2x2
    assert cm.shape == (2, 2)
    
    # Verify sum equals total predictions
    assert cm.sum() == len(y_true)
    
    # Verify all elements are non-negative
    assert np.all(cm >= 0)


@given(y_true=binary_labels, y_pred=binary_labels)
def test_property_16_per_class_metrics_computation(y_true, y_pred):
    """
    Property 16: Per-class metrics computation
    
    For any set of predictions and true labels, precision, recall, and F1-score
    should be computable for each class, with values between 0 and 1, and
    F1-score should equal 2 × (precision × recall) / (precision + recall).
    
    **Validates: Requirements 5.3, 5.4, 5.5**
    """
    # Ensure both arrays have the same length
    min_len = min(len(y_true), len(y_pred))
    assume(min_len >= 10)
    
    y_true = np.array(y_true[:min_len])
    y_pred = np.array(y_pred[:min_len])
    
    # Ensure both classes are present in y_true to avoid division by zero
    assume(0 in y_true and 1 in y_true)
    
    # Create mock model and evaluator
    model = MockModel(y_pred)
    X_test = np.zeros((len(y_true), 2))
    
    evaluator = ModelEvaluator(model, X_test, y_true)
    
    precision = evaluator.compute_precision()
    recall = evaluator.compute_recall()
    f1 = evaluator.compute_f1_score()
    
    # Verify all metrics are between 0 and 1
    for cls in ['ham', 'spam']:
        assert 0 <= precision[cls] <= 1
        assert 0 <= recall[cls] <= 1
        assert 0 <= f1[cls] <= 1
    
    # Verify F1-score formula: F1 = 2 * (P * R) / (P + R)
    for cls in ['ham', 'spam']:
        p = precision[cls]
        r = recall[cls]
        
        if p + r > 0:
            expected_f1 = 2 * (p * r) / (p + r)
            assert abs(f1[cls] - expected_f1) < 1e-6
        else:
            # When both precision and recall are 0, F1 should be 0
            assert f1[cls] == 0


@given(y_true=binary_labels, y_pred=binary_labels)
def test_property_17_metrics_output_structure(y_true, y_pred):
    """
    Property 17: Metrics output structure
    
    For any evaluation result, the output should be a dictionary containing
    keys for accuracy, confusion_matrix, precision, recall, and f1_score.
    
    **Validates: Requirements 5.6**
    """
    # Ensure both arrays have the same length
    min_len = min(len(y_true), len(y_pred))
    assume(min_len >= 10)
    
    y_true = np.array(y_true[:min_len])
    y_pred = np.array(y_pred[:min_len])
    
    # Create mock model and evaluator
    model = MockModel(y_pred)
    X_test = np.zeros((len(y_true), 2))
    
    evaluator = ModelEvaluator(model, X_test, y_true)
    results = evaluator.evaluate_all()
    
    # Verify all required keys are present
    required_keys = ['accuracy', 'confusion_matrix', 'precision', 'recall', 'f1_score', 'classification_report']
    for key in required_keys:
        assert key in results
    
    # Verify types
    assert isinstance(results['accuracy'], (float, np.floating))
    assert isinstance(results['confusion_matrix'], np.ndarray)
    assert isinstance(results['precision'], dict)
    assert isinstance(results['recall'], dict)
    assert isinstance(results['f1_score'], dict)
    assert isinstance(results['classification_report'], str)
    
    # Verify nested dictionary structure
    for metric_dict in [results['precision'], results['recall'], results['f1_score']]:
        assert 'ham' in metric_dict
        assert 'spam' in metric_dict
        assert 'weighted' in metric_dict
