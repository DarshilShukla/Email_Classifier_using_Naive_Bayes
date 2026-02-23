"""
Unit tests for the Flask web interface.

Tests the home route, predict endpoint, error handling, and response times.
"""

import pytest
import sys
import os
import time

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from app import app
from src.model_trainer import ModelTrainer
from src.text_preprocessor import TextPreprocessor
from src.model_persistence import ModelPersistence
from src.data_cleaner import DataCleaner


@pytest.fixture
def client():
    """Create a test client for the Flask app."""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


@pytest.fixture(scope='module')
def ensure_model_exists():
    """
    Ensure a trained model exists for testing.
    
    If models don't exist, train a minimal model for testing purposes.
    """
    model_path = os.path.join('models', 'spam_classifier.pkl')
    vectorizer_path = os.path.join('models', 'tfidf_vectorizer.pkl')
    
    # Check if models already exist
    if os.path.exists(model_path) and os.path.exists(vectorizer_path):
        return
    
    # Train a minimal model for testing
    try:
        # Load and clean data
        cleaner = DataCleaner()
        df = cleaner.clean_pipeline('data/spam.csv')
        
        # Preprocess text
        preprocessor = TextPreprocessor()
        preprocessed_messages = df['message'].apply(preprocessor.preprocess_text).tolist()
        
        # Vectorize
        X = preprocessor.fit_transform(preprocessed_messages)
        y = df['label'].values
        
        # Train model
        trainer = ModelTrainer()
        X_train, X_test, y_train, y_test = trainer.split_data(X, y)
        trainer.train(X_train, y_train)
        
        # Save model and vectorizer
        ModelPersistence.save_model(trainer.get_model(), model_path)
        ModelPersistence.save_vectorizer(preprocessor.vectorizer, vectorizer_path)
        
    except Exception as e:
        pytest.skip(f"Could not create test model: {str(e)}")


def test_home_route_returns_html(client, ensure_model_exists):
    """
    Test that the home route returns HTML content.
    
    Requirements: 8.1, 8.2
    """
    response = client.get('/')
    
    assert response.status_code == 200
    assert response.content_type == 'text/html; charset=utf-8'
    assert b'SMS Spam Classifier' in response.data
    assert b'<form' in response.data
    assert b'<textarea' in response.data
    assert b'<button' in response.data


def test_predict_endpoint_with_valid_message(client, ensure_model_exists):
    """
    Test the predict endpoint with a valid message.
    
    Requirements: 8.2, 8.3
    """
    response = client.post('/predict', json={
        'message': 'Hello, how are you doing today?'
    })
    
    assert response.status_code == 200
    
    data = response.get_json()
    assert 'message' in data
    assert 'prediction' in data
    assert 'confidence' in data
    assert 'processing_time' in data
    
    # Verify prediction is valid
    assert data['prediction'] in ['ham', 'spam']
    
    # Verify confidence structure
    assert 'ham' in data['confidence']
    assert 'spam' in data['confidence']
    assert 0 <= data['confidence']['ham'] <= 1
    assert 0 <= data['confidence']['spam'] <= 1
    
    # Verify processing time is reasonable
    assert data['processing_time'] > 0


def test_predict_endpoint_with_empty_message(client, ensure_model_exists):
    """
    Test the predict endpoint with an empty message.
    
    Should return 400 Bad Request.
    
    Requirements: 8.3
    """
    response = client.post('/predict', json={
        'message': ''
    })
    
    assert response.status_code == 400
    
    data = response.get_json()
    assert 'error' in data


def test_predict_endpoint_with_whitespace_only(client, ensure_model_exists):
    """
    Test the predict endpoint with whitespace-only message.
    
    Should return 400 Bad Request.
    
    Requirements: 8.3
    """
    response = client.post('/predict', json={
        'message': '   '
    })
    
    assert response.status_code == 400
    
    data = response.get_json()
    assert 'error' in data


def test_predict_endpoint_with_missing_message_field(client, ensure_model_exists):
    """
    Test the predict endpoint without message field.
    
    Should return 400 Bad Request.
    
    Requirements: 8.3
    """
    response = client.post('/predict', json={})
    
    assert response.status_code == 400
    
    data = response.get_json()
    assert 'error' in data


def test_predict_endpoint_with_form_data(client, ensure_model_exists):
    """
    Test the predict endpoint with form data instead of JSON.
    
    Requirements: 8.2, 8.3
    """
    response = client.post('/predict', data={
        'message': 'This is a test message'
    })
    
    assert response.status_code == 200
    
    data = response.get_json()
    assert 'prediction' in data
    assert data['prediction'] in ['ham', 'spam']


def test_response_time_is_reasonable(client, ensure_model_exists):
    """
    Test that prediction response time is within acceptable limits.
    
    Requirements: 8.4
    """
    start_time = time.time()
    
    response = client.post('/predict', json={
        'message': 'Free entry in 2 a wkly comp to win FA Cup final tkts'
    })
    
    elapsed_time = time.time() - start_time
    
    assert response.status_code == 200
    assert elapsed_time < 2.0  # Should complete within 2 seconds
    
    data = response.get_json()
    assert data['processing_time'] < 2.0


def test_predict_with_spam_message(client, ensure_model_exists):
    """
    Test prediction with a typical spam message.
    
    Requirements: 8.2, 8.3
    """
    spam_message = "WINNER!! You have won a $1000 prize! Call now to claim your reward!"
    
    response = client.post('/predict', json={
        'message': spam_message
    })
    
    assert response.status_code == 200
    
    data = response.get_json()
    assert data['message'] == spam_message
    assert data['prediction'] in ['ham', 'spam']
    assert 'confidence' in data


def test_predict_with_ham_message(client, ensure_model_exists):
    """
    Test prediction with a typical legitimate message.
    
    Requirements: 8.2, 8.3
    """
    ham_message = "Hey, are we still meeting for lunch tomorrow?"
    
    response = client.post('/predict', json={
        'message': ham_message
    })
    
    assert response.status_code == 200
    
    data = response.get_json()
    assert data['message'] == ham_message
    assert data['prediction'] in ['ham', 'spam']
    assert 'confidence' in data


def test_predict_with_long_message(client, ensure_model_exists):
    """
    Test prediction with a very long message.
    
    Requirements: 8.2, 8.3
    """
    long_message = "This is a test message. " * 100  # 2500+ characters
    
    response = client.post('/predict', json={
        'message': long_message
    })
    
    assert response.status_code == 200
    
    data = response.get_json()
    assert 'prediction' in data


def test_home_route_contains_form_elements(client, ensure_model_exists):
    """
    Test that the home page contains all required form elements.
    
    Requirements: 8.1
    """
    response = client.get('/')
    
    assert response.status_code == 200
    
    html = response.data.decode('utf-8')
    
    # Check for text input field
    assert 'textarea' in html
    assert 'message' in html
    
    # Check for submit button
    assert 'button' in html or 'submit' in html.lower()
    
    # Check for result display area
    assert 'result' in html


def test_json_response_structure(client, ensure_model_exists):
    """
    Test that JSON response has the correct structure.
    
    Requirements: 8.3
    """
    response = client.post('/predict', json={
        'message': 'Test message'
    })
    
    assert response.status_code == 200
    assert response.content_type == 'application/json'
    
    data = response.get_json()
    
    # Verify all required fields are present
    required_fields = ['message', 'prediction', 'confidence', 'processing_time']
    for field in required_fields:
        assert field in data, f"Missing required field: {field}"
    
    # Verify confidence has both ham and spam
    assert 'ham' in data['confidence']
    assert 'spam' in data['confidence']
