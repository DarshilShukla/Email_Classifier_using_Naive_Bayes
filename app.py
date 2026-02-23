"""
Flask web application for SMS spam classification.

This module provides a web interface for classifying SMS messages
as "ham" or "spam" using a trained machine learning model.
"""

import os
import time
from flask import Flask, render_template, request, jsonify
from src.model_persistence import ModelPersistence
from src.text_preprocessor import TextPreprocessor
from src.spam_predictor import SpamPredictor

# Initialize Flask app
app = Flask(__name__)

# Global variables for model components
predictor = None
model_loaded = False


def load_model_components():
    """
    Load the trained model, vectorizer, and preprocessor on startup.
    
    Returns:
        tuple: (model, vectorizer, preprocessor) or (None, None, None) if loading fails
    """
    try:
        model_path = os.path.join('models', 'spam_classifier_model.pkl')
        vectorizer_path = os.path.join('models', 'tfidf_vectorizer.pkl')
        
        # Load model and vectorizer
        model = ModelPersistence.load_model(model_path)
        vectorizer = ModelPersistence.load_vectorizer(vectorizer_path)
        
        # Create preprocessor
        preprocessor = TextPreprocessor()
        
        return model, vectorizer, preprocessor
    except Exception as e:
        print(f"Error loading model components: {str(e)}")
        return None, None, None


# Load model on startup
model, vectorizer, preprocessor = load_model_components()
if model is not None and vectorizer is not None and preprocessor is not None:
    predictor = SpamPredictor(model, vectorizer, preprocessor)
    model_loaded = True
    print("Model loaded successfully")
else:
    print("Warning: Model not loaded. Please train and save a model first.")


@app.route('/', methods=['GET'])
def home():
    """
    Serve the home page with the message input form.
    
    Returns:
        Rendered HTML template for the home page
    """
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle prediction requests for SMS messages.
    
    Expects JSON payload with 'message' field or form data.
    Returns JSON response with prediction, confidence, and processing time.
    
    Returns:
        JSON response with:
            - message: The input message
            - prediction: "ham" or "spam"
            - confidence: Probability scores for both classes
            - processing_time: Time taken for prediction in seconds
            
    Status Codes:
        200: Successful prediction
        400: Bad request (missing or empty message)
        500: Internal server error (model not loaded or prediction failed)
    """
    start_time = time.time()
    
    # Check if model is loaded
    if not model_loaded or predictor is None:
        return jsonify({
            'error': 'Model not loaded. Please train and save a model first.'
        }), 500
    
    try:
        # Get message from request (JSON or form data)
        if request.is_json:
            data = request.get_json()
            message = data.get('message', '').strip()
        else:
            message = request.form.get('message', '').strip()
        
        # Validate message
        if not message:
            return jsonify({
                'error': 'Message field is required and cannot be empty'
            }), 400
        
        # Make prediction
        prediction = predictor.predict(message)
        confidence = predictor.predict_proba(message)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Return response
        return jsonify({
            'message': message,
            'prediction': prediction,
            'confidence': confidence,
            'processing_time': round(processing_time, 4)
        }), 200
        
    except Exception as e:
        return jsonify({
            'error': f'Prediction failed: {str(e)}'
        }), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
