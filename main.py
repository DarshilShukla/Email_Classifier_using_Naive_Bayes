#!/usr/bin/env python3
"""
SMS Spam Classifier - Main Pipeline Script

This script runs the complete machine learning pipeline from raw data to trained model:
1. Load and clean the UCI SMS spam dataset
2. Perform exploratory data analysis with visualizations
3. Preprocess text messages
4. Train Multinomial Naive Bayes model with cross-validation
5. Evaluate model performance and display metrics
6. Save trained model and vectorizer to disk

Usage:
    python main.py [--data-path DATA_PATH] [--model-dir MODEL_DIR]
"""

import os
import sys
import argparse
from pathlib import Path

# Import all required modules
from src.data_cleaner import DataCleaner
from src.eda_analyzer import EDAAnalyzer
from src.text_preprocessor import TextPreprocessor
from src.model_trainer import ModelTrainer
from src.model_evaluator import ModelEvaluator
from src.model_persistence import ModelPersistence
from src.spam_predictor import SpamPredictor


def main(data_path: str = "data/spam.csv", model_dir: str = "models"):
    """
    Execute the complete SMS spam classification pipeline.
    
    Args:
        data_path: Path to the spam.csv dataset file
        model_dir: Directory where trained model and vectorizer will be saved
    """
    print("=" * 70)
    print("SMS SPAM CLASSIFIER - TRAINING PIPELINE")
    print("=" * 70)
    print()
    
    # ========================================================================
    # STEP 1: Load and Clean Data
    # ========================================================================
    print("[1/7] Loading and cleaning data...")
    print(f"      Data source: {data_path}")
    
    cleaner = DataCleaner()
    try:
        df = cleaner.clean_pipeline(data_path)
        print(f"      ✓ Dataset loaded: {len(df)} messages")
        print(f"      ✓ Columns: {list(df.columns)}")
    except FileNotFoundError:
        print(f"      ✗ Error: Dataset file not found at {data_path}")
        print(f"      Please download the UCI SMS Spam Collection dataset")
        print(f"      and place it at {data_path}")
        sys.exit(1)
    except Exception as e:
        print(f"      ✗ Error during data cleaning: {e}")
        sys.exit(1)
    
    print()
    
    # ========================================================================
    # STEP 2: Exploratory Data Analysis
    # ========================================================================
    print("[2/7] Performing exploratory data analysis...")
    
    analyzer = EDAAnalyzer()
    
    # Compute message statistics
    df = analyzer.compute_char_count(df)
    df = analyzer.compute_word_count(df)
    df = analyzer.compute_sentence_count(df)
    print(f"      ✓ Computed message statistics")
    
    # Get class distribution
    distribution = analyzer.get_class_distribution(df)
    print(f"      ✓ Class distribution:")
    print(f"        - Ham:  {distribution['ham_percentage']:.1f}%")
    print(f"        - Spam: {distribution['spam_percentage']:.1f}%")
    
    # Generate visualizations (optional - skipped in automated runs)
    # Uncomment the following lines to display visualizations:
    # try:
    #     print(f"      ✓ Generating visualizations...")
    #     analyzer.generate_histograms(df)
    #     analyzer.generate_correlation_heatmap(df)
    #     print(f"      ✓ Visualizations displayed")
    # except Exception as e:
    #     print(f"      ⚠ Visualization skipped (headless environment or display error)")
    print(f"      ⚠ Visualizations skipped (enable in code if needed)")
    
    print()
    
    # ========================================================================
    # STEP 3: Text Preprocessing
    # ========================================================================
    print("[3/7] Preprocessing text messages...")
    
    preprocessor = TextPreprocessor()
    
    # Apply preprocessing to all messages
    print(f"      ✓ Applying preprocessing pipeline...")
    messages = df['message'].tolist()
    preprocessed_messages = [preprocessor.preprocess_text(msg) for msg in messages]
    print(f"      ✓ Preprocessed {len(preprocessed_messages)} messages")
    
    # Vectorize using TF-IDF
    print(f"      ✓ Applying TF-IDF vectorization...")
    X = preprocessor.fit_transform(preprocessed_messages)
    y = df['label'].values
    print(f"      ✓ Feature matrix shape: {X.shape}")
    
    print()
    
    # ========================================================================
    # STEP 4: Train Model with Cross-Validation
    # ========================================================================
    print("[4/7] Training model...")
    
    trainer = ModelTrainer(model_type="multinomial_nb")
    
    # Split data
    X_train, X_test, y_train, y_test = trainer.split_data(X, y, test_size=0.2, random_state=42)
    print(f"      ✓ Data split: {len(y_train)} train, {len(y_test)} test")
    
    # Train model
    trainer.train(X_train, y_train)
    print(f"      ✓ Model trained: Multinomial Naive Bayes")
    
    # Cross-validation
    cv_results = trainer.cross_validate(X_train, y_train, cv=5)
    print(f"      ✓ Cross-validation (5-fold):")
    print(f"        - Mean accuracy: {cv_results['mean']:.4f}")
    print(f"        - Std deviation: {cv_results['std']:.4f}")
    
    print()
    
    # ========================================================================
    # STEP 5: Evaluate Model
    # ========================================================================
    print("[5/7] Evaluating model performance...")
    
    model = trainer.get_model()
    evaluator = ModelEvaluator(model, X_test, y_test)
    
    # Compute all metrics
    metrics = evaluator.evaluate_all()
    
    print(f"      ✓ Test Set Accuracy: {metrics['accuracy']:.4f}")
    print()
    print(f"      Confusion Matrix:")
    print(f"      {metrics['confusion_matrix']}")
    print()
    print(f"      Per-Class Metrics:")
    print(f"        Ham  - Precision: {metrics['precision']['ham']:.4f}, "
          f"Recall: {metrics['recall']['ham']:.4f}, "
          f"F1: {metrics['f1_score']['ham']:.4f}")
    print(f"        Spam - Precision: {metrics['precision']['spam']:.4f}, "
          f"Recall: {metrics['recall']['spam']:.4f}, "
          f"F1: {metrics['f1_score']['spam']:.4f}")
    print()
    print(f"      Classification Report:")
    print(metrics['classification_report'])
    
    print()
    
    # ========================================================================
    # STEP 6: Save Model and Vectorizer
    # ========================================================================
    print("[6/7] Saving model and vectorizer...")
    
    # Create model directory if it doesn't exist
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    
    model_path = os.path.join(model_dir, "spam_classifier_model.pkl")
    vectorizer_path = os.path.join(model_dir, "tfidf_vectorizer.pkl")
    
    persistence = ModelPersistence()
    persistence.save_model(model, model_path)
    persistence.save_vectorizer(preprocessor.vectorizer, vectorizer_path)
    
    print(f"      ✓ Model saved to: {model_path}")
    print(f"      ✓ Vectorizer saved to: {vectorizer_path}")
    
    print()
    
    # ========================================================================
    # STEP 7: Test Prediction Pipeline
    # ========================================================================
    print("[7/7] Testing prediction pipeline...")
    
    # Create predictor
    predictor = SpamPredictor(model, preprocessor.vectorizer, preprocessor)
    
    # Test with sample messages
    test_messages = [
        "Congratulations! You've won a free iPhone. Click here to claim now!",
        "Hey, are we still meeting for lunch tomorrow?",
        "URGENT: Your account has been compromised. Call us immediately!",
        "Thanks for the notes from class today, really helpful!"
    ]
    
    print(f"      Testing with sample messages:")
    for i, msg in enumerate(test_messages, 1):
        prediction = predictor.predict(msg)
        probabilities = predictor.predict_proba(msg)
        print(f"      {i}. Message: \"{msg[:50]}...\"")
        print(f"         Prediction: {prediction.upper()} "
              f"(confidence: {probabilities[prediction]:.2%})")
    
    print()
    print("=" * 70)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print()
    print(f"Model and vectorizer saved to: {model_dir}/")
    print(f"You can now use the trained model for real-time spam classification.")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train SMS spam classifier on UCI dataset"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/spam.csv",
        help="Path to spam.csv dataset (default: data/spam.csv)"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models",
        help="Directory to save trained model (default: models)"
    )
    
    args = parser.parse_args()
    
    main(data_path=args.data_path, model_dir=args.model_dir)
