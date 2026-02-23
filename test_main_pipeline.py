#!/usr/bin/env python3
"""
Integration test for main.py pipeline.

This script creates a small test dataset and runs the complete pipeline
to verify all components work together correctly.
"""

import os
import tempfile
import shutil
import pandas as pd
from main import main


def create_test_dataset(file_path: str):
    """Create a small test dataset for pipeline testing."""
    # Create a dataset with enough unique samples for training
    ham_base = [
        'Hey, how are you doing today?',
        'Can we meet for coffee tomorrow?',
        'Thanks for the help with the project',
        'See you at the meeting later',
        'Great job on the presentation!',
        'Let me know when you are free',
        'Did you finish the homework?',
        'I will call you later tonight',
        'Happy birthday! Hope you have a great day',
        'Thanks for the dinner invitation',
    ]
    
    spam_base = [
        'WINNER! You have won $1000! Click here now!',
        'Congratulations! Free iPhone waiting for you!',
        'URGENT: Your account needs verification immediately',
        'You have been selected for a special offer!',
        'Click here to claim your prize now!',
        'Limited time offer! Act now!',
        'Your loan has been approved! Call now!',
        'Free gift card waiting for you!',
        'Congratulations! You won the lottery!',
        'Claim your free vacation package today!',
    ]
    
    # Create unique messages by adding numbers
    ham_messages = []
    spam_messages = []
    
    for i in range(80):  # Create 80 unique ham messages
        msg = ham_base[i % len(ham_base)]
        ham_messages.append(f"{msg} {i}")
    
    for i in range(80):  # Create 80 unique spam messages
        msg = spam_base[i % len(spam_base)]
        spam_messages.append(f"{msg} {i}")
    
    data = {
        'v1': ['ham'] * len(ham_messages) + ['spam'] * len(spam_messages),
        'v2': ham_messages + spam_messages,
        'Unnamed: 2': [None] * (len(ham_messages) + len(spam_messages)),
        'Unnamed: 3': [None] * (len(ham_messages) + len(spam_messages)),
        'Unnamed: 4': [None] * (len(ham_messages) + len(spam_messages))
    }
    
    df = pd.DataFrame(data)
    df.to_csv(file_path, index=False, encoding='latin-1')
    print(f"Created test dataset with {len(df)} messages at {file_path}")


def test_pipeline():
    """Test the complete pipeline with a test dataset."""
    print("=" * 70)
    print("TESTING MAIN PIPELINE")
    print("=" * 70)
    print()
    
    # Create temporary directories
    temp_dir = tempfile.mkdtemp()
    data_dir = os.path.join(temp_dir, "data")
    model_dir = os.path.join(temp_dir, "models")
    os.makedirs(data_dir, exist_ok=True)
    
    try:
        # Create test dataset
        test_data_path = os.path.join(data_dir, "spam.csv")
        create_test_dataset(test_data_path)
        print()
        
        # Run the pipeline
        main(data_path=test_data_path, model_dir=model_dir)
        
        # Verify outputs
        print()
        print("=" * 70)
        print("VERIFICATION")
        print("=" * 70)
        print()
        
        model_path = os.path.join(model_dir, "spam_classifier_model.pkl")
        vectorizer_path = os.path.join(model_dir, "tfidf_vectorizer.pkl")
        
        if os.path.exists(model_path):
            print(f"✓ Model file created: {model_path}")
            print(f"  Size: {os.path.getsize(model_path)} bytes")
        else:
            print(f"✗ Model file not found: {model_path}")
        
        if os.path.exists(vectorizer_path):
            print(f"✓ Vectorizer file created: {vectorizer_path}")
            print(f"  Size: {os.path.getsize(vectorizer_path)} bytes")
        else:
            print(f"✗ Vectorizer file not found: {vectorizer_path}")
        
        print()
        print("=" * 70)
        print("TEST COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n✗ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    finally:
        # Cleanup
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print(f"\nCleaned up temporary directory: {temp_dir}")


if __name__ == "__main__":
    test_pipeline()
