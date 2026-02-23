# SMS Spam Classifier - Main Pipeline

This document describes how to use the `main.py` script to train the SMS spam classification model.

## Overview

The `main.py` script executes the complete machine learning pipeline:

1. **Data Loading & Cleaning** - Loads and cleans the UCI SMS Spam Collection dataset
2. **Exploratory Data Analysis** - Computes message statistics and class distribution
3. **Text Preprocessing** - Applies lowercase, tokenization, stopword removal, stemming, and TF-IDF vectorization
4. **Model Training** - Trains a Multinomial Naive Bayes classifier with 5-fold cross-validation
5. **Model Evaluation** - Computes accuracy, confusion matrix, precision, recall, and F1-scores
6. **Model Persistence** - Saves the trained model and vectorizer to disk
7. **Prediction Testing** - Tests the prediction pipeline with sample messages

## Prerequisites

1. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Download NLTK data:
   ```bash
   python setup_nltk.py
   ```

3. Download the UCI SMS Spam Collection dataset:
   - Visit: https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection
   - Download the `spam.csv` file
   - Place it in the `data/` directory

## Usage

### Basic Usage

Run the pipeline with default settings:

```bash
python main.py
```

This will:
- Load data from `data/spam.csv`
- Save the trained model to `models/spam_classifier_model.pkl`
- Save the vectorizer to `models/tfidf_vectorizer.pkl`

### Custom Data Path

Specify a custom path to the dataset:

```bash
python main.py --data-path /path/to/your/spam.csv
```

### Custom Model Directory

Specify a custom directory for saving the model:

```bash
python main.py --model-dir /path/to/model/directory
```

### Combined Options

```bash
python main.py --data-path data/spam.csv --model-dir trained_models
```

## Output

The script provides detailed progress information for each step:

```
======================================================================
SMS SPAM CLASSIFIER - TRAINING PIPELINE
======================================================================

[1/7] Loading and cleaning data...
      ✓ Dataset loaded: 5572 messages
      ✓ Columns: ['label', 'message']

[2/7] Performing exploratory data analysis...
      ✓ Computed message statistics
      ✓ Class distribution:
        - Ham:  86.6%
        - Spam: 13.4%

[3/7] Preprocessing text messages...
      ✓ Preprocessed 5572 messages
      ✓ Feature matrix shape: (5572, 7783)

[4/7] Training model...
      ✓ Data split: 4457 train, 1115 test
      ✓ Model trained: Multinomial Naive Bayes
      ✓ Cross-validation (5-fold):
        - Mean accuracy: 0.9820
        - Std deviation: 0.0045

[5/7] Evaluating model performance...
      ✓ Test Set Accuracy: 0.9820
      
      Confusion Matrix:
      [[965   0]
       [ 20 130]]
      
      Per-Class Metrics:
        Ham  - Precision: 0.9797, Recall: 1.0000, F1: 0.9897
        Spam - Precision: 1.0000, Recall: 0.8667, F1: 0.9286

[6/7] Saving model and vectorizer...
      ✓ Model saved to: models/spam_classifier_model.pkl
      ✓ Vectorizer saved to: models/tfidf_vectorizer.pkl

[7/7] Testing prediction pipeline...
      Testing with sample messages:
      1. Message: "Congratulations! You've won a free iPhone..."
         Prediction: SPAM (confidence: 95.80%)
      2. Message: "Hey, are we still meeting for lunch tomorrow?..."
         Prediction: HAM (confidence: 94.72%)

======================================================================
PIPELINE COMPLETED SUCCESSFULLY!
======================================================================
```

## Expected Results

When trained on the full UCI SMS Spam Collection dataset, the model should achieve:

- **Accuracy**: ≥97% on the test set
- **Precision**: High for both ham and spam classes
- **Recall**: High for both ham and spam classes
- **F1-Score**: ≥0.97 overall

## Files Generated

After running the pipeline, the following files will be created:

- `models/spam_classifier_model.pkl` - Trained Multinomial Naive Bayes classifier
- `models/tfidf_vectorizer.pkl` - Fitted TF-IDF vectorizer

These files can be loaded and used for real-time spam classification using the `SpamPredictor` class.

## Using the Trained Model

After training, you can load and use the model for predictions:

```python
from src.model_persistence import ModelPersistence
from src.text_preprocessor import TextPreprocessor
from src.spam_predictor import SpamPredictor

# Load the saved model and vectorizer
persistence = ModelPersistence()
model = persistence.load_model("models/spam_classifier_model.pkl")
vectorizer = persistence.load_vectorizer("models/tfidf_vectorizer.pkl")

# Create preprocessor and predictor
preprocessor = TextPreprocessor()
predictor = SpamPredictor(model, vectorizer, preprocessor)

# Classify a new message
message = "Congratulations! You've won a free prize!"
prediction = predictor.predict(message)
probabilities = predictor.predict_proba(message)

print(f"Prediction: {prediction}")
print(f"Confidence: {probabilities[prediction]:.2%}")
```

## Testing

To test the pipeline with a synthetic dataset:

```bash
python test_main_pipeline.py
```

This creates a small test dataset and runs the complete pipeline to verify all components work correctly.

## Troubleshooting

### Dataset Not Found

If you see the error:
```
✗ Error: Dataset file not found at data/spam.csv
```

Make sure you have downloaded the UCI SMS Spam Collection dataset and placed it in the `data/` directory.

### NLTK Data Missing

If you encounter NLTK-related errors, run:
```bash
python setup_nltk.py
```

### Insufficient Training Data

The model requires at least 100 training samples. If you're using a custom dataset, ensure it has at least 125 messages (to account for the 80/20 train/test split).

## Visualizations

By default, visualizations are disabled to allow automated runs. To enable visualizations:

1. Open `main.py`
2. Uncomment the visualization code in Step 2 (lines ~90-96)
3. Run the script

This will display:
- Histograms showing character count, word count, and sentence count distributions
- Correlation heatmap for computed features

## Requirements Validated

This pipeline validates the following requirements from the specification:

- **Requirements 1.1-1.6**: Data loading and cleaning
- **Requirements 2.1-2.5**: Exploratory data analysis
- **Requirements 3.1-3.7**: Text preprocessing
- **Requirements 4.1-4.3**: Model training
- **Requirements 5.1-5.6**: Model evaluation
- **Requirements 7.1-7.2**: Model persistence

## Next Steps

After training the model, you can:

1. Deploy the model using the `SpamPredictor` class
2. Build a web interface (see task 13 in the implementation plan)
3. Integrate the model into your application
4. Fine-tune hyperparameters for better performance


## Deployment

This section covers deploying the SMS spam classifier as a web application.

### Local Testing

Before deploying to production, test the application locally:

1. **Install web dependencies** (if implementing the web interface):
   ```bash
   pip install flask gunicorn
   ```

2. **Run the Flask development server**:
   ```bash
   python app.py
   ```
   
   The application will be available at `http://localhost:5000`

3. **Test the API endpoint**:
   ```bash
   curl -X POST http://localhost:5000/predict \
     -H "Content-Type: application/json" \
     -d '{"message": "Congratulations! You won a prize!"}'
   ```

4. **Expected response**:
   ```json
   {
     "message": "Congratulations! You won a prize!",
     "prediction": "spam",
     "confidence": 0.9580,
     "processing_time": 0.023
   }
   ```

### Heroku Deployment

Deploy the SMS spam classifier to Heroku for public access:

#### Prerequisites

1. **Create a Heroku account** at https://signup.heroku.com/
2. **Install Heroku CLI**:
   - Windows: Download from https://devcenter.heroku.com/articles/heroku-cli
   - macOS: `brew tap heroku/brew && brew install heroku`
   - Linux: `curl https://cli-assets.heroku.com/install.sh | sh`
3. **Login to Heroku**:
   ```bash
   heroku login
   ```

#### Deployment Steps

1. **Initialize Git repository** (if not already done):
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   ```

2. **Create Heroku application**:
   ```bash
   heroku create your-spam-classifier-app
   ```
   
   Replace `your-spam-classifier-app` with your desired app name (must be unique across Heroku).

3. **Verify deployment files exist**:
   - `Procfile` - Tells Heroku how to run your app
   - `runtime.txt` - Specifies Python version
   - `requirements.txt` - Lists all dependencies

4. **Add NLTK data buildpack** (required for NLTK stopwords and tokenizers):
   ```bash
   heroku buildpacks:add heroku/python
   ```

5. **Set environment variables** (optional):
   ```bash
   heroku config:set FLASK_ENV=production
   heroku config:set MODEL_PATH=models/spam_classifier_model.pkl
   ```

6. **Deploy to Heroku**:
   ```bash
   git push heroku main
   ```
   
   If your default branch is `master`:
   ```bash
   git push heroku master
   ```

7. **Ensure at least one instance is running**:
   ```bash
   heroku ps:scale web=1
   ```

8. **Open your deployed application**:
   ```bash
   heroku open
   ```

#### Post-Deployment Verification

1. **Check application logs**:
   ```bash
   heroku logs --tail
   ```

2. **Test the deployed API**:
   ```bash
   curl -X POST https://your-spam-classifier-app.herokuapp.com/predict \
     -H "Content-Type: application/json" \
     -d '{"message": "Win a free iPhone now!"}'
   ```

3. **Monitor application health**:
   ```bash
   heroku ps
   ```

#### Troubleshooting Heroku Deployment

**Issue: Application crashes on startup**
- Check logs: `heroku logs --tail`
- Verify `Procfile` is correctly formatted
- Ensure all dependencies are in `requirements.txt`

**Issue: NLTK data not found**
- Add NLTK download to your startup script
- Ensure `setup_nltk.py` runs before the app starts
- Consider adding NLTK data to your repository in a `nltk_data/` directory

**Issue: Model files not found**
- Ensure model files are committed to Git (not in `.gitignore`)
- Verify model paths in your application code
- Check that models are loaded correctly on startup

**Issue: Slug size too large**
- Remove unnecessary files from the repository
- Use `.slugignore` to exclude files from deployment
- Consider storing models in cloud storage (S3, Google Cloud Storage)

**Issue: Request timeout**
- Optimize model loading (load once on startup, not per request)
- Increase timeout settings if needed
- Consider using a worker dyno for long-running tasks

### Alternative Deployment Options

#### AWS Elastic Beanstalk

1. Install AWS CLI and EB CLI
2. Initialize Elastic Beanstalk:
   ```bash
   eb init -p python-3.9 spam-classifier
   ```
3. Create environment and deploy:
   ```bash
   eb create spam-classifier-env
   eb deploy
   ```

#### Google Cloud Platform (App Engine)

1. Create `app.yaml`:
   ```yaml
   runtime: python39
   entrypoint: gunicorn -b :$PORT app:app
   ```
2. Deploy:
   ```bash
   gcloud app deploy
   ```

#### Docker Container

1. Create `Dockerfile`:
   ```dockerfile
   FROM python:3.9-slim
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install -r requirements.txt
   COPY . .
   RUN python setup_nltk.py
   CMD ["gunicorn", "-b", "0.0.0.0:8000", "app:app"]
   ```
2. Build and run:
   ```bash
   docker build -t spam-classifier .
   docker run -p 8000:8000 spam-classifier
   ```

### Production Considerations

When deploying to production, consider:

1. **Security**:
   - Use HTTPS for all API endpoints
   - Implement rate limiting to prevent abuse
   - Validate and sanitize all user inputs
   - Set up CORS policies appropriately

2. **Performance**:
   - Load model once on startup, not per request
   - Use caching for frequently classified messages
   - Consider using a CDN for static assets
   - Monitor response times and optimize bottlenecks

3. **Monitoring**:
   - Set up application monitoring (New Relic, Datadog, etc.)
   - Track prediction accuracy over time
   - Monitor error rates and response times
   - Set up alerts for critical issues

4. **Scalability**:
   - Use auto-scaling to handle traffic spikes
   - Consider using a message queue for async processing
   - Implement load balancing for multiple instances
   - Cache model predictions when appropriate

5. **Model Updates**:
   - Implement versioning for models
   - Use blue-green deployment for model updates
   - A/B test new models before full rollout
   - Keep rollback capability for quick recovery

### Cost Optimization

- **Heroku Free Tier**: Suitable for testing and low-traffic applications
- **Heroku Hobby Tier** ($7/month): For small production applications
- **Heroku Professional** ($25+/month): For production applications with higher traffic
- Consider serverless options (AWS Lambda, Google Cloud Functions) for sporadic usage

### Support and Maintenance

For ongoing support:
- Monitor application logs regularly
- Keep dependencies updated for security patches
- Retrain model periodically with new data
- Collect user feedback to improve accuracy
- Document any configuration changes

