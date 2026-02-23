# SMS Spam Classifier 📱🚫

An ML-powered SMS spam detection system using Multinomial Naive Bayes with 96%+ accuracy. Built with Python, scikit-learn, and NLTK, it processes messages through text preprocessing, TF-IDF vectorization, and real-time classification via a Flask web interface.

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Flask](https://img.shields.io/badge/Flask-2.3+-green.svg)](https://flask.palletsprojects.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🌟 Features

- **High Accuracy**: Achieves 96%+ accuracy on UCI SMS Spam Collection dataset
- **Real-time Classification**: Instant spam/ham predictions with confidence scores
- **Web Interface**: User-friendly Flask web app for message classification
- **Complete ML Pipeline**: End-to-end workflow from data cleaning to deployment
- **Comprehensive Testing**: Unit tests, property-based tests, and integration tests
- **Production Ready**: Includes deployment configuration for Heroku

## 📊 Model Performance

- **Accuracy**: 96.32%
- **Precision (Ham)**: 95.90%
- **Precision (Spam)**: 100%
- **Recall (Ham)**: 100%
- **Recall (Spam)**: 73.79%
- **Dataset**: 5,169 messages (87.4% ham, 12.6% spam)

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- pip package manager

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/DarshilShukla/Email_Classifier_using_Naive_Bayes.git
   cd Email_Classifier_using_Naive_Bayes
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download NLTK data**:
   ```bash
   python setup_nltk.py
   ```

4. **Download the dataset**:
   - Visit: [UCI SMS Spam Collection](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection)
   - Download `spam.csv`
   - Place it in the `data/` directory

### Training the Model

Run the complete ML pipeline:

```bash
python main.py
```

This will:
- Load and clean the dataset
- Perform exploratory data analysis
- Preprocess text messages
- Train the Multinomial Naive Bayes model
- Evaluate performance metrics
- Save the trained model and vectorizer

### Running the Web Application

Start the Flask web server:

```bash
python app.py
```

Access the application at: `http://localhost:5000`

## 📁 Project Structure

```
Email_Classifier_using_Naive_Bayes/
├── src/                          # Source code modules
│   ├── data_cleaner.py          # Data loading and cleaning
│   ├── eda_analyzer.py          # Exploratory data analysis
│   ├── text_preprocessor.py     # Text preprocessing and TF-IDF
│   ├── model_trainer.py         # Model training and validation
│   ├── model_evaluator.py       # Performance evaluation
│   ├── model_persistence.py     # Model save/load functionality
│   └── spam_predictor.py        # Real-time prediction
├── tests/                        # Test suite
│   ├── unit/                    # Unit tests
│   ├── property/                # Property-based tests
│   └── integration/             # Integration tests
├── templates/                    # HTML templates
│   └── index.html               # Web interface
├── data/                         # Dataset directory
├── models/                       # Trained models
├── app.py                        # Flask web application
├── main.py                       # Training pipeline script
├── requirements.txt              # Python dependencies
├── Procfile                      # Heroku deployment config
├── runtime.txt                   # Python version specification
└── README.md                     # This file
```

## 🔧 Usage

### Command Line Training

Train with custom parameters:

```bash
python main.py --data-path data/spam.csv --model-dir models
```

### Programmatic Usage

```python
from src.model_persistence import ModelPersistence
from src.text_preprocessor import TextPreprocessor
from src.spam_predictor import SpamPredictor

# Load trained model
model = ModelPersistence.load_model("models/spam_classifier_model.pkl")
vectorizer = ModelPersistence.load_vectorizer("models/tfidf_vectorizer.pkl")

# Create predictor
preprocessor = TextPreprocessor()
predictor = SpamPredictor(model, vectorizer, preprocessor)

# Classify a message
message = "Congratulations! You've won a free iPhone!"
prediction = predictor.predict(message)
confidence = predictor.predict_proba(message)

print(f"Prediction: {prediction}")
print(f"Confidence: {confidence[prediction]:.2%}")
```

### API Usage

Send POST request to the prediction endpoint:

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"message": "Win a free prize now!"}'
```

Response:

```json
{
  "message": "Win a free prize now!",
  "prediction": "spam",
  "confidence": {
    "ham": 0.0563,
    "spam": 0.9437
  },
  "processing_time": 0.0234
}
```

## 🧪 Testing

Run the complete test suite:

```bash
# Run all tests
pytest

# Run unit tests only
pytest tests/unit/

# Run property-based tests
pytest tests/property/

# Run integration tests
pytest tests/integration/

# Run with coverage
pytest --cov=src --cov-report=html
```

## 🛠️ ML Pipeline

The system follows a 7-step pipeline:

1. **Data Loading & Cleaning**: Load UCI dataset, remove duplicates, handle nulls, encode labels
2. **Exploratory Data Analysis**: Compute message statistics, analyze class distribution
3. **Text Preprocessing**: Lowercase, tokenize, remove stopwords, stem, TF-IDF vectorization
4. **Model Training**: Train Multinomial Naive Bayes with 5-fold cross-validation
5. **Model Evaluation**: Compute accuracy, confusion matrix, precision, recall, F1-scores
6. **Model Persistence**: Save trained model and vectorizer to disk
7. **Prediction Testing**: Validate pipeline with sample messages

## 📈 Model Details

### Algorithm
- **Classifier**: Multinomial Naive Bayes
- **Vectorization**: TF-IDF (Term Frequency-Inverse Document Frequency)
- **Preprocessing**: Lowercase → Tokenize → Remove special chars → Remove stopwords → Stem

### Training Configuration
- **Train/Test Split**: 80/20
- **Cross-Validation**: 5-fold
- **Random State**: 42 (for reproducibility)

### Text Processing
- **Tokenizer**: NLTK word_tokenize
- **Stemmer**: NLTK PorterStemmer
- **Stopwords**: NLTK English stopwords corpus

## 🌐 Deployment

### Heroku Deployment

1. **Login to Heroku**:
   ```bash
   heroku login
   ```

2. **Create Heroku app**:
   ```bash
   heroku create your-spam-classifier
   ```

3. **Deploy**:
   ```bash
   git push heroku main
   ```

4. **Open application**:
   ```bash
   heroku open
   ```

### Docker Deployment

```bash
# Build image
docker build -t spam-classifier .

# Run container
docker run -p 5000:5000 spam-classifier
```

## 📊 Dataset

This project uses the **UCI SMS Spam Collection Dataset**:

- **Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection)
- **Size**: 5,574 messages
- **Classes**: Ham (legitimate) and Spam
- **Distribution**: ~87% ham, ~13% spam
- **Format**: CSV with columns: label, message

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Darshil Shukla**

- GitHub: [@DarshilShukla](https://github.com/DarshilShukla)
- Repository: [Email_Classifier_using_Naive_Bayes](https://github.com/DarshilShukla/Email_Classifier_using_Naive_Bayes)

## 🙏 Acknowledgments

- UCI Machine Learning Repository for the SMS Spam Collection dataset
- scikit-learn team for the excellent ML library
- NLTK team for natural language processing tools
- Flask team for the web framework

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

⭐ If you find this project helpful, please consider giving it a star!
