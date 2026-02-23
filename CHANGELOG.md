# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-02-23

### Added
- Initial release of SMS Spam Classifier
- Complete ML pipeline from data cleaning to model deployment
- Multinomial Naive Bayes classifier with 96%+ accuracy
- Flask web interface for real-time spam classification
- Comprehensive test suite (unit, property-based, integration tests)
- Data cleaning module with duplicate removal and label encoding
- EDA module with message statistics and visualizations
- Text preprocessing with NLTK (tokenization, stemming, stopword removal)
- TF-IDF vectorization for feature extraction
- Model training with cross-validation
- Model evaluation with detailed metrics
- Model persistence for save/load functionality
- Real-time prediction API with confidence scores
- Deployment configuration for Heroku
- Complete documentation and README
- GitHub Actions CI/CD workflow
- Contributing guidelines
- MIT License

### Features
- **Data Processing**: Handles UCI SMS Spam Collection dataset
- **Text Preprocessing**: Lowercase, tokenize, remove special chars, remove stopwords, stem
- **Model Training**: 80/20 train-test split with 5-fold cross-validation
- **Evaluation Metrics**: Accuracy, precision, recall, F1-score, confusion matrix
- **Web Interface**: User-friendly form for message classification
- **API Endpoint**: JSON API for programmatic access
- **Testing**: 90%+ code coverage with comprehensive test suite

### Performance
- Test Accuracy: 96.32%
- Ham Precision: 95.90%
- Spam Precision: 100%
- Ham Recall: 100%
- Spam Recall: 73.79%
- Processing Time: <50ms per message

### Documentation
- Comprehensive README with quick start guide
- API documentation with examples
- Deployment instructions for Heroku
- Contributing guidelines
- Code comments and docstrings

## [Unreleased]

### Planned Features
- Support for additional languages
- Real-time model retraining
- A/B testing framework
- Model versioning system
- Enhanced web UI with message history
- Batch prediction API
- Docker containerization
- Kubernetes deployment configuration
- Performance monitoring dashboard
- Model explainability features
