"""
Text preprocessing module for SMS spam classification.

This module provides the TextPreprocessor class for transforming
raw text messages into features suitable for machine learning.
"""

import re
from typing import List
import nltk
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix


class TextPreprocessor:
    """
    Handles text preprocessing for SMS messages.
    
    The TextPreprocessor class provides methods to transform raw text
    through various preprocessing steps including lowercase conversion,
    tokenization, special character removal, stopword removal, stemming,
    and TF-IDF vectorization.
    """
    
    def __init__(self):
        """
        Initialize the TextPreprocessor.
        
        Ensures required NLTK data is available.
        """
        # Ensure NLTK data is downloaded
        try:
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            nltk.download('punkt_tab', quiet=True)
        
        # Initialize TF-IDF vectorizer
        self.vectorizer = None
    
    def lowercase(self, text: str) -> str:
        """
        Convert text to lowercase.
        
        Args:
            text: Input text string
            
        Returns:
            Text converted to lowercase
            
        Raises:
            TypeError: If input is not a string
        """
        if not isinstance(text, str):
            raise TypeError(f"Input must be a string, got {type(text).__name__}")
        
        return text.lower()
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into individual words using NLTK word_tokenize.
        
        Args:
            text: Input text string
            
        Returns:
            List of token strings
            
        Raises:
            TypeError: If input is not a string
        """
        if not isinstance(text, str):
            raise TypeError(f"Input must be a string, got {type(text).__name__}")
        
        tokens = word_tokenize(text)
        return tokens
    
    def remove_special_chars(self, tokens: List[str]) -> List[str]:
        """
        Remove special characters from tokens, keeping only alphanumeric characters.
        
        Filters out tokens that become empty after removing special characters.
        
        Args:
            tokens: List of token strings
            
        Returns:
            List of tokens containing only alphanumeric characters
        """
        # Keep only alphanumeric characters in each token
        cleaned_tokens = []
        for token in tokens:
            # Remove all non-alphanumeric characters
            cleaned_token = re.sub(r'[^a-zA-Z0-9]', '', token)
            # Only keep non-empty tokens
            if cleaned_token:
                cleaned_tokens.append(cleaned_token)
        
        return cleaned_tokens

    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """
        Remove stopwords from tokens using NLTK stopwords list.
        
        Args:
            tokens: List of token strings
            
        Returns:
            List of tokens with stopwords removed
        """
        # Ensure stopwords are downloaded
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords', quiet=True)
        
        from nltk.corpus import stopwords
        stop_words = set(stopwords.words('english'))
        
        # Filter out stopwords (case-insensitive comparison)
        filtered_tokens = [token for token in tokens if token.lower() not in stop_words]
        
        return filtered_tokens
    
    def stem_tokens(self, tokens: List[str]) -> List[str]:
        """
        Apply stemming to reduce words to their root form using NLTK PorterStemmer.
        
        Args:
            tokens: List of token strings
            
        Returns:
            List of stemmed tokens
        """
        from nltk.stem import PorterStemmer
        stemmer = PorterStemmer()
        
        stemmed_tokens = [stemmer.stem(token) for token in tokens]
        
        return stemmed_tokens
    
    def preprocess_text(self, text: str) -> str:
        """
        Apply complete preprocessing pipeline to text.
        
        Chains all transformations: lowercase → tokenize → remove special chars 
        → remove stopwords → stem → join back to string.
        
        Args:
            text: Input text string
            
        Returns:
            Preprocessed text as a single string with tokens joined by spaces
            
        Raises:
            TypeError: If input is not a string
        """
        if not isinstance(text, str):
            raise TypeError(f"Input must be a string, got {type(text).__name__}")
        
        # Apply preprocessing pipeline
        text = self.lowercase(text)
        tokens = self.tokenize(text)
        tokens = self.remove_special_chars(tokens)
        tokens = self.remove_stopwords(tokens)
        tokens = self.stem_tokens(tokens)
        
        # Join tokens back into a single string
        return ' '.join(tokens)
    
    def fit_vectorizer(self, messages: List[str]) -> None:
        """
        Fit the TF-IDF vectorizer on a list of messages.
        
        Args:
            messages: List of preprocessed text messages
            
        Raises:
            ValueError: If messages list is empty
        """
        if not messages:
            raise ValueError("Cannot fit vectorizer on empty message list")
        
        self.vectorizer = TfidfVectorizer()
        self.vectorizer.fit(messages)
    
    def transform(self, messages: List[str]) -> csr_matrix:
        """
        Transform messages to TF-IDF feature vectors.
        
        Args:
            messages: List of preprocessed text messages
            
        Returns:
            Sparse matrix of TF-IDF features with shape (n_messages, n_features)
            
        Raises:
            ValueError: If vectorizer has not been fitted yet
        """
        if self.vectorizer is None:
            raise ValueError("Vectorizer must be fitted before transform. Call fit_vectorizer() first.")
        
        return self.vectorizer.transform(messages)
    
    def fit_transform(self, messages: List[str]) -> csr_matrix:
        """
        Fit the vectorizer and transform messages in one step.
        
        This is a convenience method for training data that combines
        fit_vectorizer() and transform().
        
        Args:
            messages: List of preprocessed text messages
            
        Returns:
            Sparse matrix of TF-IDF features with shape (n_messages, n_features)
            
        Raises:
            ValueError: If messages list is empty
        """
        if not messages:
            raise ValueError("Cannot fit_transform on empty message list")
        
        self.vectorizer = TfidfVectorizer()
        return self.vectorizer.fit_transform(messages)
