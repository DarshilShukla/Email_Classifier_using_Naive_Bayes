"""
Unit tests for the TextPreprocessor class.

Tests basic text preprocessing transformations including lowercase conversion,
tokenization, and special character removal.
"""

import pytest
from src.text_preprocessor import TextPreprocessor


class TestTextPreprocessor:
    """Test suite for TextPreprocessor class."""
    
    @pytest.fixture
    def preprocessor(self):
        """Create a TextPreprocessor instance for testing."""
        return TextPreprocessor()
    
    # Tests for lowercase()
    
    def test_lowercase_basic(self, preprocessor):
        """Test basic lowercase conversion."""
        text = "Hello World"
        result = preprocessor.lowercase(text)
        assert result == "hello world"
    
    def test_lowercase_mixed_case(self, preprocessor):
        """Test lowercase with mixed case text."""
        text = "ThIs Is A TeSt"
        result = preprocessor.lowercase(text)
        assert result == "this is a test"
    
    def test_lowercase_already_lowercase(self, preprocessor):
        """Test lowercase on already lowercase text."""
        text = "already lowercase"
        result = preprocessor.lowercase(text)
        assert result == "already lowercase"
    
    def test_lowercase_with_numbers(self, preprocessor):
        """Test lowercase preserves numbers."""
        text = "Test123"
        result = preprocessor.lowercase(text)
        assert result == "test123"
    
    def test_lowercase_with_special_chars(self, preprocessor):
        """Test lowercase preserves special characters."""
        text = "Hello! How are you?"
        result = preprocessor.lowercase(text)
        assert result == "hello! how are you?"
    
    def test_lowercase_empty_string(self, preprocessor):
        """Test lowercase on empty string."""
        text = ""
        result = preprocessor.lowercase(text)
        assert result == ""
    
    def test_lowercase_non_string_raises_error(self, preprocessor):
        """Test lowercase raises TypeError for non-string input."""
        with pytest.raises(TypeError):
            preprocessor.lowercase(123)
    
    # Tests for tokenize()
    
    def test_tokenize_basic(self, preprocessor):
        """Test basic tokenization."""
        text = "Hello world"
        result = preprocessor.tokenize(text)
        assert result == ["Hello", "world"]
    
    def test_tokenize_with_punctuation(self, preprocessor):
        """Test tokenization separates punctuation."""
        text = "Hello, world!"
        result = preprocessor.tokenize(text)
        assert "Hello" in result
        assert "world" in result
        assert "," in result or "!" in result  # NLTK may tokenize punctuation separately
    
    def test_tokenize_multiple_words(self, preprocessor):
        """Test tokenization with multiple words."""
        text = "This is a test message"
        result = preprocessor.tokenize(text)
        assert len(result) == 5
        assert "This" in result
        assert "test" in result
        assert "message" in result
    
    def test_tokenize_empty_string(self, preprocessor):
        """Test tokenization on empty string."""
        text = ""
        result = preprocessor.tokenize(text)
        assert result == []
    
    def test_tokenize_single_word(self, preprocessor):
        """Test tokenization on single word."""
        text = "Hello"
        result = preprocessor.tokenize(text)
        assert result == ["Hello"]
    
    def test_tokenize_non_string_raises_error(self, preprocessor):
        """Test tokenize raises TypeError for non-string input."""
        with pytest.raises(TypeError):
            preprocessor.tokenize(123)
    
    # Tests for remove_special_chars()
    
    def test_remove_special_chars_basic(self, preprocessor):
        """Test basic special character removal."""
        tokens = ["hello", "world"]
        result = preprocessor.remove_special_chars(tokens)
        assert result == ["hello", "world"]
    
    def test_remove_special_chars_with_punctuation(self, preprocessor):
        """Test removal of punctuation from tokens."""
        tokens = ["hello,", "world!"]
        result = preprocessor.remove_special_chars(tokens)
        assert result == ["hello", "world"]
    
    def test_remove_special_chars_only_punctuation(self, preprocessor):
        """Test tokens with only punctuation are removed."""
        tokens = ["hello", "!", "world", "?"]
        result = preprocessor.remove_special_chars(tokens)
        assert result == ["hello", "world"]
    
    def test_remove_special_chars_with_numbers(self, preprocessor):
        """Test alphanumeric tokens are preserved."""
        tokens = ["test123", "abc456"]
        result = preprocessor.remove_special_chars(tokens)
        assert result == ["test123", "abc456"]
    
    def test_remove_special_chars_mixed(self, preprocessor):
        """Test mixed alphanumeric and special characters."""
        tokens = ["hello!", "test123", "@#$", "world?"]
        result = preprocessor.remove_special_chars(tokens)
        assert result == ["hello", "test123", "world"]
    
    def test_remove_special_chars_empty_list(self, preprocessor):
        """Test empty token list."""
        tokens = []
        result = preprocessor.remove_special_chars(tokens)
        assert result == []
    
    def test_remove_special_chars_all_special(self, preprocessor):
        """Test list with only special character tokens."""
        tokens = ["!", "@", "#", "$"]
        result = preprocessor.remove_special_chars(tokens)
        assert result == []
    
    # Integration tests
    
    def test_lowercase_then_tokenize(self, preprocessor):
        """Test chaining lowercase and tokenize."""
        text = "Hello World"
        lowercased = preprocessor.lowercase(text)
        tokens = preprocessor.tokenize(lowercased)
        assert "hello" in tokens
        assert "world" in tokens
    
    def test_full_basic_pipeline(self, preprocessor):
        """Test complete basic preprocessing pipeline."""
        text = "Hello, World! This is a TEST."
        
        # Step 1: Lowercase
        lowercased = preprocessor.lowercase(text)
        assert lowercased == "hello, world! this is a test."
        
        # Step 2: Tokenize
        tokens = preprocessor.tokenize(lowercased)
        assert "hello" in tokens
        assert "test" in tokens
        
        # Step 3: Remove special chars
        cleaned = preprocessor.remove_special_chars(tokens)
        assert "hello" in cleaned
        assert "world" in cleaned
        assert "test" in cleaned
        # Punctuation should be removed
        assert "," not in cleaned
        assert "!" not in cleaned
        assert "." not in cleaned

    # Tests for remove_stopwords()
    
    def test_remove_stopwords_basic(self, preprocessor):
        """Test basic stopword removal."""
        tokens = ["this", "is", "a", "test"]
        result = preprocessor.remove_stopwords(tokens)
        # "this", "is", "a" are stopwords, "test" is not
        assert "test" in result
        assert "this" not in result
        assert "is" not in result
        assert "a" not in result
    
    def test_remove_stopwords_no_stopwords(self, preprocessor):
        """Test tokens with no stopwords."""
        tokens = ["hello", "world", "test"]
        result = preprocessor.remove_stopwords(tokens)
        assert result == ["hello", "world", "test"]
    
    def test_remove_stopwords_all_stopwords(self, preprocessor):
        """Test tokens with only stopwords."""
        tokens = ["the", "is", "at", "which"]
        result = preprocessor.remove_stopwords(tokens)
        assert result == []
    
    def test_remove_stopwords_case_insensitive(self, preprocessor):
        """Test stopword removal is case-insensitive."""
        tokens = ["The", "Test", "IS", "Good"]
        result = preprocessor.remove_stopwords(tokens)
        # "The" and "IS" are stopwords (case-insensitive)
        assert "Test" in result
        assert "Good" in result
        assert "The" not in result
        assert "IS" not in result
    
    def test_remove_stopwords_empty_list(self, preprocessor):
        """Test empty token list."""
        tokens = []
        result = preprocessor.remove_stopwords(tokens)
        assert result == []
    
    # Tests for stem_tokens()
    
    def test_stem_tokens_basic(self, preprocessor):
        """Test basic stemming."""
        tokens = ["running", "runs"]
        result = preprocessor.stem_tokens(tokens)
        # Both should stem to "run"
        assert all(token == "run" for token in result)
    
    def test_stem_tokens_variations(self, preprocessor):
        """Test stemming word variations."""
        tokens = ["jumping", "jumps", "jumped"]
        result = preprocessor.stem_tokens(tokens)
        # All should stem to "jump"
        assert all(token == "jump" for token in result)
    
    def test_stem_tokens_already_stemmed(self, preprocessor):
        """Test stemming already stemmed words."""
        tokens = ["test", "run", "jump"]
        result = preprocessor.stem_tokens(tokens)
        assert "test" in result
        assert "run" in result
        assert "jump" in result
    
    def test_stem_tokens_empty_list(self, preprocessor):
        """Test empty token list."""
        tokens = []
        result = preprocessor.stem_tokens(tokens)
        assert result == []
    
    # Tests for preprocess_text()
    
    def test_preprocess_text_basic(self, preprocessor):
        """Test complete preprocessing pipeline."""
        text = "Hello, World! This is a TEST."
        result = preprocessor.preprocess_text(text)
        
        # Should be lowercase, no punctuation, no stopwords, stemmed
        assert isinstance(result, str)
        assert result.islower()
        assert "," not in result
        assert "!" not in result
        assert "." not in result
    
    def test_preprocess_text_removes_stopwords(self, preprocessor):
        """Test that preprocessing removes stopwords."""
        text = "This is a test message"
        result = preprocessor.preprocess_text(text)
        
        # "this", "is", "a" are stopwords and should be removed
        assert "this" not in result.lower()
        assert "is" not in result.lower()
        # "test" and "message" should remain (stemmed)
        assert "test" in result or "messag" in result
    
    def test_preprocess_text_applies_stemming(self, preprocessor):
        """Test that preprocessing applies stemming."""
        text = "running jumping testing"
        result = preprocessor.preprocess_text(text)
        
        # Should be stemmed to root forms
        assert "run" in result
        assert "jump" in result
        assert "test" in result
    
    def test_preprocess_text_empty_string(self, preprocessor):
        """Test preprocessing empty string."""
        text = ""
        result = preprocessor.preprocess_text(text)
        assert result == ""
    
    def test_preprocess_text_only_stopwords(self, preprocessor):
        """Test preprocessing text with only stopwords."""
        text = "the is at which"
        result = preprocessor.preprocess_text(text)
        # All stopwords should be removed, result should be empty
        assert result == ""
    
    def test_preprocess_text_only_special_chars(self, preprocessor):
        """Test preprocessing text with only special characters."""
        text = "!@#$%^&*()"
        result = preprocessor.preprocess_text(text)
        # All special chars should be removed
        assert result == ""
    
    def test_preprocess_text_non_string_raises_error(self, preprocessor):
        """Test preprocess_text raises TypeError for non-string input."""
        with pytest.raises(TypeError):
            preprocessor.preprocess_text(123)
    
    def test_preprocess_text_returns_string(self, preprocessor):
        """Test that preprocess_text returns a string."""
        text = "Hello world"
        result = preprocessor.preprocess_text(text)
        assert isinstance(result, str)
    
    # Integration test for complete pipeline
    
    def test_complete_preprocessing_pipeline(self, preprocessor):
        """Test the complete preprocessing pipeline with a realistic message."""
        text = "Hello! Are you coming to the party? I'm so excited!!!"
        result = preprocessor.preprocess_text(text)
        
        # Verify transformations
        assert isinstance(result, str)
        assert result.islower()  # Should be lowercase
        assert "!" not in result  # No special chars
        assert "?" not in result
        # Common stopwords should be removed
        words = result.split()
        assert "the" not in words
        # Content words should remain (possibly stemmed)
        assert any(word.startswith("com") for word in words)  # "coming" -> "come"
        assert any(word.startswith("parti") for word in words)  # "party" -> "parti"
        assert any(word.startswith("excit") for word in words)  # "excited" -> "excit"

    # Tests for TF-IDF vectorization
    
    def test_fit_vectorizer_basic(self, preprocessor):
        """Test fitting the TF-IDF vectorizer."""
        messages = ["hello world", "test message", "another test"]
        preprocessor.fit_vectorizer(messages)
        
        assert preprocessor.vectorizer is not None
        assert hasattr(preprocessor.vectorizer, 'vocabulary_')
    
    def test_fit_vectorizer_empty_list_raises_error(self, preprocessor):
        """Test fit_vectorizer raises ValueError for empty list."""
        with pytest.raises(ValueError, match="Cannot fit vectorizer on empty message list"):
            preprocessor.fit_vectorizer([])
    
    def test_transform_basic(self, preprocessor):
        """Test transforming messages to TF-IDF features."""
        messages = ["hello world", "test message"]
        preprocessor.fit_vectorizer(messages)
        
        result = preprocessor.transform(messages)
        
        # Should return sparse matrix
        assert result.shape[0] == 2  # 2 messages
        assert result.shape[1] > 0  # Should have features
        # All values should be non-negative (TF-IDF property)
        assert (result.data >= 0).all()
    
    def test_transform_without_fit_raises_error(self, preprocessor):
        """Test transform raises ValueError if vectorizer not fitted."""
        messages = ["hello world"]
        
        with pytest.raises(ValueError, match="Vectorizer must be fitted before transform"):
            preprocessor.transform(messages)
    
    def test_transform_new_messages(self, preprocessor):
        """Test transforming new messages after fitting."""
        train_messages = ["hello world", "test message"]
        test_messages = ["hello test"]
        
        preprocessor.fit_vectorizer(train_messages)
        result = preprocessor.transform(test_messages)
        
        assert result.shape[0] == 1  # 1 message
        assert result.shape[1] > 0  # Should have features
    
    def test_fit_transform_basic(self, preprocessor):
        """Test fit_transform combines fit and transform."""
        messages = ["hello world", "test message", "another test"]
        
        result = preprocessor.fit_transform(messages)
        
        # Should return sparse matrix
        assert result.shape[0] == 3  # 3 messages
        assert result.shape[1] > 0  # Should have features
        # Vectorizer should be fitted
        assert preprocessor.vectorizer is not None
        assert hasattr(preprocessor.vectorizer, 'vocabulary_')
        # All values should be non-negative
        assert (result.data >= 0).all()
    
    def test_fit_transform_empty_list_raises_error(self, preprocessor):
        """Test fit_transform raises ValueError for empty list."""
        with pytest.raises(ValueError, match="Cannot fit_transform on empty message list"):
            preprocessor.fit_transform([])
    
    def test_fit_transform_single_message(self, preprocessor):
        """Test fit_transform with single message."""
        messages = ["hello world"]
        
        result = preprocessor.fit_transform(messages)
        
        assert result.shape[0] == 1
        assert result.shape[1] > 0
    
    def test_vectorization_produces_sparse_matrix(self, preprocessor):
        """Test that vectorization produces sparse matrix."""
        from scipy.sparse import issparse
        
        messages = ["hello world", "test message"]
        result = preprocessor.fit_transform(messages)
        
        assert issparse(result)
    
    def test_vectorization_shape_matches_input(self, preprocessor):
        """Test that output shape matches number of input messages."""
        messages = ["msg1", "msg2", "msg3", "msg4", "msg5"]
        result = preprocessor.fit_transform(messages)
        
        assert result.shape[0] == len(messages)
    
    # Integration test for preprocessing + vectorization
    
    def test_preprocess_then_vectorize(self, preprocessor):
        """Test complete pipeline: preprocess text then vectorize."""
        raw_messages = [
            "Hello! This is a TEST message.",
            "Another test MESSAGE here!",
            "Final message for testing."
        ]
        
        # Preprocess all messages
        preprocessed = [preprocessor.preprocess_text(msg) for msg in raw_messages]
        
        # Vectorize preprocessed messages
        result = preprocessor.fit_transform(preprocessed)
        
        # Verify output
        assert result.shape[0] == 3
        assert result.shape[1] > 0
        assert (result.data >= 0).all()
    
    def test_empty_message_handling(self, preprocessor):
        """Test handling of empty messages after preprocessing."""
        messages = ["hello world", "", "test"]
        
        # fit_transform should handle empty strings
        result = preprocessor.fit_transform(messages)
        
        assert result.shape[0] == 3
    
    def test_messages_with_only_special_chars(self, preprocessor):
        """Test messages that become empty after preprocessing."""
        raw_messages = ["Hello world", "!@#$%", "Test message"]
        preprocessed = [preprocessor.preprocess_text(msg) for msg in raw_messages]
        
        # Should handle empty preprocessed messages
        result = preprocessor.fit_transform(preprocessed)
        
        assert result.shape[0] == 3
