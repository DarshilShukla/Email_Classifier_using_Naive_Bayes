"""
Property-based tests for text preprocessing module.

These tests verify universal properties that should hold across all valid inputs
using Hypothesis for property-based testing.
"""

import pytest
from hypothesis import given, strategies as st, settings
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from src.text_preprocessor import TextPreprocessor


class TestPreprocessingProperties:
    """Property-based tests for TextPreprocessor class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.preprocessor = TextPreprocessor()
        # Ensure NLTK data is available
        try:
            self.stop_words = set(stopwords.words('english'))
        except LookupError:
            import nltk
            nltk.download('stopwords', quiet=True)
            self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
    
    @given(st.text(min_size=0, max_size=500))
    @settings(max_examples=100)
    def test_property_lowercase_transformation(self, text):
        """
        Property 6: Lowercase transformation
        
        For any text, verify no uppercase remains.
        
        **Validates: Requirements 3.1**
        """
        # Apply lowercase transformation
        result = self.preprocessor.lowercase(text)
        
        # Property: No uppercase characters should remain
        assert result == result.lower(), f"Uppercase characters found in: {result}"
        
        # Property: Result should be a string
        assert isinstance(result, str)
        
        # Property: Length should be preserved
        assert len(result) == len(text)
    
    @given(st.text(min_size=0, max_size=500))
    @settings(max_examples=100)
    def test_property_tokenization_produces_list(self, text):
        """
        Property 7: Tokenization produces list
        
        For any text, verify output is list of tokens.
        
        **Validates: Requirements 3.2**
        """
        # Apply tokenization
        result = self.preprocessor.tokenize(text)
        
        # Property: Result should be a list
        assert isinstance(result, list)
        
        # Property: All elements should be strings
        assert all(isinstance(token, str) for token in result)
        
        # Property: For non-empty text, tokens should be non-empty strings
        for token in result:
            assert len(token) > 0, f"Empty token found in result: {result}"
    
    @given(
        st.lists(
            st.text(
                alphabet=st.characters(
                    whitelist_categories=('Lu', 'Ll', 'Nd', 'Po', 'Ps', 'Pe', 'Pd', 'Pc')
                ),
                min_size=1,
                max_size=20
            ),
            min_size=1,
            max_size=50
        )
    )
    @settings(max_examples=100)
    def test_property_special_character_removal(self, tokens):
        """
        Property 8: Special character removal
        
        For any tokens, verify only alphanumeric remain.
        
        **Validates: Requirements 3.3**
        """
        # Apply special character removal
        result = self.preprocessor.remove_special_chars(tokens)
        
        # Property: Result should be a list
        assert isinstance(result, list)
        
        # Property: All tokens should contain only alphanumeric characters
        for token in result:
            assert token.isalnum(), f"Non-alphanumeric token found: '{token}'"
        
        # Property: All tokens should be non-empty
        for token in result:
            assert len(token) > 0, f"Empty token found in result"
    
    @given(
        st.lists(
            st.sampled_from([
                'hello', 'world', 'test', 'message', 'python', 'code',
                'the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 'or'
            ]),
            min_size=1,
            max_size=50
        )
    )
    @settings(max_examples=100)
    def test_property_stopword_removal(self, tokens):
        """
        Property 9: Stopword removal
        
        For any tokens, verify no NLTK stopwords remain.
        
        **Validates: Requirements 3.4**
        """
        # Apply stopword removal
        result = self.preprocessor.remove_stopwords(tokens)
        
        # Property: Result should be a list
        assert isinstance(result, list)
        
        # Property: No token should be in the stopwords list (case-insensitive)
        for token in result:
            assert token.lower() not in self.stop_words, \
                f"Stopword '{token}' found in result: {result}"
        
        # Property: Result should be a subset of input tokens
        assert set(result).issubset(set(tokens))
    
    @given(
        st.lists(
            st.sampled_from([
                'running', 'runs', 'ran',
                'jumping', 'jumps', 'jumped',
                'testing', 'tests', 'tested',
                'walking', 'walks', 'walked'
            ]),
            min_size=1,
            max_size=20
        )
    )
    @settings(max_examples=100)
    def test_property_stemming_application(self, tokens):
        """
        Property 10: Stemming application
        
        For any tokens, verify known variations reduce to same root.
        
        **Validates: Requirements 3.5**
        """
        # Apply stemming
        result = self.preprocessor.stem_tokens(tokens)
        
        # Property: Result should be a list
        assert isinstance(result, list)
        
        # Property: Result should have same length as input
        assert len(result) == len(tokens)
        
        # Property: All elements should be strings
        assert all(isinstance(token, str) for token in result)
        
        # Property: Known variations should reduce to same root
        # Test specific known variations
        if 'running' in tokens and 'runs' in tokens:
            running_idx = tokens.index('running')
            runs_idx = tokens.index('runs')
            assert result[running_idx] == result[runs_idx], \
                f"'running' and 'runs' should stem to same root"
        
        if 'jumping' in tokens and 'jumps' in tokens:
            jumping_idx = tokens.index('jumping')
            jumps_idx = tokens.index('jumps')
            assert result[jumping_idx] == result[jumps_idx], \
                f"'jumping' and 'jumps' should stem to same root"
        
        # Property: Stemmed tokens should not be longer than original
        for original, stemmed in zip(tokens, result):
            assert len(stemmed) <= len(original), \
                f"Stemmed token '{stemmed}' is longer than original '{original}'"
    
    @given(
        st.lists(
            st.from_regex(r'[a-zA-Z ]{5,50}', fullmatch=True),
            min_size=2,
            max_size=20
        )
    )
    @settings(max_examples=100)
    def test_property_tfidf_vectorization_output(self, messages):
        """
        Property 11: TF-IDF vectorization output
        
        For any messages, verify sparse matrix output with correct shape.
        
        **Validates: Requirements 3.6**
        """
        from scipy.sparse import issparse
        
        # Filter out completely empty messages
        valid_messages = [msg for msg in messages if len(msg.strip()) > 0]
        
        # Ensure we have at least some valid messages
        if not valid_messages:
            valid_messages = ["hello world", "test message"]
        
        try:
            # Apply TF-IDF vectorization
            result = self.preprocessor.fit_transform(valid_messages)
            
            # Property: Result should be a sparse matrix
            assert issparse(result), "Result should be a sparse matrix"
            
            # Property: Shape should be (n_messages, n_features)
            assert result.shape[0] == len(valid_messages), \
                f"Expected {len(valid_messages)} rows, got {result.shape[0]}"
            assert result.shape[1] > 0, "Should have at least one feature"
            
            # Property: All values should be non-negative (TF-IDF property)
            assert (result.data >= 0).all(), "TF-IDF values should be non-negative"
        except ValueError as e:
            # If we get empty vocabulary error, it's because the input was all stopwords
            # This is acceptable behavior - just ensure the error message is correct
            if "empty vocabulary" in str(e):
                pass  # This is expected for edge cases
            else:
                raise
    
    @given(st.text(min_size=5, max_size=500, alphabet=st.characters(
        whitelist_categories=('Lu', 'Ll', 'Nd', 'Zs')
    )))
    @settings(max_examples=100)
    def test_property_end_to_end_preprocessing_pipeline(self, message):
        """
        Property 12: End-to-end preprocessing pipeline
        
        For any message, verify complete pipeline produces numerical vector.
        
        **Validates: Requirements 3.7**
        """
        from scipy.sparse import issparse
        
        # Apply complete preprocessing pipeline
        preprocessed = self.preprocessor.preprocess_text(message)
        
        # Property: Preprocessed text should be a string
        assert isinstance(preprocessed, str)
        
        # Property: Preprocessed text should be lowercase
        assert preprocessed == preprocessed.lower()
        
        # Skip vectorization if preprocessing resulted in empty string
        if len(preprocessed.strip()) == 0:
            return
        
        try:
            # Now vectorize the preprocessed text
            vectorized = self.preprocessor.fit_transform([preprocessed])
            
            # Property: Vectorization should produce sparse matrix
            assert issparse(vectorized)
            
            # Property: Shape should be (1, n_features) for single message
            assert vectorized.shape[0] == 1
            assert vectorized.shape[1] > 0
            
            # Property: All values should be non-negative
            if vectorized.data.size > 0:
                assert (vectorized.data >= 0).all()
            
            # Property: Result is numerical (can be used for ML)
            assert vectorized.dtype in ['float64', 'float32', 'float']
        except ValueError as e:
            # If we get empty vocabulary error, it's because preprocessing removed everything
            # This is acceptable behavior for edge cases
            if "empty vocabulary" in str(e):
                pass  # This is expected when all content is filtered out
            else:
                raise
