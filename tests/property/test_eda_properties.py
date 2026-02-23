"""
Property-based tests for EDA module.

These tests verify universal properties that should hold across all valid inputs
using Hypothesis for property-based testing.
"""

import pandas as pd
import pytest
from hypothesis import given, strategies as st, settings
from src.eda_analyzer import EDAAnalyzer


class TestEDAProperties:
    """Property-based tests for EDAAnalyzer class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = EDAAnalyzer()
    
    @given(
        st.lists(
            st.text(min_size=0, max_size=500),
            min_size=1,
            max_size=50
        )
    )
    @settings(max_examples=100)
    def test_property_message_statistics_computation(self, messages):
        """
        Property 5: Message statistics computation
        
        For any message, verify char/word/sentence counts are correct.
        
        **Validates: Requirements 2.1, 2.2, 2.3**
        """
        # Create DataFrame with messages
        df = pd.DataFrame({'message': messages})
        
        # Compute all statistics
        df_with_stats = self.analyzer.compute_char_count(df)
        df_with_stats = self.analyzer.compute_word_count(df_with_stats)
        df_with_stats = self.analyzer.compute_sentence_count(df_with_stats)
        
        # Verify properties for each message
        for idx, row in df_with_stats.iterrows():
            message = row['message']
            char_count = row['char_count']
            word_count = row['word_count']
            sentence_count = row['sentence_count']
            
            # Property 1: Character count should equal length of message string
            assert char_count == len(message), \
                f"Character count mismatch for message: {repr(message)}"
            
            # Property 2: Word count should equal number of whitespace-separated tokens
            expected_word_count = len(message.split())
            assert word_count == expected_word_count, \
                f"Word count mismatch for message: {repr(message)}"
            
            # Property 3: Sentence count should be based on sentence-ending punctuation
            # Count sentence-ending punctuation: . ! ?
            import re
            sentence_endings = re.findall(r'[.!?]', message)
            expected_sentence_count = len(sentence_endings)
            
            # If no punctuation but text exists, should count as 1 sentence
            if expected_sentence_count == 0 and message.strip():
                expected_sentence_count = 1
            
            assert sentence_count == expected_sentence_count, \
                f"Sentence count mismatch for message: {repr(message)}"
            
            # Additional invariants
            # Property 4: Character count should be non-negative
            assert char_count >= 0
            
            # Property 5: Word count should be non-negative
            assert word_count >= 0
            
            # Property 6: Sentence count should be non-negative
            assert sentence_count >= 0
            
            # Property 7: If message is empty, all counts should be 0
            if not message:
                assert char_count == 0
                assert word_count == 0
                assert sentence_count == 0
            
            # Property 8: If message has text, char_count should be positive
            if message.strip():
                assert char_count > 0
