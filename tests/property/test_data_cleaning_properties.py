"""
Property-based tests for data cleaning module.

These tests verify universal properties that should hold across all valid inputs
using Hypothesis for property-based testing.
"""

import pandas as pd
import pytest
from hypothesis import given, strategies as st, settings
from src.data_cleaner import DataCleaner


class TestDataCleaningProperties:
    """Property-based tests for DataCleaner class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.cleaner = DataCleaner()
    
    @given(
        st.lists(
            st.tuples(
                st.sampled_from(['ham', 'spam']),
                st.text(min_size=1, max_size=100),
                st.text(min_size=0, max_size=50)
            ),
            min_size=1,
            max_size=50
        )
    )
    @settings(max_examples=100)
    def test_property_unnamed_column_removal(self, data):
        """
        Property 1: Unnamed column removal
        
        For any DataFrame with unnamed columns, verify they are removed.
        
        **Validates: Requirements 1.3**
        """
        # Create DataFrame with unnamed columns
        labels, messages, unnamed_data = zip(*data)
        df = pd.DataFrame({
            'label': labels,
            'message': messages,
            'Unnamed: 2': unnamed_data,
            'Unnamed: 3': [f"extra_{i}" for i in range(len(data))]
        })
        
        # Apply drop_unnamed_columns
        df_cleaned = self.cleaner.drop_unnamed_columns(df)
        
        # Property: No columns should contain 'Unnamed' in their name
        unnamed_cols = [col for col in df_cleaned.columns if 'Unnamed' in str(col)]
        assert len(unnamed_cols) == 0, f"Found unnamed columns: {unnamed_cols}"
        
        # Property: Original non-unnamed columns should be preserved
        assert 'label' in df_cleaned.columns
        assert 'message' in df_cleaned.columns
    
    @given(
        st.lists(
            st.tuples(
                st.sampled_from(['ham', 'spam']),
                st.text(min_size=1, max_size=100)
            ),
            min_size=2,
            max_size=50
        )
    )
    @settings(max_examples=100)
    def test_property_duplicate_message_removal(self, data):
        """
        Property 2: Duplicate message removal
        
        For any DataFrame with duplicates, verify removal preserves one instance.
        
        **Validates: Requirements 1.4**
        """
        # Create DataFrame with intentional duplicates
        labels, messages = zip(*data)
        
        # Add a duplicate of the first message
        labels_with_dup = list(labels) + [labels[0]]
        messages_with_dup = list(messages) + [messages[0]]
        
        df = pd.DataFrame({
            'label': labels_with_dup,
            'message': messages_with_dup
        })
        
        original_count = len(df)
        unique_messages = df['message'].nunique()
        
        # Apply remove_duplicates
        df_no_dups = self.cleaner.remove_duplicates(df)
        
        # Property: Number of rows should equal number of unique messages
        assert len(df_no_dups) == unique_messages
        
        # Property: All messages in result should be unique
        assert df_no_dups['message'].nunique() == len(df_no_dups)
        
        # Property: At least one instance of each unique message is preserved
        original_unique_messages = set(df['message'].unique())
        result_messages = set(df_no_dups['message'].unique())
        assert original_unique_messages == result_messages
        
        # Property: Result should have fewer or equal rows than original
        assert len(df_no_dups) <= original_count
    
    @given(
        st.lists(
            st.tuples(
                st.one_of(st.sampled_from(['ham', 'spam']), st.none()),
                st.one_of(st.text(min_size=1, max_size=100), st.none())
            ),
            min_size=1,
            max_size=50
        )
    )
    @settings(max_examples=100)
    def test_property_null_value_handling(self, data):
        """
        Property 3: Null value handling
        
        For any DataFrame with nulls, verify rows are dropped.
        
        **Validates: Requirements 1.5**
        """
        # Create DataFrame with potential null values
        labels, messages = zip(*data)
        df = pd.DataFrame({
            'label': labels,
            'message': messages
        })
        
        # Count rows with nulls
        rows_with_nulls = df.isnull().any(axis=1).sum()
        expected_clean_rows = len(df) - rows_with_nulls
        
        # Apply drop_nulls
        df_no_nulls = self.cleaner.drop_nulls(df)
        
        # Property: Result should have no null values
        assert df_no_nulls.isnull().sum().sum() == 0
        
        # Property: Number of rows should equal original minus rows with nulls
        assert len(df_no_nulls) == expected_clean_rows
        
        # Property: All remaining rows should be complete (no nulls)
        for idx, row in df_no_nulls.iterrows():
            assert not row.isnull().any()
    
    @given(
        st.lists(
            st.tuples(
                st.sampled_from(['ham', 'spam']),
                st.text(min_size=1, max_size=100)
            ),
            min_size=1,
            max_size=50
        )
    )
    @settings(max_examples=100)
    def test_property_label_encoding_consistency(self, data):
        """
        Property 4: Label encoding consistency
        
        For any DataFrame with ham/spam labels, verify encoding is correct.
        
        **Validates: Requirements 1.6**
        """
        # Create DataFrame with ham/spam labels
        labels, messages = zip(*data)
        df = pd.DataFrame({
            'label': labels,
            'message': messages
        })
        
        # Count original ham and spam
        original_ham_count = (df['label'] == 'ham').sum()
        original_spam_count = (df['label'] == 'spam').sum()
        
        # Apply encode_labels
        df_encoded = self.cleaner.encode_labels(df)
        
        # Property: 'ham' should be encoded as 0
        ham_rows = df[df['label'] == 'ham'].index
        for idx in ham_rows:
            if idx in df_encoded.index:
                assert df_encoded.loc[idx, 'label'] == 0
        
        # Property: 'spam' should be encoded as 1
        spam_rows = df[df['label'] == 'spam'].index
        for idx in spam_rows:
            if idx in df_encoded.index:
                assert df_encoded.loc[idx, 'label'] == 1
        
        # Property: Count of 0s should equal original ham count
        assert (df_encoded['label'] == 0).sum() == original_ham_count
        
        # Property: Count of 1s should equal original spam count
        assert (df_encoded['label'] == 1).sum() == original_spam_count
        
        # Property: Only 0 and 1 should be present in encoded labels
        assert set(df_encoded['label'].unique()).issubset({0, 1})
        
        # Property: Label column should be numeric type
        assert df_encoded['label'].dtype in ['int64', 'int32', 'int']
