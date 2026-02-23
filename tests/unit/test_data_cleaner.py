"""
Unit tests for the DataCleaner class.
"""

import pytest
import pandas as pd
import tempfile
import os
from src.data_cleaner import DataCleaner


class TestDataCleaner:
    """Test suite for DataCleaner class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.cleaner = DataCleaner()
    
    def test_load_data_success(self):
        """Test loading a valid CSV file."""
        # Create a temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            f.write('v1,v2,Unnamed: 2\n')
            f.write('ham,Hello world,extra\n')
            f.write('spam,Buy now!,extra\n')
            temp_file = f.name
        
        try:
            df = self.cleaner.load_data(temp_file)
            
            assert isinstance(df, pd.DataFrame)
            assert 'v1' in df.columns
            assert 'v2' in df.columns
            assert len(df) == 2
        finally:
            os.unlink(temp_file)
    
    def test_load_data_file_not_found(self):
        """Test loading a non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError) as exc_info:
            self.cleaner.load_data('nonexistent_file.csv')
        
        assert 'Dataset file not found' in str(exc_info.value)
    
    def test_load_data_missing_required_columns(self):
        """Test loading a CSV without required columns raises ValueError."""
        # Create a CSV without v1 and v2 columns
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            f.write('col1,col2\n')
            f.write('value1,value2\n')
            temp_file = f.name
        
        try:
            with pytest.raises(ValueError) as exc_info:
                self.cleaner.load_data(temp_file)
            
            assert "must contain 'v1' and 'v2' columns" in str(exc_info.value)
        finally:
            os.unlink(temp_file)
    
    def test_rename_columns(self):
        """Test renaming v1 to label and v2 to message."""
        df = pd.DataFrame({
            'v1': ['ham', 'spam'],
            'v2': ['Hello', 'Buy now!']
        })
        
        df_renamed = self.cleaner.rename_columns(df)
        
        assert 'label' in df_renamed.columns
        assert 'message' in df_renamed.columns
        assert 'v1' not in df_renamed.columns
        assert 'v2' not in df_renamed.columns
        assert list(df_renamed['label']) == ['ham', 'spam']
        assert list(df_renamed['message']) == ['Hello', 'Buy now!']
    
    def test_drop_unnamed_columns(self):
        """Test dropping columns with 'Unnamed' in their name."""
        df = pd.DataFrame({
            'label': ['ham', 'spam'],
            'message': ['Hello', 'Buy now!'],
            'Unnamed: 2': ['extra1', 'extra2'],
            'Unnamed: 3': ['extra3', 'extra4']
        })
        
        df_cleaned = self.cleaner.drop_unnamed_columns(df)
        
        assert 'label' in df_cleaned.columns
        assert 'message' in df_cleaned.columns
        assert 'Unnamed: 2' not in df_cleaned.columns
        assert 'Unnamed: 3' not in df_cleaned.columns
        assert len(df_cleaned.columns) == 2
    
    def test_drop_unnamed_columns_no_unnamed(self):
        """Test dropping unnamed columns when none exist."""
        df = pd.DataFrame({
            'label': ['ham', 'spam'],
            'message': ['Hello', 'Buy now!']
        })
        
        df_cleaned = self.cleaner.drop_unnamed_columns(df)
        
        assert list(df_cleaned.columns) == ['label', 'message']
        assert len(df_cleaned) == 2
    
    def test_integration_load_rename_drop(self):
        """Test the integration of load, rename, and drop operations."""
        # Create a temporary CSV file with unnamed columns
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            f.write('v1,v2,Unnamed: 2,Unnamed: 3\n')
            f.write('ham,Hello world,extra1,extra2\n')
            f.write('spam,Buy now!,extra3,extra4\n')
            temp_file = f.name
        
        try:
            # Load data
            df = self.cleaner.load_data(temp_file)
            
            # Rename columns
            df = self.cleaner.rename_columns(df)
            
            # Drop unnamed columns
            df = self.cleaner.drop_unnamed_columns(df)
            
            # Verify final state
            assert list(df.columns) == ['label', 'message']
            assert len(df) == 2
            assert list(df['label']) == ['ham', 'spam']
            assert list(df['message']) == ['Hello world', 'Buy now!']
        finally:
            os.unlink(temp_file)

    def test_remove_duplicates(self):
        """Test removing duplicate messages."""
        df = pd.DataFrame({
            'label': ['ham', 'spam', 'ham', 'spam'],
            'message': ['Hello', 'Buy now!', 'Hello', 'Win prize!']
        })
        
        df_no_dups = self.cleaner.remove_duplicates(df)
        
        assert len(df_no_dups) == 3
        assert list(df_no_dups['message']) == ['Hello', 'Buy now!', 'Win prize!']
    
    def test_remove_duplicates_no_duplicates(self):
        """Test removing duplicates when none exist."""
        df = pd.DataFrame({
            'label': ['ham', 'spam'],
            'message': ['Hello', 'Buy now!']
        })
        
        df_no_dups = self.cleaner.remove_duplicates(df)
        
        assert len(df_no_dups) == 2
        assert list(df_no_dups['message']) == ['Hello', 'Buy now!']
    
    def test_drop_nulls(self):
        """Test dropping rows with null values."""
        df = pd.DataFrame({
            'label': ['ham', 'spam', None, 'ham'],
            'message': ['Hello', None, 'Test', 'World']
        })
        
        df_no_nulls = self.cleaner.drop_nulls(df)
        
        assert len(df_no_nulls) == 2
        assert list(df_no_nulls['label']) == ['ham', 'ham']
        assert list(df_no_nulls['message']) == ['Hello', 'World']
    
    def test_drop_nulls_no_nulls(self):
        """Test dropping nulls when none exist."""
        df = pd.DataFrame({
            'label': ['ham', 'spam'],
            'message': ['Hello', 'Buy now!']
        })
        
        df_no_nulls = self.cleaner.drop_nulls(df)
        
        assert len(df_no_nulls) == 2
    
    def test_encode_labels(self):
        """Test encoding ham→0 and spam→1."""
        df = pd.DataFrame({
            'label': ['ham', 'spam', 'ham', 'spam'],
            'message': ['Hello', 'Buy now!', 'Hi there', 'Win prize!']
        })
        
        df_encoded = self.cleaner.encode_labels(df)
        
        assert list(df_encoded['label']) == [0, 1, 0, 1]
        assert df_encoded['label'].dtype in [int, 'int64']
    
    def test_encode_labels_invalid_label(self):
        """Test encoding with invalid labels raises ValueError."""
        df = pd.DataFrame({
            'label': ['ham', 'spam', 'invalid'],
            'message': ['Hello', 'Buy now!', 'Test']
        })
        
        with pytest.raises(ValueError) as exc_info:
            self.cleaner.encode_labels(df)
        
        assert 'Invalid labels found' in str(exc_info.value)
    
    def test_encode_labels_missing_column(self):
        """Test encoding without label column raises ValueError."""
        df = pd.DataFrame({
            'message': ['Hello', 'Buy now!']
        })
        
        with pytest.raises(ValueError) as exc_info:
            self.cleaner.encode_labels(df)
        
        assert "must contain 'label' column" in str(exc_info.value)
    
    def test_clean_pipeline_success(self):
        """Test the complete cleaning pipeline."""
        # Create a temporary CSV file with all issues
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            f.write('v1,v2,Unnamed: 2\n')
            f.write('ham,Hello world,extra1\n')
            f.write('spam,Buy now!,extra2\n')
            f.write('ham,Hello world,extra3\n')  # Duplicate
            f.write('ham,,extra4\n')  # Null message
            f.write('spam,Win prize!,extra5\n')
            temp_file = f.name
        
        try:
            df = self.cleaner.clean_pipeline(temp_file)
            
            # Verify columns are renamed
            assert 'label' in df.columns
            assert 'message' in df.columns
            assert 'v1' not in df.columns
            assert 'v2' not in df.columns
            
            # Verify unnamed columns are dropped
            assert 'Unnamed: 2' not in df.columns
            
            # Verify duplicates are removed (should have 3 unique messages)
            assert len(df) == 3
            
            # Verify labels are encoded
            assert set(df['label'].unique()) == {0, 1}
            assert df['label'].dtype in [int, 'int64']
            
            # Verify no nulls remain
            assert df.isnull().sum().sum() == 0
        finally:
            os.unlink(temp_file)
    
    def test_clean_pipeline_file_not_found(self):
        """Test clean_pipeline with non-existent file."""
        with pytest.raises(FileNotFoundError):
            self.cleaner.clean_pipeline('nonexistent_file.csv')
    
    def test_clean_pipeline_preserves_data_integrity(self):
        """Test that clean_pipeline preserves data integrity."""
        # Create a clean dataset
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            f.write('v1,v2\n')
            f.write('ham,Hello world\n')
            f.write('spam,Buy now!\n')
            f.write('ham,Good morning\n')
            temp_file = f.name
        
        try:
            df = self.cleaner.clean_pipeline(temp_file)
            
            # Verify all rows are preserved
            assert len(df) == 3
            
            # Verify messages are intact
            assert 'Hello world' in df['message'].values
            assert 'Buy now!' in df['message'].values
            assert 'Good morning' in df['message'].values
            
            # Verify labels are correctly encoded
            ham_count = (df['label'] == 0).sum()
            spam_count = (df['label'] == 1).sum()
            assert ham_count == 2
            assert spam_count == 1
        finally:
            os.unlink(temp_file)
    
    @pytest.mark.skipif(not os.path.exists('data/spam.csv'), 
                        reason="UCI dataset not available")
    def test_load_actual_uci_dataset(self):
        """Test loading the actual UCI SMS Spam Collection dataset."""
        df = self.cleaner.load_data('data/spam.csv')
        
        # Verify DataFrame is loaded
        assert isinstance(df, pd.DataFrame)
        
        # Verify required columns exist
        assert 'v1' in df.columns
        assert 'v2' in df.columns
        
        # Verify dataset is not empty
        assert len(df) > 0
        
        # Verify labels are ham or spam
        unique_labels = df['v1'].unique()
        assert all(label in ['ham', 'spam'] for label in unique_labels)
    
    @pytest.mark.skipif(not os.path.exists('data/spam.csv'), 
                        reason="UCI dataset not available")
    def test_cleaned_dataset_size(self):
        """Test that cleaned dataset has approximately 5,572 messages."""
        df = self.cleaner.clean_pipeline('data/spam.csv')
        
        # Verify dataset has approximately 5,572 messages after cleaning
        # Allow for small variance (±50 messages)
        expected_size = 5572
        tolerance = 50
        
        assert abs(len(df) - expected_size) <= tolerance, \
            f"Expected ~{expected_size} messages, got {len(df)}"
        
        # Verify all required columns exist
        assert 'label' in df.columns
        assert 'message' in df.columns
        
        # Verify labels are encoded as 0 and 1
        assert set(df['label'].unique()).issubset({0, 1})
        
        # Verify no null values
        assert df.isnull().sum().sum() == 0
        
        # Verify no duplicate messages
        assert df['message'].nunique() == len(df)
    
    def test_load_data_invalid_csv_format(self):
        """Test loading a CSV with invalid format (missing v1 or v2)."""
        # Create a CSV with wrong column names
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            f.write('label,text\n')
            f.write('ham,Hello world\n')
            f.write('spam,Buy now!\n')
            temp_file = f.name
        
        try:
            with pytest.raises(ValueError) as exc_info:
                self.cleaner.load_data(temp_file)
            
            assert "must contain 'v1' and 'v2' columns" in str(exc_info.value)
        finally:
            os.unlink(temp_file)
    
    def test_load_data_empty_file(self):
        """Test loading an empty CSV file."""
        # Create an empty CSV with only headers
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            f.write('v1,v2\n')
            temp_file = f.name
        
        try:
            df = self.cleaner.load_data(temp_file)
            
            # Should load successfully but have 0 rows
            assert isinstance(df, pd.DataFrame)
            assert len(df) == 0
            assert 'v1' in df.columns
            assert 'v2' in df.columns
        finally:
            os.unlink(temp_file)
    
    def test_clean_pipeline_invalid_format(self):
        """Test clean_pipeline with invalid CSV format."""
        # Create a CSV without required columns
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            f.write('wrong_col1,wrong_col2\n')
            f.write('value1,value2\n')
            temp_file = f.name
        
        try:
            with pytest.raises(ValueError) as exc_info:
                self.cleaner.clean_pipeline(temp_file)
            
            assert "must contain 'v1' and 'v2' columns" in str(exc_info.value)
        finally:
            os.unlink(temp_file)
    
    def test_clean_pipeline_with_corrupted_data(self):
        """Test clean_pipeline handles corrupted data gracefully."""
        # Create a CSV with mixed valid and invalid labels
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            f.write('v1,v2\n')
            f.write('ham,Hello world\n')
            f.write('spam,Buy now!\n')
            f.write('invalid,This should fail\n')
            temp_file = f.name
        
        try:
            with pytest.raises(ValueError) as exc_info:
                self.cleaner.clean_pipeline(temp_file)
            
            assert 'Invalid labels found' in str(exc_info.value)
        finally:
            os.unlink(temp_file)
