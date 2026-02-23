"""
Unit tests for the EDAAnalyzer class.

Tests the feature computation methods for character count, word count,
and sentence count.
"""

import pytest
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
from src.eda_analyzer import EDAAnalyzer


class TestEDAAnalyzer:
    """Test suite for EDAAnalyzer class."""
    
    @pytest.fixture
    def analyzer(self):
        """Create an EDAAnalyzer instance for testing."""
        return EDAAnalyzer()
    
    @pytest.fixture
    def sample_df(self):
        """Create a sample DataFrame for testing."""
        return pd.DataFrame({
            'label': [0, 1, 0],
            'message': [
                'Hello world',
                'Buy now! Get 50% off. Limited time offer!',
                'Meeting at 3pm'
            ]
        })
    
    # Character count tests
    
    def test_compute_char_count_basic(self, analyzer, sample_df):
        """Test character count computation on basic messages."""
        result = analyzer.compute_char_count(sample_df)
        
        assert 'char_count' in result.columns
        assert result['char_count'].tolist() == [11, 41, 14]
    
    def test_compute_char_count_empty_message(self, analyzer):
        """Test character count with empty message."""
        df = pd.DataFrame({'message': ['']})
        result = analyzer.compute_char_count(df)
        
        assert result['char_count'].iloc[0] == 0
    
    def test_compute_char_count_special_chars(self, analyzer):
        """Test character count includes special characters."""
        df = pd.DataFrame({'message': ['Hello! @#$ 123']})
        result = analyzer.compute_char_count(df)
        
        assert result['char_count'].iloc[0] == 14
    
    def test_compute_char_count_missing_column(self, analyzer):
        """Test error handling when message column is missing."""
        df = pd.DataFrame({'label': [0, 1]})
        
        with pytest.raises(ValueError, match="must contain 'message' column"):
            analyzer.compute_char_count(df)
    
    # Word count tests
    
    def test_compute_word_count_basic(self, analyzer, sample_df):
        """Test word count computation on basic messages."""
        result = analyzer.compute_word_count(sample_df)
        
        assert 'word_count' in result.columns
        assert result['word_count'].tolist() == [2, 8, 3]
    
    def test_compute_word_count_empty_message(self, analyzer):
        """Test word count with empty message."""
        df = pd.DataFrame({'message': ['']})
        result = analyzer.compute_word_count(df)
        
        assert result['word_count'].iloc[0] == 0
    
    def test_compute_word_count_single_word(self, analyzer):
        """Test word count with single word."""
        df = pd.DataFrame({'message': ['Hello']})
        result = analyzer.compute_word_count(df)
        
        assert result['word_count'].iloc[0] == 1
    
    def test_compute_word_count_multiple_spaces(self, analyzer):
        """Test word count with multiple spaces between words."""
        df = pd.DataFrame({'message': ['Hello    world']})
        result = analyzer.compute_word_count(df)
        
        # split() handles multiple spaces correctly
        assert result['word_count'].iloc[0] == 2
    
    def test_compute_word_count_missing_column(self, analyzer):
        """Test error handling when message column is missing."""
        df = pd.DataFrame({'label': [0, 1]})
        
        with pytest.raises(ValueError, match="must contain 'message' column"):
            analyzer.compute_word_count(df)
    
    # Sentence count tests
    
    def test_compute_sentence_count_basic(self, analyzer, sample_df):
        """Test sentence count computation on basic messages."""
        result = analyzer.compute_sentence_count(sample_df)
        
        assert 'sentence_count' in result.columns
        # 'Hello world' -> 1 (no punctuation, but has text)
        # 'Buy now! Get 50% off. Limited time offer!' -> 3 (!, ., !)
        # 'Meeting at 3pm' -> 1 (no punctuation, but has text)
        assert result['sentence_count'].tolist() == [1, 3, 1]
    
    def test_compute_sentence_count_single_sentence(self, analyzer):
        """Test sentence count with single sentence."""
        df = pd.DataFrame({'message': ['This is a sentence.']})
        result = analyzer.compute_sentence_count(df)
        
        assert result['sentence_count'].iloc[0] == 1
    
    def test_compute_sentence_count_multiple_punctuation(self, analyzer):
        """Test sentence count with different punctuation marks."""
        df = pd.DataFrame({'message': ['Hello! How are you? I am fine.']})
        result = analyzer.compute_sentence_count(df)
        
        assert result['sentence_count'].iloc[0] == 3
    
    def test_compute_sentence_count_no_punctuation(self, analyzer):
        """Test sentence count with no punctuation."""
        df = pd.DataFrame({'message': ['Hello world']})
        result = analyzer.compute_sentence_count(df)
        
        # Should count as 1 sentence if text exists
        assert result['sentence_count'].iloc[0] == 1
    
    def test_compute_sentence_count_empty_message(self, analyzer):
        """Test sentence count with empty message."""
        df = pd.DataFrame({'message': ['']})
        result = analyzer.compute_sentence_count(df)
        
        assert result['sentence_count'].iloc[0] == 0
    
    def test_compute_sentence_count_only_punctuation(self, analyzer):
        """Test sentence count with only punctuation."""
        df = pd.DataFrame({'message': ['...!!!???']})
        result = analyzer.compute_sentence_count(df)
        
        # Should count each punctuation mark
        assert result['sentence_count'].iloc[0] == 9
    
    def test_compute_sentence_count_missing_column(self, analyzer):
        """Test error handling when message column is missing."""
        df = pd.DataFrame({'label': [0, 1]})
        
        with pytest.raises(ValueError, match="must contain 'message' column"):
            analyzer.compute_sentence_count(df)
    
    # Integration tests
    
    def test_all_features_together(self, analyzer, sample_df):
        """Test computing all features on the same DataFrame."""
        result = analyzer.compute_char_count(sample_df)
        result = analyzer.compute_word_count(result)
        result = analyzer.compute_sentence_count(result)
        
        assert 'char_count' in result.columns
        assert 'word_count' in result.columns
        assert 'sentence_count' in result.columns
        assert len(result) == 3
    
    def test_preserves_original_columns(self, analyzer, sample_df):
        """Test that original columns are preserved."""
        result = analyzer.compute_char_count(sample_df)
        
        assert 'label' in result.columns
        assert 'message' in result.columns
        assert result['label'].tolist() == sample_df['label'].tolist()
        assert result['message'].tolist() == sample_df['message'].tolist()

    # Visualization tests
    
    def test_generate_histograms_basic(self, analyzer, sample_df, monkeypatch):
        """Test histogram generation with valid data."""
        # Add required columns
        df = analyzer.compute_char_count(sample_df)
        df = analyzer.compute_word_count(df)
        df = analyzer.compute_sentence_count(df)
        
        # Mock plt.show() to prevent display during tests
        show_called = []
        monkeypatch.setattr('matplotlib.pyplot.show', lambda: show_called.append(True))
        
        # Should not raise an error
        analyzer.generate_histograms(df)
        assert len(show_called) == 1
    
    def test_generate_histograms_missing_columns(self, analyzer, sample_df):
        """Test histogram generation with missing columns."""
        with pytest.raises(ValueError, match="must contain columns"):
            analyzer.generate_histograms(sample_df)
    
    def test_generate_correlation_heatmap_basic(self, analyzer, sample_df, monkeypatch):
        """Test correlation heatmap generation with valid data."""
        # Add required columns
        df = analyzer.compute_char_count(sample_df)
        df = analyzer.compute_word_count(df)
        df = analyzer.compute_sentence_count(df)
        
        # Mock plt.show() to prevent display during tests
        show_called = []
        monkeypatch.setattr('matplotlib.pyplot.show', lambda: show_called.append(True))
        
        # Should not raise an error
        analyzer.generate_correlation_heatmap(df)
        assert len(show_called) == 1
    
    def test_generate_correlation_heatmap_no_numerical_columns(self, analyzer):
        """Test correlation heatmap with no numerical columns."""
        df = pd.DataFrame({'message': ['Hello', 'World']})
        
        with pytest.raises(ValueError, match="must contain numerical columns"):
            analyzer.generate_correlation_heatmap(df)
    
    def test_get_class_distribution_basic(self, analyzer, sample_df):
        """Test class distribution computation."""
        result = analyzer.get_class_distribution(sample_df)
        
        assert 'ham_percentage' in result
        assert 'spam_percentage' in result
        # 2 ham (0) and 1 spam (1) out of 3 total
        assert result['ham_percentage'] == pytest.approx(66.67, rel=0.01)
        assert result['spam_percentage'] == pytest.approx(33.33, rel=0.01)
    
    def test_get_class_distribution_all_ham(self, analyzer):
        """Test class distribution with all ham messages."""
        df = pd.DataFrame({'label': [0, 0, 0, 0]})
        result = analyzer.get_class_distribution(df)
        
        assert result['ham_percentage'] == 100.0
        assert result['spam_percentage'] == 0.0
    
    def test_get_class_distribution_all_spam(self, analyzer):
        """Test class distribution with all spam messages."""
        df = pd.DataFrame({'label': [1, 1, 1]})
        result = analyzer.get_class_distribution(df)
        
        assert result['ham_percentage'] == 0.0
        assert result['spam_percentage'] == 100.0
    
    def test_get_class_distribution_missing_column(self, analyzer):
        """Test class distribution with missing label column."""
        df = pd.DataFrame({'message': ['Hello', 'World']})
        
        with pytest.raises(ValueError, match="must contain 'label' column"):
            analyzer.get_class_distribution(df)
    
    def test_get_class_distribution_empty_dataframe(self, analyzer):
        """Test class distribution with empty DataFrame."""
        df = pd.DataFrame({'label': []})
        
        with pytest.raises(ValueError, match="DataFrame is empty"):
            analyzer.get_class_distribution(df)
    
    # UCI Dataset tests
    
    @pytest.mark.skipif(
        not pd.io.common.file_exists('data/spam.csv'),
        reason="UCI dataset not available"
    )
    def test_uci_dataset_class_distribution(self, analyzer):
        """
        Test class distribution on UCI SMS Spam Collection dataset.
        
        Validates Requirement 2.6: Dataset should contain approximately
        87% ham and 13% spam messages.
        """
        from src.data_cleaner import DataCleaner
        
        # Load and clean the UCI dataset
        cleaner = DataCleaner()
        df = cleaner.clean_pipeline('data/spam.csv')
        
        # Get class distribution
        distribution = analyzer.get_class_distribution(df)
        
        # Verify approximately 87% ham, 13% spam (allow 2% tolerance)
        assert distribution['ham_percentage'] == pytest.approx(87.0, abs=2.0), \
            f"Expected ~87% ham, got {distribution['ham_percentage']:.2f}%"
        assert distribution['spam_percentage'] == pytest.approx(13.0, abs=2.0), \
            f"Expected ~13% spam, got {distribution['spam_percentage']:.2f}%"
    
    @pytest.mark.skipif(
        not pd.io.common.file_exists('data/spam.csv'),
        reason="UCI dataset not available"
    )
    def test_uci_dataset_visualizations(self, analyzer, monkeypatch):
        """
        Test that visualization functions run without errors on UCI dataset.
        
        Validates Requirement 2.6: Visualizations should be generated successfully.
        """
        from src.data_cleaner import DataCleaner
        
        # Load and clean the UCI dataset
        cleaner = DataCleaner()
        df = cleaner.clean_pipeline('data/spam.csv')
        
        # Add EDA features
        df = analyzer.compute_char_count(df)
        df = analyzer.compute_word_count(df)
        df = analyzer.compute_sentence_count(df)
        
        # Mock plt.show() to prevent display during tests
        show_called = []
        monkeypatch.setattr('matplotlib.pyplot.show', lambda: show_called.append(True))
        
        # Test histograms - should not raise an error
        analyzer.generate_histograms(df)
        assert len(show_called) == 1
        
        # Test correlation heatmap - should not raise an error
        analyzer.generate_correlation_heatmap(df)
        assert len(show_called) == 2
