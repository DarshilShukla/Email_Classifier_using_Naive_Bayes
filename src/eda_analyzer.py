"""
Exploratory Data Analysis module for SMS spam classification.

This module provides the EDAAnalyzer class for analyzing message
characteristics and generating insights from the dataset.
"""

import pandas as pd
import re
from typing import Dict
import matplotlib.pyplot as plt
import seaborn as sns


class EDAAnalyzer:
    """
    Handles exploratory data analysis of the SMS spam dataset.
    
    The EDAAnalyzer class provides methods to compute message statistics
    (character count, word count, sentence count) and generate visualizations
    to understand patterns that distinguish spam from ham messages.
    """
    
    def compute_char_count(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute character count for each message.
        
        Adds a 'char_count' column containing the length of each message string.
        
        Args:
            df: DataFrame with 'message' column
            
        Returns:
            DataFrame with added 'char_count' column
            
        Raises:
            ValueError: If 'message' column is missing
        """
        if 'message' not in df.columns:
            raise ValueError("DataFrame must contain 'message' column")
        
        df_with_char_count = df.copy()
        df_with_char_count['char_count'] = df_with_char_count['message'].apply(len)
        
        return df_with_char_count
    
    def compute_word_count(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute word count for each message.
        
        Adds a 'word_count' column containing the number of whitespace-separated
        tokens in each message.
        
        Args:
            df: DataFrame with 'message' column
            
        Returns:
            DataFrame with added 'word_count' column
            
        Raises:
            ValueError: If 'message' column is missing
        """
        if 'message' not in df.columns:
            raise ValueError("DataFrame must contain 'message' column")
        
        df_with_word_count = df.copy()
        df_with_word_count['word_count'] = df_with_word_count['message'].apply(
            lambda x: len(x.split())
        )
        
        return df_with_word_count
    
    def compute_sentence_count(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute sentence count for each message.
        
        Adds a 'sentence_count' column containing the number of sentences
        in each message, determined by counting sentence-ending punctuation
        marks (., !, ?).
        
        Args:
            df: DataFrame with 'message' column
            
        Returns:
            DataFrame with added 'sentence_count' column
            
        Raises:
            ValueError: If 'message' column is missing
        """
        if 'message' not in df.columns:
            raise ValueError("DataFrame must contain 'message' column")
        
        df_with_sentence_count = df.copy()
        
        def count_sentences(text: str) -> int:
            """Count sentences by finding sentence-ending punctuation."""
            # Count occurrences of sentence-ending punctuation: . ! ?
            sentence_endings = re.findall(r'[.!?]', text)
            count = len(sentence_endings)
            # If no sentence-ending punctuation found but text exists, count as 1 sentence
            if count == 0 and text.strip():
                return 1
            return count
        
        df_with_sentence_count['sentence_count'] = df_with_sentence_count['message'].apply(
            count_sentences
        )
        
        return df_with_sentence_count

    def generate_histograms(self, df: pd.DataFrame) -> None:
        """
        Generate histograms showing message length distributions.
        
        Creates a figure with three subplots showing distributions of
        character count, word count, and sentence count for the messages.
        
        Args:
            df: DataFrame with 'char_count', 'word_count', and 'sentence_count' columns
            
        Raises:
            ValueError: If required columns are missing
        """
        required_cols = ['char_count', 'word_count', 'sentence_count']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"DataFrame must contain columns: {missing_cols}")
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        axes[0].hist(df['char_count'], bins=50, edgecolor='black')
        axes[0].set_xlabel('Character Count')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Character Count Distribution')
        
        axes[1].hist(df['word_count'], bins=50, edgecolor='black')
        axes[1].set_xlabel('Word Count')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Word Count Distribution')
        
        axes[2].hist(df['sentence_count'], bins=20, edgecolor='black')
        axes[2].set_xlabel('Sentence Count')
        axes[2].set_ylabel('Frequency')
        axes[2].set_title('Sentence Count Distribution')
        
        plt.tight_layout()
        plt.show()
    
    def generate_correlation_heatmap(self, df: pd.DataFrame) -> None:
        """
        Generate a correlation heatmap for computed features.
        
        Creates a heatmap showing correlations between numerical features
        (char_count, word_count, sentence_count, and label if present).
        
        Args:
            df: DataFrame with numerical feature columns
            
        Raises:
            ValueError: If DataFrame has no numerical columns
        """
        # Select only numerical columns
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        if not numerical_cols:
            raise ValueError("DataFrame must contain numerical columns")
        
        # Compute correlation matrix
        corr_matrix = df[numerical_cols].corr()
        
        # Create heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                    square=True, linewidths=1, fmt='.2f')
        plt.title('Feature Correlation Heatmap')
        plt.tight_layout()
        plt.show()
    
    def get_class_distribution(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Compute ham/spam class distribution percentages.
        
        Calculates the percentage of ham (0) and spam (1) messages in the dataset.
        
        Args:
            df: DataFrame with 'label' column (0 for ham, 1 for spam)
            
        Returns:
            Dictionary with keys 'ham_percentage' and 'spam_percentage'
            
        Raises:
            ValueError: If 'label' column is missing
        """
        if 'label' not in df.columns:
            raise ValueError("DataFrame must contain 'label' column")
        
        total_count = len(df)
        if total_count == 0:
            raise ValueError("DataFrame is empty")
        
        ham_count = (df['label'] == 0).sum()
        spam_count = (df['label'] == 1).sum()
        
        ham_percentage = (ham_count / total_count) * 100
        spam_percentage = (spam_count / total_count) * 100
        
        return {
            'ham_percentage': ham_percentage,
            'spam_percentage': spam_percentage
        }
