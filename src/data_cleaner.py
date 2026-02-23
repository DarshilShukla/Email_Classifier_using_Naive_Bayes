"""
Data cleaning module for SMS spam classification.

This module provides the DataCleaner class for loading and cleaning
the UCI SMS Spam Collection dataset.
"""

import pandas as pd
from typing import Optional


class DataCleaner:
    """
    Handles loading and cleaning of the SMS spam dataset.
    
    The DataCleaner class provides methods to load raw CSV data,
    rename columns, remove unnamed columns, handle duplicates,
    drop null values, and encode labels for machine learning.
    """
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load the SMS spam dataset from a CSV file.
        
        Args:
            file_path: Path to the spam.csv file
            
        Returns:
            DataFrame containing the raw dataset
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If required columns (v1, v2) are missing
        """
        try:
            df = pd.read_csv(file_path, encoding='latin-1')
        except FileNotFoundError:
            raise FileNotFoundError(f"Dataset file not found at: {file_path}")
        
        # Verify required columns exist
        if 'v1' not in df.columns or 'v2' not in df.columns:
            raise ValueError("Dataset must contain 'v1' and 'v2' columns")
        
        return df
    
    def rename_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Rename columns from v1 to 'label' and v2 to 'message'.
        
        Args:
            df: DataFrame with original column names
            
        Returns:
            DataFrame with renamed columns
        """
        df_renamed = df.rename(columns={'v1': 'label', 'v2': 'message'})
        return df_renamed
    
    def drop_unnamed_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove all columns with names matching 'Unnamed'.
        
        Args:
            df: DataFrame potentially containing unnamed columns
            
        Returns:
            DataFrame with unnamed columns removed
        """
        # Find columns that contain 'Unnamed' in their name
        unnamed_cols = [col for col in df.columns if 'Unnamed' in str(col)]
        
        if unnamed_cols:
            df_cleaned = df.drop(columns=unnamed_cols)
            return df_cleaned
        
        return df
    
    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicate messages from the dataset.
        
        Preserves the first occurrence of each unique message.
        
        Args:
            df: DataFrame potentially containing duplicate messages
            
        Returns:
            DataFrame with duplicate messages removed
        """
        df_no_duplicates = df.drop_duplicates(subset=['message'], keep='first')
        return df_no_duplicates
    
    def drop_nulls(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove all rows containing null values.
        
        Args:
            df: DataFrame potentially containing null values
            
        Returns:
            DataFrame with rows containing nulls removed
        """
        df_no_nulls = df.dropna()
        return df_no_nulls
    
    def encode_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode labels: map 'ham' to 0 and 'spam' to 1.
        
        Args:
            df: DataFrame with 'label' column containing 'ham' and 'spam'
            
        Returns:
            DataFrame with encoded labels
            
        Raises:
            ValueError: If labels contain values other than 'ham' or 'spam'
            TypeError: If label column is not string type
        """
        if 'label' not in df.columns:
            raise ValueError("DataFrame must contain 'label' column")
        
        # Check for invalid labels
        unique_labels = df['label'].unique()
        valid_labels = {'ham', 'spam'}
        invalid_labels = set(unique_labels) - valid_labels
        
        if invalid_labels:
            raise ValueError(f"Invalid labels found: {invalid_labels}. Only 'ham' and 'spam' are allowed.")
        
        # Map ham→0, spam→1
        df_encoded = df.copy()
        df_encoded['label'] = df_encoded['label'].map({'ham': 0, 'spam': 1})
        
        return df_encoded
    
    def clean_pipeline(self, file_path: str) -> pd.DataFrame:
        """
        Execute the complete data cleaning pipeline.
        
        Chains all cleaning operations in the correct order:
        1. Load data from CSV
        2. Rename columns (v1→label, v2→message)
        3. Drop unnamed columns
        4. Remove duplicate messages
        5. Drop rows with null values
        6. Encode labels (ham→0, spam→1)
        
        Args:
            file_path: Path to the spam.csv file
            
        Returns:
            Cleaned DataFrame ready for analysis and modeling
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If required columns are missing or labels are invalid
        """
        # Step 1: Load data
        df = self.load_data(file_path)
        
        # Step 2: Rename columns
        df = self.rename_columns(df)
        
        # Step 3: Drop unnamed columns
        df = self.drop_unnamed_columns(df)
        
        # Step 4: Remove duplicates
        df = self.remove_duplicates(df)
        
        # Step 5: Drop nulls
        df = self.drop_nulls(df)
        
        # Step 6: Encode labels
        df = self.encode_labels(df)
        
        return df

    def clean_pipeline(self, file_path: str) -> pd.DataFrame:
        """
        Execute the complete data cleaning pipeline.

        Chains all cleaning operations in the correct order:
        1. Load data from CSV
        2. Rename columns (v1→label, v2→message)
        3. Drop unnamed columns
        4. Remove duplicate messages
        5. Drop rows with null values
        6. Encode labels (ham→0, spam→1)

        Args:
            file_path: Path to the spam.csv file

        Returns:
            Cleaned DataFrame ready for analysis and modeling

        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If required columns are missing or labels are invalid
        """
        # Step 1: Load data
        df = self.load_data(file_path)

        # Step 2: Rename columns
        df = self.rename_columns(df)

        # Step 3: Drop unnamed columns
        df = self.drop_unnamed_columns(df)

        # Step 4: Remove duplicates
        df = self.remove_duplicates(df)

        # Step 5: Drop nulls
        df = self.drop_nulls(df)

        # Step 6: Encode labels
        df = self.encode_labels(df)

        return df
