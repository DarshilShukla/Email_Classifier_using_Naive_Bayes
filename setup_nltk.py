"""Setup script to download required NLTK data."""

import nltk


def download_nltk_data():
    """Download required NLTK datasets."""
    print("Downloading NLTK data...")
    
    # Download stopwords corpus
    print("Downloading stopwords...")
    nltk.download('stopwords', quiet=False)
    
    # Download punkt tokenizer
    print("Downloading punkt tokenizer...")
    nltk.download('punkt', quiet=False)
    
    print("NLTK data download complete!")


if __name__ == "__main__":
    download_nltk_data()
