"""
Data Preprocessing Functions
============================
Text cleaning and feature engineering functions.
Part of Movie Reviews Analysis project by Chis Bogdan & Ciurescu Raul.
"""

import re
import pandas as pd
import numpy as np
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter

class TextPreprocessor:
    """Advanced text preprocessing for movie reviews analysis."""
    
    def __init__(self):
        # Download required NLTK data
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
        except:
            pass
        
        self.stop_words = set(stopwords.words('english'))
        
    def advanced_text_cleaning(self, text):
        """
        Comprehensive text cleaning pipeline.
        
        Args:
            text (str): Raw text to clean
            
        Returns:
            str: Cleaned and normalized text
        """
        if pd.isna(text) or not isinstance(text, str):
            return ""
        
        # Remove HTML tags and entities
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'&\w+;', '', text)
        
        # Normalize punctuation
        text = re.sub(r'[^\w\s\.\!\?]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Convert to lowercase
        text = text.lower()
        
        return text
    
    def extract_text_features(self, text):
        """
        Extract various features from text.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            dict: Dictionary of extracted features
        """
        if not text:
            return {
                'word_count': 0,
                'char_count': 0,
                'sentence_count': 0,
                'avg_word_length': 0,
                'exclamation_count': 0,
                'question_count': 0,
                'uppercase_ratio': 0
            }
        
        # Basic counts
        words = text.split()
        word_count = len(words)
        char_count = len(text)
        sentence_count = len(re.findall(r'[.!?]+', text))
        
        # Average word length
        avg_word_length = np.mean([len(word) for word in words]) if words else 0
        
        # Punctuation analysis
        exclamation_count = text.count('!')
        question_count = text.count('?')
        
        # Uppercase ratio (for original text)
        uppercase_count = sum(1 for c in text if c.isupper())
        uppercase_ratio = uppercase_count / len(text) if text else 0
        
        return {
            'word_count': word_count,
            'char_count': char_count,
            'sentence_count': sentence_count,
            'avg_word_length': round(avg_word_length, 2),
            'exclamation_count': exclamation_count,
            'question_count': question_count,
            'uppercase_ratio': round(uppercase_ratio, 3)
        }
    
    def get_sentiment_features(self, text):
        """
        Extract sentiment-related features using TextBlob.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            dict: Sentiment features
        """
        if not text:
            return {
                'sentiment_polarity': 0.0,
                'sentiment_subjectivity': 0.0,
                'sentiment_magnitude': 0.0
            }
        
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        magnitude = abs(polarity)  # How strong the sentiment is
        
        return {
            'sentiment_polarity': round(polarity, 3),
            'sentiment_subjectivity': round(subjectivity, 3),
            'sentiment_magnitude': round(magnitude, 3)
        }
    
    def categorize_sentiment(self, polarity_score, threshold=0.1):
        """
        Categorize sentiment polarity into positive/negative/neutral.
        
        Args:
            polarity_score (float): Sentiment polarity (-1 to 1)
            threshold (float): Threshold for neutral classification
            
        Returns:
            str: Sentiment category
        """
        if polarity_score > threshold:
            return 'Positive'
        elif polarity_score < -threshold:
            return 'Negative'
        else:
            return 'Neutral'

class FeatureEngineer:
    """Feature engineering for machine learning models."""
    
    def __init__(self):
        self.tfidf_vectorizer = None
        self.count_vectorizer = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
    
    def create_tfidf_features(self, texts, max_features=100, ngram_range=(1, 1)):
        """
        Create TF-IDF feature matrix from texts.
        
        Args:
            texts (list): List of text documents
            max_features (int): Maximum number of features
            ngram_range (tuple): N-gram range for feature extraction
            
        Returns:
            scipy.sparse.matrix: TF-IDF feature matrix
        """
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=ngram_range,
            min_df=2,
            max_df=0.8
        )
        
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
        return tfidf_matrix
    
    def get_feature_names(self):
        """Get feature names from fitted TF-IDF vectorizer."""
        if self.tfidf_vectorizer:
            return self.tfidf_vectorizer.get_feature_names_out()
        return []
    
    def create_combined_features(self, df):
        """
        Create combined feature matrix for machine learning.
        
        Args:
            df (pandas.DataFrame): DataFrame with review data
            
        Returns:
            numpy.ndarray: Combined feature matrix
        """
        # Text features (TF-IDF)
        tfidf_features = self.create_tfidf_features(df['review_text'], max_features=50)
        
        # Numerical features
        numerical_features = df[['word_count', 'char_count', 'sentiment_subjectivity']].values
        
        # Scale numerical features
        numerical_features_scaled = self.scaler.fit_transform(numerical_features)
        
        # Combine features
        combined_features = np.hstack([
            tfidf_features.toarray(),
            numerical_features_scaled
        ])
        
        return combined_features

def process_movie_reviews(csv_file_path):
    """
    Complete preprocessing pipeline for movie reviews data.
    
    Args:
        csv_file_path (str): Path to CSV file with movie reviews
        
    Returns:
        pandas.DataFrame: Processed DataFrame ready for analysis
    """
    print("Starting comprehensive data preprocessing...")
    
    # Load data
    df = pd.read_csv(csv_file_path)
    print(f"Loaded {len(df)} reviews from {df['movie_title'].nunique()} movies")
    
    # Initialize processors
    text_processor = TextPreprocessor()
    
    # Clean text
    print("Cleaning review text...")
    df['cleaned_text'] = df['review_text'].apply(text_processor.advanced_text_cleaning)
    
    # Extract text features
    print("Extracting text features...")
    text_features = df['cleaned_text'].apply(text_processor.extract_text_features)
    text_features_df = pd.DataFrame(text_features.tolist())
    
    # Extract sentiment features
    print("Analyzing sentiment...")
    sentiment_features = df['cleaned_text'].apply(text_processor.get_sentiment_features)
    sentiment_features_df = pd.DataFrame(sentiment_features.tolist())
    
    # Categorize sentiment
    df['sentiment_category'] = sentiment_features_df['sentiment_polarity'].apply(
        text_processor.categorize_sentiment
    )
    
    # Combine all features
    processed_df = pd.concat([df, text_features_df, sentiment_features_df], axis=1)
    
    # Remove duplicates and handle missing values
    processed_df = processed_df.drop_duplicates(subset=['cleaned_text'])
    processed_df = processed_df.dropna(subset=['cleaned_text'])
    
    print(f"Preprocessing complete! Final dataset: {len(processed_df)} reviews")
    print(f"Sentiment distribution:")
    print(processed_df['sentiment_category'].value_counts())
    
    return processed_df

# Example usage and testing
if __name__ == "__main__":
    # Test the preprocessing functions
    sample_reviews = [
        "This movie was AMAZING!!! Best film of the year by far.",
        "Terrible acting and boring plot. Complete waste of time.",
        "Pretty good movie, not great but entertaining enough.",
        "Outstanding performances and brilliant cinematography throughout."
    ]
    
    processor = TextPreprocessor()
    
    print("Testing text preprocessing functions:")
    print("-" * 40)
    
    for i, review in enumerate(sample_reviews, 1):
        print(f"Review {i}: {review}")
        cleaned = processor.advanced_text_cleaning(review)
        features = processor.extract_text_features(review)
        sentiment = processor.get_sentiment_features(cleaned)
        category = processor.categorize_sentiment(sentiment['sentiment_polarity'])
        
        print(f"  Cleaned: {cleaned}")
        print(f"  Features: {features}")
        print(f"  Sentiment: {sentiment}")
        print(f"  Category: {category}")
        print()