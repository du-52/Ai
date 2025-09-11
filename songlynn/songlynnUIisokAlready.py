import streamlit as st
import os
import urllib.request
import tarfile
import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from scipy.sparse import hstack, csr_matrix
import glob
from collections import Counter
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings('ignore')

# Configure Streamlit page
st.set_page_config(
    page_title="IMDB Sentiment Analyzer",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FF6B35;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sentiment-positive {
        color: #28a745;
        font-weight: bold;
    }
    .sentiment-negative {
        color: #dc3545;
        font-weight: bold;
    }
    .sentiment-neutral {
        color: #6c757d;
        font-weight: bold;
    }
    .confidence-high {
        color: #28a745;
    }
    .confidence-medium {
        color: #ffc107;
    }
    .confidence-low {
        color: #dc3545;
    }
</style>
""", unsafe_allow_html=True)

# All the original classes remain the same
class IMDBDatasetLoader:
    def __init__(self, data_path="aclImdb"):
        self.data_path = data_path
        self.url = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
        
    def download_and_extract(self):
        """Download and extract IMDB dataset if not present"""
        if not os.path.exists(self.data_path):
            with st.spinner("Downloading IMDB dataset... This may take a few minutes."):
                filename = "aclImdb_v1.tar.gz"
                
                try:
                    progress_bar = st.progress(0)
                    
                    def show_progress(block_num, block_size, total_size):
                        downloaded = block_num * block_size
                        progress = min(downloaded / total_size, 1.0)
                        progress_bar.progress(progress)
                    
                    urllib.request.urlretrieve(self.url, filename, show_progress)
                    st.success("Download completed!")
                    
                    st.info("Extracting dataset...")
                    with tarfile.open(filename, 'r:gz') as tar:
                        tar.extractall('.')
                        
                    os.remove(filename)  # Clean up
                    st.success("Dataset extracted successfully!")
                    
                except Exception as e:
                    st.error(f"Error downloading dataset: {e}")
                    return False
        else:
            st.info("IMDB dataset already exists!")
        return True
    
    def load_reviews_from_folder(self, folder_path, label):
        """Load reviews from a folder"""
        reviews = []
        labels = []
        
        if not os.path.exists(folder_path):
            st.warning(f"Folder not found: {folder_path}")
            return reviews, labels
            
        files = glob.glob(os.path.join(folder_path, "*.txt"))
        
        for file_path in files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    review = f.read().strip()
                    if len(review) > 10:  # Filter very short reviews
                        reviews.append(review)
                        labels.append(label)
            except Exception as e:
                continue
                
        return reviews, labels
    
    def create_three_class_labels(self, texts, binary_labels):
        """Convert binary labels to three-class using rule-based neutral detection"""
        processor = FixedTextProcessor()
        three_class_labels = []
        neutral_count = 0
        
        for i, (text, binary_label) in enumerate(zip(texts, binary_labels)):
            # Calculate sentiment score
            sentiment_score, _ = processor.calculate_sentiment_score(text)
            
            # Decision logic for neutral detection
            is_neutral = False
            
            # Check for neutral indicators
            if (abs(sentiment_score) < 0.3 or  # Low sentiment score
                any(word in text.lower() for word in ['okay', 'alright', 'average', 'so-so', 'mixed', 'mediocre']) or
                ('good' in text.lower() and 'but' in text.lower()) or  # Mixed sentiment
                ('like' in text.lower() and 'but' in text.lower()) or
                ('not bad' in text.lower() and 'not good' in text.lower())):
                is_neutral = True
                
            # Assign three-class label
            if is_neutral:
                three_class_labels.append(1)  # Neutral
                neutral_count += 1
            else:
                three_class_labels.append(binary_label * 2)  # 0->0 (negative), 1->2 (positive)
                
        return three_class_labels
    
    def load_imdb_data(self, subset_size=None):
        """Load IMDB dataset with three-class conversion"""
        if not self.download_and_extract():
            return None, None, None, None
            
        # Load training data
        train_pos_path = os.path.join(self.data_path, "train", "pos")
        train_neg_path = os.path.join(self.data_path, "train", "neg")
        
        train_pos_reviews, train_pos_labels = self.load_reviews_from_folder(train_pos_path, 1)
        train_neg_reviews, train_neg_labels = self.load_reviews_from_folder(train_neg_path, 0)
        
        # Load test data
        test_pos_path = os.path.join(self.data_path, "test", "pos")
        test_neg_path = os.path.join(self.data_path, "test", "neg")
        
        test_pos_reviews, test_pos_labels = self.load_reviews_from_folder(test_pos_path, 1)
        test_neg_reviews, test_neg_labels = self.load_reviews_from_folder(test_neg_path, 0)
        
        # Combine data
        train_texts = train_pos_reviews + train_neg_reviews
        train_binary_labels = train_pos_labels + train_neg_labels
        
        test_texts = test_pos_reviews + test_neg_reviews
        test_binary_labels = test_pos_labels + test_neg_labels
        
        # Apply subset if specified
        if subset_size:
            train_size = min(subset_size, len(train_texts))
            test_size = min(subset_size // 4, len(test_texts))
            
            # Random sampling
            train_indices = np.random.choice(len(train_texts), train_size, replace=False)
            test_indices = np.random.choice(len(test_texts), test_size, replace=False)
            
            train_texts = [train_texts[i] for i in train_indices]
            train_binary_labels = [train_binary_labels[i] for i in train_indices]
            test_texts = [test_texts[i] for i in test_indices]
            test_binary_labels = [test_binary_labels[i] for i in test_indices]
        
        # Convert to three-class
        train_labels = self.create_three_class_labels(train_texts, train_binary_labels)
        test_labels = self.create_three_class_labels(test_texts, test_binary_labels)
        
        # Shuffle data
        train_indices = np.random.permutation(len(train_texts))
        train_texts = [train_texts[i] for i in train_indices]
        train_labels = [train_labels[i] for i in train_indices]
        
        test_indices = np.random.permutation(len(test_texts))
        test_texts = [test_texts[i] for i in test_indices]
        test_labels = [test_labels[i] for i in test_indices]
        
        return train_texts, train_labels, test_texts, test_labels

class FixedTextProcessor:
    def __init__(self):
        # Enhanced sentiment lexicon with proper weights
        self.sentiment_lexicon = {
            # Strong positive (weight: 2.5)
            'amazing': 2.5, 'excellent': 2.5, 'fantastic': 2.5, 'wonderful': 2.5, 
            'brilliant': 2.5, 'outstanding': 2.5, 'superb': 2.5, 'magnificent': 2.5, 
            'awesome': 2.5, 'perfect': 2.5, 'incredible': 2.5, 'phenomenal': 2.5,
            'love': 2.5, 'adore': 2.5, 'best': 2.5, 'greatest': 2.5, 'masterpiece': 2.5,
            
            # Moderate positive (weight: 1.8)
            'good': 1.8, 'nice': 1.8, 'great': 1.8, 'fine': 1.8, 'enjoyable': 1.8,
            'satisfying': 1.8, 'solid': 1.8, 'recommend': 1.8, 'positive': 1.8, 
            'like': 1.8, 'enjoy': 1.8, 'pleased': 1.8, 'happy': 1.8, 'impressive': 1.8,
            
            # Mild positive (weight: 1.2)
            'decent': 1.2, 'pleasant': 1.2, 'reasonable': 1.2, 'acceptable': 1.2,
            'fair': 1.2, 'alright': 1.2, 'okay': 0.8,  # Lower weight for okay
            
            # Neutral (weight: 0)
            'average': 0, 'ordinary': 0, 'typical': 0, 'standard': 0, 'normal': 0,
            'regular': 0, 'usual': 0, 'common': 0, 'mediocre': 0, 'mixed': 0,
            'so-so': 0, 'meh': 0, 'whatever': 0,
            
            # Mild negative (weight: -1.5)
            'disappointing': -1.5, 'boring': -1.5, 'dull': -1.5, 'slow': -1.5, 
            'weak': -1.5, 'poor': -1.5, 'lacking': -1.5, 'limited': -1.5, 
            'problematic': -1.5, 'flawed': -1.5, 'annoying': -1.5, 'confusing': -1.5,
            
            # Moderate negative (weight: -2.2)
            'bad': -2.2, 'dislike': -2.2, 'hate': -2.2, 'disgusting': -2.2, 
            'pathetic': -2.2, 'useless': -2.2, 'worthless': -2.2, 'stupid': -2.2, 
            'ridiculous': -2.2, 'frustrating': -2.2, 'disappointing': -2.2,
            
            # Strong negative (weight: -2.8)
            'terrible': -2.8, 'awful': -2.8, 'horrible': -2.8, 'worst': -2.8,
            'abysmal': -2.8, 'atrocious': -2.8, 'dreadful': -2.8, 'appalling': -2.8,
            'horrendous': -2.8, 'despise': -2.8, 'abhor': -2.8, 'loathe': -2.8
        }
        
        # Enhanced negation detection
        self.negation_words = {
            'not', 'no', 'never', 'nothing', 'nowhere', 'nobody', 'none',
            'neither', 'nor', 'cannot', "can't", "won't", "shouldn't", 
            "wouldn't", "couldn't", "doesn't", "don't", "isn't", "aren't",
            "wasn't", "weren't", "hasn't", "haven't", "hadn't", "mustn't",
            'without', 'hardly', 'barely', 'scarcely', 'seldom', 'rarely'
        }
        
        # Intensifiers
        self.intensifiers = {
            'very': 1.6, 'extremely': 2.2, 'really': 1.4, 'quite': 1.3, 'rather': 1.2,
            'pretty': 1.2, 'absolutely': 1.9, 'completely': 1.8, 'totally': 1.7,
            'entirely': 1.7, 'incredibly': 1.9, 'amazingly': 1.8, 'so': 1.5,
            'truly': 1.4, 'genuinely': 1.4, 'utterly': 1.8, 'thoroughly': 1.5
        }
        
        # Diminishers
        self.diminishers = {
            'somewhat': 0.7, 'slightly': 0.6, 'kind of': 0.7, 'sort of': 0.7,
            'a bit': 0.8, 'a little': 0.8, 'rather': 0.8, 'fairly': 0.8,
            'moderately': 0.7, 'mildly': 0.6, 'just': 0.8
        }
        
        # Fixed double negation patterns
        self.double_negative_patterns = {
            'not bad': 1.8,        # Clearly positive
            'not terrible': 1.5,   # Positive
            'not awful': 1.5,      # Positive  
            'not horrible': 1.5,   # Positive
            'not boring': 1.3,     # Slightly positive
            'not disappointing': 1.4, # Positive
            'not good': -1.8,      # Clearly negative
            'not great': -1.5,     # Negative
            'not excellent': -1.2, # Negative
            'not amazing': -1.0,   # Slightly negative
            'not like': -2.0,      # Negative
            'not love': -2.2,      # Negative
            "don't like": -2.5,    # Strong negative
            "don't love": -2.8,    # Strong negative
            "didn't like": -2.3,   # Strong negative
            "didn't enjoy": -2.0,  # Negative
            "doesn't like": -2.4,  # Strong negative
            "won't recommend": -2.0 # Negative
        }
        
        # Contrast indicators
        self.contrast_words = {
            'but', 'however', 'although', 'though', 'yet', 'nevertheless',
            'nonetheless', 'while', 'whereas', 'except', 'despite', 'still'
        }
        
        try:
            self.lemmatizer = WordNetLemmatizer()
            self.stop_words = set(stopwords.words('english')) - self.negation_words
            # Keep sentiment words
            sentiment_words = set(self.sentiment_lexicon.keys())
            self.stop_words = self.stop_words - sentiment_words - set(self.intensifiers.keys())
        except:
            self.lemmatizer = None
            self.stop_words = {'the', 'a', 'an', 'and', 'or', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
    
    def handle_contractions(self, text):
        """Handle contractions properly"""
        contractions = {
            "won't": "will not", "can't": "cannot", "n't": " not",
            "'re": " are", "'ve": " have", "'ll": " will", "'d": " would", 
            "'m": " am", "it's": "it is", "that's": "that is",
            "don't": "do not", "doesn't": "does not", "didn't": "did not",
            "haven't": "have not", "hasn't": "has not", "hadn't": "had not",
            "wouldn't": "would not", "shouldn't": "should not", "couldn't": "could not",
            "mustn't": "must not", "isn't": "is not", "aren't": "are not", 
            "wasn't": "was not", "weren't": "were not"
        }
        for contraction, expansion in contractions.items():
            text = text.replace(contraction, expansion)
        return text
    
    def calculate_sentiment_score(self, text):
        """Calculate sentiment score with improved negative detection"""
        text_lower = text.lower()
        
        # Check for double negation patterns first (highest priority)
        for pattern, score in self.double_negative_patterns.items():
            if pattern in text_lower:
                return score, f"Double negation: '{pattern}'"
        
        words = text_lower.split()
        total_score = 0
        word_count = 0
        context_notes = []
        
        i = 0
        while i < len(words):
            word = words[i]
            
            if word in self.sentiment_lexicon:
                base_score = self.sentiment_lexicon[word]
                multiplier = 1.0
                
                # Check for intensifiers
                if i > 0 and words[i-1] in self.intensifiers:
                    multiplier = self.intensifiers[words[i-1]]
                    context_notes.append(f"Intensifier: {words[i-1]} {word}")
                
                # Check for diminishers
                elif i > 0 and words[i-1] in self.diminishers:
                    multiplier = self.diminishers[words[i-1]]
                    context_notes.append(f"Diminisher: {words[i-1]} {word}")
                
                # Check for negation (look back up to 3 words)
                negated = False
                negation_word = None
                for j in range(max(0, i-3), i):
                    if words[j] in self.negation_words:
                        negated = True
                        negation_word = words[j]
                        break
                
                # Apply negation and multiplier
                if negated:
                    final_score = -base_score * multiplier
                    context_notes.append(f"Negation: {negation_word} ... {word}")
                else:
                    final_score = base_score * multiplier
                
                total_score += final_score
                word_count += 1
            
            i += 1
        
        # Check for contrast (reduces confidence in extreme scores)
        has_contrast = any(contrast in text_lower for contrast in self.contrast_words)
        if has_contrast:
            context_notes.append("Contrast detected")
            total_score *= 0.8  # Dampen the score
        
        # Normalize score
        if word_count > 0:
            normalized_score = total_score / word_count
        else:
            normalized_score = 0
        
        return normalized_score, "; ".join(context_notes)
    
    def preprocess_text(self, text):
        """Preprocess text while preserving sentiment-relevant features"""
        try:
            # Handle contractions first
            text = self.handle_contractions(text)
            text = text.lower().strip()
            
            # Remove HTML tags but keep the text
            text = re.sub(r'<.*?>', ' ', text)
            
            # Replace multiple punctuation with single space
            text = re.sub(r'[^\w\s]', ' ', text)
            text = re.sub(r'\s+', ' ', text)
            
            # Tokenize
            try:
                tokens = word_tokenize(text) if 'word_tokenize' in dir() else text.split()
            except:
                tokens = text.split()
            
            # Filter and lemmatize while keeping sentiment words
            if self.lemmatizer:
                processed_tokens = []
                for token in tokens:
                    if len(token) > 1:
                        # Keep negations, sentiment words, and intensifiers
                        if (token in self.negation_words or 
                            token in self.sentiment_lexicon or 
                            token in self.intensifiers or
                            token not in self.stop_words):
                            lemmatized = self.lemmatizer.lemmatize(token)
                            processed_tokens.append(lemmatized)
                tokens = processed_tokens
            else:
                tokens = [token for token in tokens if len(token) > 1 and token not in self.stop_words]
            
            return ' '.join(tokens)
            
        except Exception as e:
            return text.lower()

class FixedThreeClassSentimentAnalyzer:
    def __init__(self):
        self.processor = FixedTextProcessor()
        self.vectorizer = None
        self.model = None
        self.is_trained = False
        self.label_names = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    
    def create_enhanced_features(self, texts):
        """Create enhanced features for the model"""
        processed_texts = []
        additional_features = []
        
        for text in texts:
            try:
                processed = self.processor.preprocess_text(text)
                processed_texts.append(processed)
                
                # Calculate rule-based sentiment score
                sentiment_score, _ = self.processor.calculate_sentiment_score(text)
                
                # Extract additional features
                features = [
                    sentiment_score,                    # Rule-based sentiment score
                    len(text.split()),                  # Word count
                    len(text),                          # Character count
                    text.count('!'),                    # Exclamation marks
                    text.count('?'),                    # Question marks
                    len([w for w in text.split() if w.isupper()]),  # Uppercase words
                    text.lower().count('very'),         # Intensifier count
                    text.lower().count('not'),          # Negation count
                    1 if any(contrast in text.lower() for contrast in self.processor.contrast_words) else 0,  # Has contrast
                    len([w for w in text.lower().split() if w in self.processor.sentiment_lexicon]),  # Sentiment words
                    text.lower().count('but'),          # But count (contrast indicator)
                    text.lower().count('however'),      # However count
                ]
                additional_features.append(features)
                
            except Exception as e:
                processed_texts.append(text.lower())
                additional_features.append([0] * 12)
        
        return processed_texts, np.array(additional_features)
    
    def train(self, texts, labels):
        """Train the fixed model with better architecture"""
        # Check label distribution
        unique_labels, counts = np.unique(labels, return_counts=True)
        
        # Create enhanced features
        processed_texts, additional_features = self.create_enhanced_features(texts)
        
        # TF-IDF with optimized parameters
        self.vectorizer = TfidfVectorizer(
            max_features=12000,
            ngram_range=(1, 2),  # Unigrams and bigrams
            min_df=3,
            max_df=0.8,
            sublinear_tf=True,
            strip_accents='unicode',
            lowercase=True
        )
        
        tfidf_features = self.vectorizer.fit_transform(processed_texts)
        
        # Combine TF-IDF and additional features
        additional_sparse = csr_matrix(additional_features)
        X = hstack([tfidf_features, additional_sparse])
        
        # Use ensemble of models for better performance
        logistic = LogisticRegression(
            random_state=42, 
            max_iter=1000, 
            class_weight='balanced',
            C=1.0,
            solver='liblinear'
        )
        
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced_subsample',
            n_jobs=-1
        )
        
        # Voting classifier
        self.model = VotingClassifier(
            estimators=[('lr', logistic), ('rf', rf)],
            voting='soft'
        )
        
        self.model.fit(X, labels)
        self.is_trained = True
    
    def predict(self, text):
        """Enhanced prediction with better rule-based integration"""
        if not self.is_trained:
            raise ValueError("Model not trained!")
        
        try:
            # Rule-based analysis
            sentiment_score, context_info = self.processor.calculate_sentiment_score(text)
            
            # Strong rule-based decisions for very clear cases
            if abs(sentiment_score) > 2.5:  # Very strong sentiment
                if sentiment_score > 0:
                    prediction = 2  # Positive
                    confidence = min(0.95, 0.8 + abs(sentiment_score) * 0.05)
                else:
                    prediction = 0  # Negative
                    confidence = min(0.95, 0.8 + abs(sentiment_score) * 0.05)
                
                probabilities = [0.05, 0.05, 0.9] if prediction == 2 else [0.9, 0.05, 0.05]
                
                return {
                    'original_text': text,
                    'prediction': prediction,
                    'sentiment': self.label_names[prediction],
                    'confidence': confidence,
                    'probabilities': {
                        'negative': probabilities[0],
                        'neutral': probabilities[1], 
                        'positive': probabilities[2]
                    },
                    'context_notes': [f"Strong rule-based decision (score: {sentiment_score:.2f})", context_info],
                    'rule_based_score': sentiment_score
                }
            
            # Use ML model for other cases
            processed_texts, additional_features = self.create_enhanced_features([text])
            tfidf_features = self.vectorizer.transform(processed_texts)
            
            additional_sparse = csr_matrix(additional_features)
            X = hstack([tfidf_features, additional_sparse])
            
            # Get ML predictions
            ml_prediction = self.model.predict(X)[0]
            ml_probabilities = self.model.predict_proba(X)[0]
            
            # Adjust predictions based on rule-based score
            adjusted_probs = ml_probabilities.copy()
            
            if abs(sentiment_score) > 1.0:  # Moderate rule-based signal
                adjustment = min(0.3, abs(sentiment_score) * 0.15)
                
                if sentiment_score > 0:  # Rule suggests positive
                    adjusted_probs[2] += adjustment
                    adjusted_probs[0] = max(0.05, adjusted_probs[0] - adjustment/2)
                    adjusted_probs[1] = max(0.05, adjusted_probs[1] - adjustment/2)
                else:  # Rule suggests negative
                    adjusted_probs[0] += adjustment
                    adjusted_probs[2] = max(0.05, adjusted_probs[2] - adjustment/2)
                    adjusted_probs[1] = max(0.05, adjusted_probs[1] - adjustment/2)
                
                # Renormalize
                adjusted_probs = adjusted_probs / adjusted_probs.sum()
                final_prediction = np.argmax(adjusted_probs)
            else:
                final_prediction = ml_prediction
                adjusted_probs = ml_probabilities
            
            return {
                'original_text': text,
                'prediction': final_prediction,
                'sentiment': self.label_names[final_prediction],
                'confidence': max(adjusted_probs),
                'probabilities': {
                    'negative': adjusted_probs[0],
                    'neutral': adjusted_probs[1], 
                    'positive': adjusted_probs[2]
                },
                'context_notes': [f"Hybrid ML+Rule prediction (rule score: {sentiment_score:.2f})", context_info] if context_info else [f"Hybrid ML+Rule prediction (rule score: {sentiment_score:.2f})"],
                'rule_based_score': sentiment_score
            }
            
        except Exception as e:
            return self.fallback_prediction(text)
    
    def fallback_prediction(self, text):
        """Fallback rule-based prediction"""
        sentiment_score, context = self.processor.calculate_sentiment_score(text)
        
        if sentiment_score > 0.8:
            prediction, sentiment_label = 2, 'Positive'
        elif sentiment_score < -0.8:
            prediction, sentiment_label = 0, 'Negative'
        else:
            prediction, sentiment_label = 1, 'Neutral'
        
        return {
            'original_text': text,
            'prediction': prediction,
            'sentiment': sentiment_label,
            'confidence': 0.7,
            'probabilities': {'negative': 0.2, 'neutral': 0.6, 'positive': 0.2},
            'context_notes': [f'Fallback rule-based (score: {sentiment_score:.2f})'],
            'rule_based_score': sentiment_score
        }

# Initialize NLTK downloads
@st.cache_resource
def setup_nltk():
    """Download NLTK resources"""
    nltk_downloads = ['punkt', 'punkt_tab', 'stopwords', 'wordnet', 'omw-1.4']
    for item in nltk_downloads:
        try:
            nltk.download(item, quiet=True)
        except:
            pass
    return True

@st.cache_resource
def load_and_train_model(subset_size=2000):
    """Load dataset and train model with caching"""
    setup_nltk()
    
    loader = IMDBDatasetLoader()
    train_texts, train_labels, test_texts, test_labels = loader.load_imdb_data(subset_size=subset_size)
    
    if train_texts is None:
        return None, None
        
    analyzer = FixedThreeClassSentimentAnalyzer()
    analyzer.train(train_texts, train_labels)
    
    return analyzer, (test_texts, test_labels)

def create_probability_chart(probabilities):
    """Create a horizontal bar chart for sentiment probabilities"""
    sentiments = ['Negative', 'Neutral', 'Positive']
    values = [probabilities['negative'], probabilities['neutral'], probabilities['positive']]
    colors = ['#ff4444', '#ffaa00', '#44ff44']
    
    fig = go.Figure(go.Bar(
        x=values,
        y=sentiments,
        orientation='h',
        marker_color=colors,
        text=[f'{v:.1%}' for v in values],
        textposition='inside'
    ))
    
    fig.update_layout(
        title="Sentiment Probability Distribution",
        xaxis_title="Probability",
        yaxis_title="Sentiment",
        height=300,
        showlegend=False,
        xaxis=dict(range=[0, 1], tickformat='.0%')
    )
    
    return fig

def create_confidence_gauge(confidence):
    """Create a gauge chart for confidence level"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = confidence * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Confidence Level"},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 80], 'color': "yellow"},
                {'range': [80, 100], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(height=300)
    return fig

def display_result_streamlit(result):
    """Display prediction results in Streamlit format"""
    sentiment = result['sentiment']
    confidence = result['confidence']
    
    # Color coding based on sentiment
    if sentiment == 'Positive':
        sentiment_color = 'sentiment-positive'
        emoji = '😊'
    elif sentiment == 'Negative':
        sentiment_color = 'sentiment-negative'
        emoji = '😔'
    else:
        sentiment_color = 'sentiment-neutral'
        emoji = '😐'
    
    # Confidence color coding
    if confidence > 0.8:
        conf_color = 'confidence-high'
        conf_text = "Very Confident"
    elif confidence > 0.6:
        conf_color = 'confidence-medium'
        conf_text = "Confident"
    else:
        conf_color = 'confidence-low'
        conf_text = "Low Confidence"
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(f"## {emoji} Predicted Sentiment: <span class='{sentiment_color}'>{sentiment}</span>", 
                   unsafe_allow_html=True)
        st.markdown(f"**<span class='{conf_color}'>Confidence: {confidence:.1%} ({conf_text})</span>**", 
                   unsafe_allow_html=True)
        
        if 'rule_based_score' in result:
            score = result['rule_based_score']
            st.markdown(f"**Rule-based Score:** {score:+.2f}")
        
        if result.get('context_notes'):
            with st.expander("Analysis Details"):
                for note in result['context_notes']:
                    if note:  # Only display non-empty notes
                        st.write(f"• {note}")
    
    with col2:
        # Confidence gauge
        fig_gauge = create_confidence_gauge(confidence)
        st.plotly_chart(fig_gauge, use_container_width=True)
    
    # Probability chart
    fig_probs = create_probability_chart(result['probabilities'])
    st.plotly_chart(fig_probs, use_container_width=True)

def run_demo_tests(analyzer):
    """Run demonstration tests"""
    demo_cases = [
        ("very bad", "Negative"),
        ("i dont like it", "Negative"), 
        ("terrible movie", "Negative"),
        ("I hate this film", "Negative"),
        ("not bad", "Positive"),
        ("not terrible", "Positive"),
        ("excellent movie", "Positive"),
        ("I love this", "Positive"),
        ("okay film", "Neutral"),
        ("average movie", "Neutral"),
        ("good but boring", "Neutral"),
        ("really terrible", "Negative"),
        ("extremely good", "Positive")
    ]
    
    results = []
    for text, expected in demo_cases:
        try:
            result = analyzer.predict(text)
            predicted = result['sentiment']
            confidence = result['confidence']
            rule_score = result.get('rule_based_score', 0)
            is_correct = predicted == expected
            
            results.append({
                'Text': text,
                'Expected': expected,
                'Predicted': predicted,
                'Correct': '✓' if is_correct else '✗',
                'Confidence': f"{confidence:.3f}",
                'Rule Score': f"{rule_score:+.2f}"
            })
        except:
            results.append({
                'Text': text,
                'Expected': expected,
                'Predicted': 'ERROR',
                'Correct': '✗',
                'Confidence': 'N/A',
                'Rule Score': 'N/A'
            })
    
    df = pd.DataFrame(results)
    st.dataframe(df, use_container_width=True)
    
    correct_count = sum(1 for r in results if r['Correct'] == '✓')
    accuracy = correct_count / len(results)
    st.metric("Demo Accuracy", f"{accuracy:.1%}", f"{correct_count}/{len(results)}")

# Main Streamlit App
def main():
    st.markdown('<h1 class="main-header">🎬 IMDB Sentiment Analyzer</h1>', unsafe_allow_html=True)
    st.markdown("### Analyze movie review sentiment with advanced ML + rule-based hybrid approach")
    
    # Sidebar
    st.sidebar.header("Settings")
    
    # Model loading
    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = False
        st.session_state.analyzer = None
    
    subset_size = st.sidebar.selectbox(
        "Dataset Size (for training)",
        [500, 1000, 2000, 5000],
        index=2,
        help="Smaller size = faster loading, larger size = better accuracy"
    )
    
    if not st.session_state.model_loaded:
        if st.sidebar.button("Load & Train Model", type="primary"):
            with st.spinner("Loading dataset and training model... This may take several minutes."):
                analyzer, test_data = load_and_train_model(subset_size)
                
                if analyzer is not None:
                    st.session_state.analyzer = analyzer
                    st.session_state.test_data = test_data
                    st.session_state.model_loaded = True
                    st.sidebar.success("Model trained successfully!")
                    st.rerun()
                else:
                    st.sidebar.error("Failed to load model. Please try again.")
    else:
        st.sidebar.success("✅ Model loaded and ready!")
        
        if st.sidebar.button("Reset Model"):
            st.session_state.model_loaded = False
            st.session_state.analyzer = None
            st.rerun()
    
    # Main interface
    if st.session_state.model_loaded and st.session_state.analyzer is not None:
        
        # Tabs for different functionalities
        tab1, tab2, tab3 = st.tabs(["🎭 Analyze Review", "🧪 Demo Tests", "📊 About"])
        
        with tab1:
            st.header("Analyze Movie Review")
            
            # Text input methods
            input_method = st.radio(
                "Choose input method:",
                ["Type Review", "Select Example"],
                horizontal=True
            )
            
            if input_method == "Type Review":
                user_input = st.text_area(
                    "Enter your movie review:",
                    height=150,
                    placeholder="Type your movie review here... e.g., 'This movie was absolutely fantastic! Great acting and storyline.'"
                )
            else:
                example_reviews = [
                    "This movie was absolutely terrible. Worst film I've ever seen!",
                    "Not bad, but nothing special. Just an average film.",
                    "Fantastic movie! Amazing acting and brilliant storyline.",
                    "I don't like this movie at all. Very disappointing.",
                    "Pretty good film, though it has some issues.",
                    "Extremely boring and confusing plot. Waste of time.",
                    "Not terrible, but not great either. So-so performance.",
                    "I love this movie! Best film of the year!",
                    "Okay movie, nothing to write home about.",
                    "Very bad acting and horrible script. Awful movie."
                ]
                
                user_input = st.selectbox(
                    "Select an example review:",
                    example_reviews
                )
            
            if st.button("Analyze Sentiment", type="primary", disabled=not user_input):
                if user_input and len(user_input.strip()) > 2:
                    try:
                        with st.spinner("Analyzing sentiment..."):
                            result = st.session_state.analyzer.predict(user_input)
                        
                        st.success("Analysis complete!")
                        
                        # Display original text
                        with st.expander("Original Review", expanded=False):
                            st.write(user_input)
                        
                        # Display results
                        display_result_streamlit(result)
                        
                    except Exception as e:
                        st.error(f"Analysis error: {str(e)}")
                else:
                    st.warning("Please enter at least 3 characters.")
        
        with tab2:
            st.header("Demonstration Tests")
            st.write("Test the model on various challenging cases to see how well it handles different types of sentiment.")
            
            if st.button("Run Demo Tests", type="primary"):
                with st.spinner("Running demonstration tests..."):
                    run_demo_tests(st.session_state.analyzer)
        
        with tab3:
            st.header("About This Application")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🎯 Features")
                st.write("""
                - **Three-class sentiment analysis**: Negative, Neutral, Positive
                - **Hybrid approach**: Combines ML models with rule-based logic
                - **Enhanced negation handling**: Properly detects "don't like", "not bad", etc.
                - **IMDB dataset training**: Trained on real movie reviews
                - **Confidence scoring**: Shows how confident the model is
                """)
                
                st.subheader("🔧 Technical Details")
                st.write("""
                - **Models**: Ensemble of Logistic Regression + Random Forest
                - **Features**: TF-IDF + linguistic features + sentiment scores
                - **Dataset**: IMDB movie reviews (25k training samples)
                - **Preprocessing**: Advanced text cleaning and lemmatization
                """)
            
            with col2:
                st.subheader("✅ Improvements Made")
                st.write("""
                - Fixed negative sentiment detection
                - Better handling of contractions ("don't", "can't")
                - Improved double negation logic ("not bad" → positive)
                - Enhanced rule-based sentiment scoring
                - Proper intensifier handling ("very bad", "really good")
                - Context-aware analysis (contrast detection)
                """)
                
                st.subheader("📈 Model Performance")
                st.info("""
                The model combines machine learning with linguistic rules to achieve 
                better accuracy on challenging cases like negations and complex sentiment expressions.
                """)
    
    else:
        # Welcome screen
        st.info("👆 Please click 'Load & Train Model' in the sidebar to get started.")
        
        st.header("Welcome to IMDB Sentiment Analyzer!")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### 😔 Negative")
            st.write("Detects negative sentiment in movie reviews")
            st.code("'This movie was terrible!'")
        
        with col2:
            st.markdown("### 😐 Neutral") 
            st.write("Identifies neutral or mixed sentiment")
            st.code("'It was okay, nothing special.'")
        
        with col3:
            st.markdown("### 😊 Positive")
            st.write("Recognizes positive sentiment")
            st.code("'Amazing film! Loved it!'")
        
        st.markdown("---")
        st.markdown("### 🚀 Getting Started")
        st.write("1. Click **'Load & Train Model'** in the sidebar")
        st.write("2. Wait for the model to download the IMDB dataset and train")
        st.write("3. Start analyzing movie reviews!")

if __name__ == "__main__":
    main()