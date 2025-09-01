#is good but nuetral problem



# ============================
# Complete IMDB Three-Class Sentiment Analysis System
# Supports Positive/Negative/Neutral Classification with Real IMDB Dataset
# ============================

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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from scipy.sparse import hstack, csr_matrix

warnings.filterwarnings('ignore')

# ============================
# 1. Dataset Download and Processing
# ============================

def download_imdb_dataset():
    """Download and extract Stanford IMDB dataset"""
    url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    dataset_path = "aclImdb_v1.tar.gz"
    
    if not os.path.exists(dataset_path):
        print("📥 Downloading IMDB dataset...")
        urllib.request.urlretrieve(url, dataset_path)
        print("✅ Dataset downloaded!")
    
    if not os.path.exists("aclImdb"):
        print("📦 Extracting dataset...")
        with tarfile.open(dataset_path, "r:gz") as tar:
            tar.extractall()
        print("✅ Dataset extracted!")

def load_imdb_data(data_dir):
    """Load IMDB data from directories"""
    reviews = []
    labels = []
    
    for label_type in ["pos", "neg"]:
        folder = os.path.join(data_dir, label_type)
        if not os.path.exists(folder):
            continue
            
        for filename in os.listdir(folder):
            if filename.endswith(".txt"):
                try:
                    with open(os.path.join(folder, filename), encoding="utf-8", errors='ignore') as f:
                        content = f.read()
                        reviews.append(content)
                        labels.append(1 if label_type == "pos" else 0)
                except Exception as e:
                    print(f"Error reading {filename}: {e}")
                    continue
                    
    return reviews, labels

def create_three_class_from_binary(reviews, binary_labels, neutral_ratio=0.2):
    """Convert binary sentiment data to three-class by creating neutral samples"""
    
    # Separate positive and negative reviews
    pos_reviews = [reviews[i] for i in range(len(reviews)) if binary_labels[i] == 1]
    neg_reviews = [reviews[i] for i in range(len(reviews)) if binary_labels[i] == 0]
    
    # Create neutral samples by mixing and modifying existing reviews
    neutral_reviews = []
    neutral_patterns = [
        "This movie is okay. {}",
        "It's alright. {}",
        "The movie is so-so. {}",
        "Mixed feelings about this film. {}",
        "It's decent but nothing special. {}",
        "Average movie. {}",
        "It's fine, I guess. {}",
        "The film is mediocre. {}",
        "Nothing extraordinary. {}",
        "Standard movie. {}"
    ]
    
    # Take samples from both positive and negative to create neutral
    sample_size = min(len(pos_reviews), len(neg_reviews))
    neutral_size = int(sample_size * neutral_ratio)
    
    # Create neutral by combining elements
    for i in range(neutral_size):
        if i % 2 == 0 and i < len(pos_reviews):
            # Take positive review and make it neutral
            original = pos_reviews[i]
            # Extract middle sentences to make it more neutral
            sentences = original.split('. ')
            if len(sentences) > 2:
                middle_part = '. '.join(sentences[1:3])
                pattern = np.random.choice(neutral_patterns)
                neutral_text = pattern.format(middle_part)
            else:
                neutral_text = f"This movie is okay. {original[:100]}..."
        else:
            # Take negative review and make it neutral
            idx = i % len(neg_reviews)
            original = neg_reviews[idx]
            sentences = original.split('. ')
            if len(sentences) > 2:
                middle_part = '. '.join(sentences[1:3])
                pattern = np.random.choice(neutral_patterns)
                neutral_text = pattern.format(middle_part)
            else:
                neutral_text = f"The movie is average. {original[:100]}..."
                
        neutral_reviews.append(neutral_text)
    
    # Combine all reviews
    all_reviews = pos_reviews + neg_reviews + neutral_reviews
    all_labels = ([2] * len(pos_reviews) + 
                 [0] * len(neg_reviews) + 
                 [1] * len(neutral_reviews))  # 0:negative, 1:neutral, 2:positive
    
    return all_reviews, all_labels

# ============================
# 2. Advanced Text Processor (Three-Class)
# ============================

class ThreeClassTextProcessor:
    def __init__(self):
        # Negation words
        self.negation_words = {
            'not', 'no', 'never', 'nothing', 'nowhere', 'nobody', 'none',
            'neither', 'nor', 'cannot', "can't", "won't", "shouldn't", 
            "wouldn't", "couldn't", "doesn't", "don't", "isn't", "aren't",
            "wasn't", "weren't", "hasn't", "haven't", "hadn't", "mustn't",
            "needn't", "without", "hardly", "barely", "scarcely"
        }
        
        # Intensifiers
        self.intensifiers = {
            'very', 'extremely', 'really', 'quite', 'rather', 'pretty',
            'absolutely', 'completely', 'totally', 'entirely', 'perfectly',
            'incredibly', 'amazingly', 'surprisingly', 'remarkably', 'highly',
            'deeply', 'truly', 'genuinely', 'utterly', 'thoroughly'
        }
        
        # Contrast words
        self.contrast_words = {
            'but', 'however', 'although', 'though', 'yet', 'nevertheless',
            'nonetheless', 'while', 'whereas', 'except', 'despite', 'still'
        }
        
        # Strong positive words
        self.strong_positive_words = {
            'amazing', 'excellent', 'fantastic', 'wonderful', 'brilliant',
            'outstanding', 'superb', 'magnificent', 'awesome', 'perfect',
            'incredible', 'phenomenal', 'extraordinary', 'exceptional',
            'love', 'adore', 'best', 'greatest', 'masterpiece', 'flawless'
        }
        
        # Mild positive words
        self.mild_positive_words = {
            'good', 'nice', 'okay', 'fine', 'decent', 'pleasant', 'enjoyable',
            'satisfying', 'solid', 'reasonable', 'acceptable', 'fair',
            'like', 'enjoy', 'appreciate', 'recommend', 'positive'
        }
        
        # Neutral words
        self.neutral_words = {
            'average', 'ordinary', 'typical', 'standard', 'normal', 'regular',
            'usual', 'common', 'so-so', 'mediocre', 'mixed', 'varies',
            'depends', 'different', 'interesting', 'unique', 'strange',
            'weird', 'unusual', 'particular', 'specific', 'moderate'
        }
        
        # Strong negative words
        self.strong_negative_words = {
            'terrible', 'awful', 'horrible', 'disgusting', 'pathetic',
            'worst', 'hate', 'despise', 'abysmal', 'atrocious', 'dreadful',
            'appalling', 'horrendous', 'catastrophic', 'disastrous'
        }
        
        # Mild negative words
        self.mild_negative_words = {
            'bad', 'poor', 'weak', 'disappointing', 'boring', 'dull',
            'uninteresting', 'slow', 'confusing', 'unclear', 'difficult',
            'problematic', 'flawed', 'limited', 'lacking', 'negative'
        }
        
        # Double negation handling
        self.double_negative_phrases = {
            'not bad': 1,  # Slightly positive -> neutral
            'not terrible': 1,
            'not awful': 1,
            'not horrible': 1,
            'not worst': 1,
            'not boring': 1,
            'not disappointing': 1,
            'cannot complain': 1,
            'not good': 0,  # Negative
            'not great': 0,
            'not excellent': 0,
            'not amazing': 0,
            'not perfect': 0
        }
        
        try:
            self.lemmatizer = WordNetLemmatizer()
            self.stop_words = set(stopwords.words('english')) - self.negation_words - self.intensifiers - self.contrast_words
        except:
            self.lemmatizer = None
            self.stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
    
    def handle_contractions(self, text):
        """Handle contractions"""
        contractions = {
            "won't": "will not", "can't": "cannot", "n't": " not",
            "'re": " are", "'ve": " have", "'ll": " will", "'d": " would", 
            "'m": " am", "it's": "it is", "that's": "that is"
        }
        for contraction, expansion in contractions.items():
            text = text.replace(contraction, expansion)
        return text
    
    def detect_sentiment_signals(self, text):
        """Detect sentiment signals for three-class classification"""
        text_lower = text.lower()
        
        strong_pos_count = sum(1 for word in self.strong_positive_words if word in text_lower)
        mild_pos_count = sum(1 for word in self.mild_positive_words if word in text_lower)
        neutral_count = sum(1 for word in self.neutral_words if word in text_lower)
        mild_neg_count = sum(1 for word in self.mild_negative_words if word in text_lower)
        strong_neg_count = sum(1 for word in self.strong_negative_words if word in text_lower)
        
        return {
            'strong_positive': strong_pos_count,
            'mild_positive': mild_pos_count,
            'neutral': neutral_count,
            'mild_negative': mild_neg_count,
            'strong_negative': strong_neg_count
        }
    
    def process_negation_context(self, tokens):
        """Process negation context"""
        processed_tokens = []
        negation_scope = 0
        
        for i, token in enumerate(tokens):
            if token in self.negation_words:
                processed_tokens.append(token)
                negation_scope = min(3, len(tokens) - i - 1)
            elif negation_scope > 0:
                if (token in self.strong_positive_words or 
                    token in self.mild_positive_words or
                    token in self.mild_negative_words or 
                    token in self.strong_negative_words):
                    processed_tokens.append(f"NOT_{token}")
                else:
                    processed_tokens.append(token)
                negation_scope -= 1
            else:
                processed_tokens.append(token)
        
        return processed_tokens
    
    def advanced_preprocess(self, text):
        """Advanced preprocessing main function"""
        try:
            # Basic cleaning
            text = text.lower().strip()
            text = self.handle_contractions(text)
            text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
            
            # Handle double negations
            for phrase, _ in self.double_negative_phrases.items():
                if phrase in text:
                    text = text.replace(phrase, f"DOUBLE_NEG_{phrase.replace(' ', '_')}")
            
            # Clean special characters
            text = re.sub(r'[^a-zA-Z\s_]', ' ', text)
            
            # Tokenize
            try:
                tokens = word_tokenize(text)
            except:
                tokens = text.split()
            
            # Remove stop words
            if self.lemmatizer:
                tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                         if token not in self.stop_words and len(token) > 1]
            else:
                tokens = [token for token in tokens 
                         if token not in self.stop_words and len(token) > 1]
            
            # Process negation context
            tokens = self.process_negation_context(tokens)
            
            return ' '.join(tokens)
            
        except Exception as e:
            print(f"Preprocessing error: {e}")
            return text.lower()

# ============================
# 3. Three-Class Sentiment Analyzer
# ============================

class ThreeClassSentimentAnalyzer:
    def __init__(self):
        self.processor = ThreeClassTextProcessor()
        self.vectorizer = None
        self.model = None
        self.is_trained = False
        self.label_names = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    
    def create_features(self, texts):
        """Create features for three-class classification"""
        processed_texts = []
        additional_features = []
        
        for text in texts:
            try:
                processed = self.processor.advanced_preprocess(text)
                processed_texts.append(processed)
                
                # Extract sentiment signal features
                sentiment_signals = self.processor.detect_sentiment_signals(text)
                features = [
                    sentiment_signals['strong_positive'],
                    sentiment_signals['mild_positive'], 
                    sentiment_signals['neutral'],
                    sentiment_signals['mild_negative'],
                    sentiment_signals['strong_negative'],
                    len(text.split()),  # Text length
                    text.count('!'),    # Exclamation marks
                    text.count('?'),    # Question marks
                    text.count('.'),    # Periods
                    len([w for w in text.split() if w.isupper()]),  # Uppercase words
                ]
                additional_features.append(features)
                
            except Exception as e:
                print(f"Feature extraction error: {e}")
                processed_texts.append(text.lower())
                additional_features.append([0] * 10)  # Default features
        
        return processed_texts, np.array(additional_features)
    
    def train(self, texts, labels):
        """Train three-class model"""
        print("🚀 Training three-class sentiment analysis model...")
        
        try:
            # Check label distribution
            unique_labels, counts = np.unique(labels, return_counts=True)
            print(f"Label distribution: {dict(zip(unique_labels, counts))}")
            
            # Create features
            processed_texts, additional_features = self.create_features(texts)
            
            # TF-IDF features
            self.vectorizer = TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.9,
                sublinear_tf=True
            )
            
            tfidf_features = self.vectorizer.fit_transform(processed_texts)
            
            # Combine features
            additional_sparse = csr_matrix(additional_features)
            X = hstack([tfidf_features, additional_sparse])
            
            print(f"Feature matrix shape: {X.shape}")
            
            # Train multiclass model
            self.model = LogisticRegression(
                random_state=42,
                max_iter=1000,
                multi_class='ovr',  # One-vs-Rest for multiclass
                class_weight='balanced'
            )
            
            self.model.fit(X, labels)
            self.is_trained = True
            print("✅ Three-class model training completed!")
            
        except Exception as e:
            print(f"Training failed: {e}")
            raise
    
    def predict(self, text):
        """Predict sentiment (three-class)"""
        if not self.is_trained:
            raise ValueError("Model not trained!")
        
        try:
            # Apply quick rules
            quick_result = self.apply_quick_rules(text)
            if quick_result['override']:
                return quick_result['result']
            
            # Feature extraction
            processed_texts, additional_features = self.create_features([text])
            tfidf_features = self.vectorizer.transform(processed_texts)
            
            additional_sparse = csr_matrix(additional_features)
            X = hstack([tfidf_features, additional_sparse])
            
            # Predict
            prediction = self.model.predict(X)[0]
            probabilities = self.model.predict_proba(X)[0]
            
            # Context adjustment
            adjusted = self.adjust_for_context(text, prediction, probabilities)
            
            return {
                'original_text': text,
                'processed_text': processed_texts[0],
                'prediction': adjusted['prediction'],
                'sentiment': self.label_names[adjusted['prediction']],
                'confidence': adjusted['confidence'],
                'probabilities': {
                    'negative': adjusted['probabilities'][0],
                    'neutral': adjusted['probabilities'][1], 
                    'positive': adjusted['probabilities'][2]
                },
                'context_notes': adjusted['notes']
            }
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return self.default_prediction(text)
    
    def apply_quick_rules(self, text):
        """Apply quick classification rules"""
        text_lower = text.lower().strip()
        words = text_lower.split()
        
        # Single word quick judgment
        if len(words) == 1:
            word = words[0]
            if word in self.processor.strong_positive_words:
                return self.create_quick_result(text, 2, 0.85, "Strong positive word")
            elif word in self.processor.strong_negative_words:
                return self.create_quick_result(text, 0, 0.85, "Strong negative word")
            elif word in self.processor.neutral_words:
                return self.create_quick_result(text, 1, 0.75, "Neutral word")
        
        # Double negation check
        for phrase, label in self.processor.double_negative_phrases.items():
            if phrase in text_lower:
                return self.create_quick_result(text, label, 0.70, f"Double negation: '{phrase}'")
        
        # Special neutral phrases
        neutral_phrases = ['so so', 'so-so', 'meh', 'whatever', 'alright', 'okay i guess']
        if any(phrase in text_lower for phrase in neutral_phrases):
            return self.create_quick_result(text, 1, 0.70, "Clear neutral expression")
        
        return {'override': False, 'result': None}
    
    def create_quick_result(self, text, prediction, confidence, note):
        """Create quick judgment result"""
        probs = [0.33, 0.33, 0.34]
        probs[prediction] = confidence
        remaining = (1 - confidence) / 2
        for i in range(3):
            if i != prediction:
                probs[i] = remaining
        
        return {
            'override': True,
            'result': {
                'original_text': text,
                'processed_text': text.lower(),
                'prediction': prediction,
                'sentiment': self.label_names[prediction],
                'confidence': confidence,
                'probabilities': {
                    'negative': probs[0],
                    'neutral': probs[1],
                    'positive': probs[2]
                },
                'context_notes': [f"Quick rule: {note}"]
            }
        }
    
    def adjust_for_context(self, text, prediction, probabilities):
        """Adjust prediction based on context"""
        notes = []
        
        # Check double negations
        text_lower = text.lower()
        for phrase, correct_label in self.processor.double_negative_phrases.items():
            if phrase in text_lower:
                notes.append(f"Detected double negation: '{phrase}'")
                if prediction != correct_label:
                    new_probs = [0.2, 0.2, 0.6] if correct_label == 2 else ([0.6, 0.2, 0.2] if correct_label == 0 else [0.25, 0.5, 0.25])
                    return {
                        'prediction': correct_label,
                        'confidence': max(new_probs),
                        'probabilities': new_probs,
                        'notes': notes
                    }
        
        # Check contrast structures
        if any(word in text_lower for word in self.processor.contrast_words):
            notes.append("Detected contrast structure")
        
        # Check sentiment intensity
        sentiment_signals = self.processor.detect_sentiment_signals(text)
        total_strong = sentiment_signals['strong_positive'] + sentiment_signals['strong_negative']
        if total_strong > 0:
            notes.append(f"Detected {total_strong} strong sentiment words")
        
        return {
            'prediction': prediction,
            'confidence': max(probabilities),
            'probabilities': probabilities,
            'notes': notes
        }
    
    def default_prediction(self, text):
        """Default prediction result"""
        return {
            'original_text': text,
            'processed_text': text.lower(),
            'prediction': 1,  # Default to neutral
            'sentiment': 'Neutral',
            'confidence': 0.4,
            'probabilities': {'negative': 0.3, 'neutral': 0.4, 'positive': 0.3},
            'context_notes': ['Prediction error, returning default result']
        }

# ============================
# 4. Evaluation and Visualization
# ============================

def evaluate_three_class_model(analyzer, test_texts, test_labels):
    """Evaluate three-class model"""
    predictions = []
    confidences = []
    
    print("🔍 Evaluating model...")
    for i, text in enumerate(test_texts):
        if i % 1000 == 0:
            print(f"Progress: {i}/{len(test_texts)}")
        
        result = analyzer.predict(text)
        predictions.append(result['prediction'])
        confidences.append(result['confidence'])
    
    # Calculate accuracy
    accuracy = accuracy_score(test_labels, predictions)
    print(f"\n📊 Three-class accuracy: {accuracy:.4f}")
    
    # Classification report
    print("\n📈 Detailed classification report:")
    target_names = ['Negative', 'Neutral', 'Positive']
    print(classification_report(test_labels, predictions, target_names=target_names))
    
    # Confusion matrix
    cm = confusion_matrix(test_labels, predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names, yticklabels=target_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Three-Class Sentiment Analysis Confusion Matrix')
    plt.show()
    
    # Confidence distribution
    plt.figure(figsize=(10, 6))
    plt.hist(confidences, bins=30, alpha=0.7, edgecolor='black')
    plt.xlabel('Confidence Score')
    plt.ylabel('Frequency')
    plt.title('Distribution of Model Confidence Scores')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return accuracy, predictions

# ============================
# 5. Interactive System
# ============================

def interactive_system(analyzer):
    """Interactive three-class system"""
    print("\n" + "="*80)
    print("🎬 IMDB Three-Class Sentiment Analyzer")
    print("   Supports Positive/Negative/Neutral Classification!")
    print("="*80)
    print("📝 Commands: 'quit' to exit | 'test' for demo | 'help' for info")
    print("-" * 80)
    
    while True:
        user_input = input("\n🎭 Enter your movie review: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("👋 Thank you for using the three-class sentiment analyzer!")
            break
            
        elif user_input.lower() == 'test':
            run_demo_tests(analyzer)
            continue
            
        elif user_input.lower() == 'help':
            print_help()
            continue
            
        elif len(user_input.strip()) < 2:
            print("⚠️ Please enter at least 2 characters")
            continue
        
        try:
            result = analyzer.predict(user_input)
            display_result(result)
            
        except Exception as e:
            print(f"❌ Analysis error: {str(e)}")

def display_result(result):
    """Display three-class result"""
    print("\n" + "="*75)
    print("📊 Three-Class Analysis Result:")
    print("="*75)
    print(f"Original text: {result['original_text'][:100]}{'...' if len(result['original_text']) > 100 else ''}")
    
    if result['context_notes']:
        print(f"🔍 Context analysis: {'; '.join(result['context_notes'])}")
    
    print("-" * 75)
    print(f"🎭 Sentiment: {result['sentiment']}")
    print(f"📈 Confidence: {result['confidence']:.4f} ({result['confidence']*100:.2f}%)")
    print(f"📊 Detailed probabilities:")
    print(f"   😔 Negative: {result['probabilities']['negative']:.4f} ({result['probabilities']['negative']*100:.2f}%)")
    print(f"   😐 Neutral:  {result['probabilities']['neutral']:.4f} ({result['probabilities']['neutral']*100:.2f}%)")
    print(f"   😊 Positive: {result['probabilities']['positive']:.4f} ({result['probabilities']['positive']*100:.2f}%)")
    
    # Confidence assessment
    if result['confidence'] > 0.7:
        confidence_level = "Very confident"
        emoji = "🎯"
    elif result['confidence'] > 0.5:
        confidence_level = "Moderately confident"
        emoji = "👍"
    else:
        confidence_level = "Low confidence"
        emoji = "🤔"
    
    print(f"{emoji} Model is {confidence_level} in this prediction")

def run_demo_tests(analyzer):
    """Run demonstration tests"""
    print("\n🧪 Running demonstration tests...")
    
    test_cases = [
        # Positive tests
        ("This movie is absolutely amazing!", "Positive"),
        ("I love this film so much!", "Positive"),
        ("Excellent performance and great story!", "Positive"),
        
        # Negative tests
        ("This movie is terrible and boring", "Negative"),
        ("I hate this film completely", "Negative"),
        ("Awful acting and poor direction", "Negative"),
        
        # Neutral tests
        ("This movie is okay, nothing special", "Neutral"),
        ("It's alright, I guess", "Neutral"),
        ("The film is so-so", "Neutral"),
        ("Average movie with mixed feelings", "Neutral"),
        
        # Complex cases
        ("not bad actually", "Positive/Neutral"),
        ("not good at all", "Negative"),
        ("I like it but it's quite boring", "Neutral"),
        ("good movie but has some problems", "Neutral"),
        ("meh, whatever", "Neutral"),
    ]
    
    print(f"\n{'#':<3} {'Test Text':<40} {'Expected':<15} {'Predicted':<10} {'Conf':<6}")
    print("-" * 80)
    
    for i, (text, expected) in enumerate(test_cases, 1):
        try:
            result = analyzer.predict(text)
            sentiment_short = result['sentiment']
            
            print(f"{i:<3} {text:<40} {expected:<15} {sentiment_short:<10} {result['confidence']:.3f}")
            
        except Exception as e:
            print(f"{i:<3} {text:<40} {'Error':<15} {'N/A':<10}")

def print_help():
    """Display help information"""
    print("\n📚 Three-Class Sentiment Analyzer Help:")
    print("=" * 60)
    print("🎯 Supports three sentiment classes:")
    print("  • 😊 Positive: Expressing love, satisfaction, recommendation")
    print("  • 😐 Neutral: Expressing average, ordinary, mixed feelings")
    print("  • 😔 Negative: Expressing dislike, criticism, disappointment")
    print("\n🌟 Special handling capabilities:")
    print("  • Double negation: 'not bad' → Positive tendency")
    print("  • Contrast structures: 'good but boring' → Balanced analysis")
    print("  • Neutral expressions: 'so-so', 'meh', 'average' → Neutral")
    print("  • Intensity recognition: Distinguishes 'good' from 'amazing'")

# ============================
# 6. Main Program
# ============================

def main():
    """Main program"""
    print("🚀 Starting IMDB Three-Class Sentiment Analysis System...")
    
    try:
        # Download NLTK resources
        nltk_downloads = ['punkt', 'punkt_tab', 'stopwords', 'wordnet']
        for item in nltk_downloads:
            try:
                nltk.download(item, quiet=True)
            except:
                pass
        
        # Download and load IMDB dataset
        download_imdb_dataset()
        
        print("📊 Loading IMDB dataset...")
        train_reviews, train_labels = load_imdb_data("aclImdb/train")
        test_reviews, test_labels = load_imdb_data("aclImdb/test")
        
        print(f"Dataset loaded:")
        print(f"  Training samples: {len(train_reviews)}")
        print(f"  Test samples: {len(test_reviews)}")
        print(f"  Positive samples: {sum(train_labels) + sum(test_labels)}")
        print(f"  Negative samples: {len(train_labels) + len(test_labels) - sum(train_labels) - sum(test_labels)}")
        
        # Convert binary to three-class
        print("\n🔄 Converting binary sentiment to three-class...")
        all_reviews = train_reviews + test_reviews
        all_labels = train_labels + test_labels
        
        three_class_reviews, three_class_labels = create_three_class_from_binary(
            all_reviews, all_labels, neutral_ratio=0.15
        )
        
        # Split data
        train_texts, test_texts, train_labels, test_labels = train_test_split(
            three_class_reviews, three_class_labels, 
            test_size=0.2, random_state=42, stratify=three_class_labels
        )
        
        print(f"\nThree-class data statistics:")
        unique, counts = np.unique(three_class_labels, return_counts=True)
        label_names = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
        for label, count in zip(unique, counts):
            print(f"  {label_names[label]}: {count} samples")
        
        # Create and train analyzer
        analyzer = ThreeClassSentimentAnalyzer()
        
        # Use subset for faster training during demo
        train_subset = min(10000, len(train_texts))
        test_subset = min(2000, len(test_texts))
        
        print(f"\n🎯 Training on {train_subset} samples for demo...")
        analyzer.train(train_texts[:train_subset], train_labels[:train_subset])
        
        # Evaluate model
        print(f"\n📈 Evaluating on {test_subset} samples...")
        accuracy, predictions = evaluate_three_class_model(
            analyzer, test_texts[:test_subset], test_labels[:test_subset]
        )
        
        print(f"\n✅ Three-class system initialized successfully!")
        print(f"📊 Model accuracy: {accuracy:.4f}")
        
        # Quick demonstration
        print("\n🎯 Quick demonstration - Three-class effects:")
        demo_texts = [
            "This movie is absolutely fantastic and amazing!",    # Positive
            "This movie is terrible and completely awful",       # Negative  
            "This movie is okay, nothing really special",        # Neutral
            "The film is so-so, average at best",               # Neutral
            "not bad, actually quite decent",                   # Positive (double negation)
            "I like it but it's pretty boring overall",         # Neutral (contrast)
            "meh, whatever, it's fine I guess",                 # Neutral
            "absolutely brilliant masterpiece!"                  # Positive
        ]
        
        print(f"\n{'Text':<50} {'Prediction':<10} {'Confidence':<10}")
        print("-" * 75)
        
        for text in demo_texts:
            result = analyzer.predict(text)
            sentiment = result['sentiment']
            conf = result['confidence']
            print(f"{text[:47]+'...' if len(text) > 47 else text:<50} {sentiment:<10} {conf:.3f}")
        
        # Start interactive mode
        print("\n🎮 Starting interactive mode...")
        interactive_system(analyzer)
        
    except KeyboardInterrupt:
        print("\n👋 Program interrupted by user. Goodbye!")
    except Exception as e:
        print(f"❌ System initialization failed: {str(e)}")
        print("Please ensure all required dependencies are installed:")
        print("pip install nltk sklearn numpy pandas matplotlib seaborn scipy")
        
        # Fallback: Create simple demo with synthetic data
        print("\n🔄 Falling back to synthetic data demo...")
        try:
            create_demo_with_synthetic_data()
        except Exception as demo_error:
            print(f"❌ Demo also failed: {demo_error}")

def create_demo_with_synthetic_data():
    """Create a demo with synthetic data if IMDB download fails"""
    print("📊 Creating synthetic three-class data for demonstration...")
    
    # Positive samples
    positive_samples = [
        "This movie is absolutely fantastic! Amazing acting and incredible storyline.",
        "Brilliant performance by all actors. Highly recommended!",
        "One of the best films I have ever seen. Outstanding direction.",
        "Excellent cinematography and superb acting throughout.",
        "I love this movie! It's perfect in every way.",
        "Amazing special effects and wonderful story development.",
        "Fantastic film with excellent character development.",
        "This movie is great! Really enjoyed watching it.",
        "Wonderful performances and beautiful cinematography.",
        "Incredible movie that exceeded all expectations.",
    ] * 50
    
    # Negative samples
    negative_samples = [
        "This movie is terrible. Poor acting and boring plot.",
        "Awful film with bad direction and weak storyline.",
        "Very disappointing and complete waste of time.",
        "I hate this film. It's absolutely horrible.",
        "Worst movie I've ever seen. Completely awful.",
        "Terrible script and poor performances throughout.",
        "This movie is really bad and not worth watching.",
        "Horrible direction and terrible acting from everyone.",
        "Very bad film that I strongly dislike.",
        "Disappointing movie with many serious flaws.",
    ] * 50
    
    # Neutral samples
    neutral_samples = [
        "This movie is okay. Nothing special but watchable.",
        "It's an average film, not great but not terrible either.",
        "The movie is decent. Some good parts, some not so good.",
        "It's fine, I guess. Not really my type of movie.",
        "This film is alright. Could be better, could be worse.",
        "The movie is so-so. Mixed feelings about it.",
        "It's an ordinary film with typical storyline.",
        "The movie is mediocre. Nothing stands out particularly.",
        "It's acceptable but nothing extraordinary.",
        "Standard movie with average performances throughout.",
        "The film is reasonable but not memorable.",
        "It's a normal movie, pretty typical for this genre.",
        "Mixed opinions about this film. Some parts good, others not.",
        "The movie is interesting but has both strengths and weaknesses.",
        "It's different but I'm not sure if I like it or not.",
    ] * 30
    
    # Combine data
    texts = positive_samples + negative_samples + neutral_samples
    labels = ([2] * len(positive_samples) + 
             [0] * len(negative_samples) + 
             [1] * len(neutral_samples))
    
    # Shuffle
    indices = np.random.permutation(len(texts))
    texts = [texts[i] for i in indices]
    labels = [labels[i] for i in indices]
    
    # Split data
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"Synthetic data created:")
    unique, counts = np.unique(labels, return_counts=True)
    label_names = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    for label, count in zip(unique, counts):
        print(f"  {label_names[label]}: {count} samples")
    
    # Create and train analyzer
    analyzer = ThreeClassSentimentAnalyzer()
    analyzer.train(train_texts, train_labels)
    
    # Evaluate
    accuracy, predictions = evaluate_three_class_model(analyzer, test_texts, test_labels)
    
    print(f"\n✅ Synthetic data demo completed!")
    print(f"📊 Model accuracy: {accuracy:.4f}")
    
    # Interactive mode
    interactive_system(analyzer)

if __name__ == "__main__":
    main()

