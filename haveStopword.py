import os   #handle file path, folder operation
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

warnings.filterwarnings('ignore')

class FixedTextProcessor:
    def __init__(self, stopwords_file=None):
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
        
        # 修改的部分：加载停用词
        try:
            self.lemmatizer = WordNetLemmatizer()
            # 尝试从文件加载停用词
            self.stop_words = self.load_stopwords_from_file(stopwords_file)
            
        except Exception as e:
            print(f"Warning: Error loading stopwords or lemmatizer: {e}")
            self.lemmatizer = None
            self.stop_words = {'the', 'a', 'an', 'and', 'or', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
    
    def load_stopwords_from_file(self, stopwords_file=None):
        """从文件加载停用词列表"""
        try:
            if stopwords_file and os.path.exists(stopwords_file):
                # 方法1：从指定文件加载
                print(f"Loading stopwords from file: {stopwords_file}")
                with open(stopwords_file, 'r', encoding='utf-8') as f:
                    file_stopwords = set()
                    for line in f:
                        # 处理每一行，去除空白字符
                        word = line.strip().lower()
                        if word and not word.startswith('#'):  # 忽略空行和注释行
                            file_stopwords.add(word)
                
                # 从停用词中排除否定词和情感词
                file_stopwords = file_stopwords - self.negation_words
                sentiment_words = set(self.sentiment_lexicon.keys())
                file_stopwords = file_stopwords - sentiment_words - set(self.intensifiers.keys())
                
                print(f"Loaded {len(file_stopwords)} stopwords from file")
                return file_stopwords
            
            else:
                # 方法2：尝试从项目目录查找常见的停用词文件
                possible_files = [
                    'stopwords.txt',
                    'stop_words.txt', 
                    'english_stopwords.txt',
                    'data/stopwords.txt',
                    'resources/stopwords.txt'
                ]
                
                for filename in possible_files:
                    if os.path.exists(filename):
                        print(f"Found stopwords file: {filename}")
                        with open(filename, 'r', encoding='utf-8') as f:
                            file_stopwords = set()
                            for line in f:
                                word = line.strip().lower()
                                if word and not word.startswith('#'):
                                    file_stopwords.add(word)
                        
                        # 从停用词中排除否定词和情感词
                        file_stopwords = file_stopwords - self.negation_words
                        sentiment_words = set(self.sentiment_lexicon.keys())
                        file_stopwords = file_stopwords - sentiment_words - set(self.intensifiers.keys())
                        
                        print(f"Loaded {len(file_stopwords)} stopwords from {filename}")
                        return file_stopwords
                
                # 方法3：如果没有找到文件，尝试使用NLTK
                print("No stopwords file found, trying NLTK...")
                nltk_stopwords = set(stopwords.words('english'))
                # 从停用词中排除否定词和情感词
                nltk_stopwords = nltk_stopwords - self.negation_words
                sentiment_words = set(self.sentiment_lexicon.keys())
                nltk_stopwords = nltk_stopwords - sentiment_words - set(self.intensifiers.keys())
                
                print(f"Using NLTK stopwords: {len(nltk_stopwords)} words")
                return nltk_stopwords
                
        except ImportError:
            print("NLTK not available, using basic stopwords")
            # 方法4：使用基本停用词列表作为后备
            basic_stopwords = {
                'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 
                'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 
                'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself',
                'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 
                'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 
                'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 
                'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 
                'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 
                'by', 'for', 'with', 'through', 'during', 'before', 'after', 
                'above', 'below', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 
                'under', 'again', 'further', 'then', 'once'
            }
            return basic_stopwords
        
        except Exception as e:
            print(f"Error loading stopwords: {e}")
            return {'the', 'a', 'an', 'and', 'or', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
    
    def create_stopwords_file(self, filename='stopwords.txt'):
        """创建一个示例停用词文件"""
        sample_stopwords = [
            "# English stopwords for sentiment analysis",
            "# One word per line, lines starting with # are comments",
            "",
            "a", "an", "and", "are", "as", "at", "be", "by", "for", 
            "from", "has", "he", "in", "is", "it", "its", "of", "on", 
            "that", "the", "to", "was", "will", "with", "the", "this", 
            "but", "they", "have", "had", "what", "said", "each", "she", 
            "which", "do", "how", "their", "if", "up", "out", "many", 
            "then", "them", "these", "so", "some", "her", "would", 
            "make", "like", "into", "him", "time", "two", "more", "go", 
            "no", "way", "could", "my", "than", "first", "been", "call", 
            "who", "its", "now", "find", "long", "down", "day", "did", 
            "get", "come", "made", "may", "part"
        ]
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                for word in sample_stopwords:
                    f.write(word + '\n')
            print(f"Created sample stopwords file: {filename}")
        except Exception as e:
            print(f"Error creating stopwords file: {e}")

    # 其他方法保持不变...
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

# 使用示例
def example_usage():
    """如何使用修改后的处理器"""
    
    # 创建示例停用词文件
    processor = FixedTextProcessor()
    processor.create_stopwords_file('my_stopwords.txt')
    
    # 方法1: 指定停用词文件
    processor1 = FixedTextProcessor(stopwords_file='my_stopwords.txt')
    
    # 方法2: 让它自动查找停用词文件
    processor2 = FixedTextProcessor()
    
    # 方法3: 如果没有文件，会使用NLTK或基本停用词
    processor3 = FixedTextProcessor(stopwords_file='nonexistent.txt')
    
    print("All processors initialized successfully!")

if __name__ == "__main__":
    example_usage()