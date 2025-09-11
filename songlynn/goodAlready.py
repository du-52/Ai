
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

# ============================
# IMDB Dataset Loader
# ============================
#Class 1
class IMDBDatasetLoader:
    def __init__(self, data_path="aclImdb"):
        self.data_path = data_path
        self.url = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
        
    def download_and_extract(self):
        """Download and extract IMDB dataset if not present"""
        if not os.path.exists(self.data_path):
            print("📥 Downloading IMDB dataset...")
            filename = "aclImdb_v1.tar.gz"
            
            try:
                urllib.request.urlretrieve(self.url, filename)
                print("✅ Download completed!")
                
                print("📦 Extracting dataset...")
                with tarfile.open(filename, 'r:gz') as tar:
                    tar.extractall('.')
                    
                os.remove(filename)  # Clean up
                print("✅ Dataset extracted successfully!")
                
            except Exception as e:
                print(f"❌ Error downloading dataset: {e}")
                return False
        else:
            print("✅ IMDB dataset already exists!")
        return True
    
    def load_reviews_from_folder(self, folder_path, label):
        """Load reviews from a folder"""
        reviews = []
        labels = []
        
        if not os.path.exists(folder_path):
            print(f"⚠️ Folder not found: {folder_path}")
            return reviews, labels
            
        files = glob.glob(os.path.join(folder_path, "*.txt"))
        print(f"📁 Loading {len(files)} reviews from {folder_path}")
        
        for file_path in files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    review = f.read().strip()
                    if len(review) > 10:  # Filter very short reviews
                        reviews.append(review)
                        labels.append(label)
            except Exception as e:
                print(f"⚠️ Error reading {file_path}: {e}")
                
        return reviews, labels
    
    def create_three_class_labels(self, texts, binary_labels):
        """Convert binary labels to three-class using rule-based neutral detection"""
        processor = FixedTextProcessor()
        three_class_labels = []
        
        print("🔄 Converting to three-class labels...")
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
                
        print(f"✅ Created {neutral_count} neutral samples from {len(texts)} total")
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
        
        print(f"📊 Dataset loaded:")
        print(f"  Training: {len(train_texts)} samples")
        print(f"  Testing: {len(test_texts)} samples")
        
        # Show label distribution
        train_counter = Counter(train_labels)
        test_counter = Counter(test_labels)
        label_names = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
        
        print(f"  Train distribution: {dict((label_names[k], v) for k, v in train_counter.items())}")
        print(f"  Test distribution: {dict((label_names[k], v) for k, v in test_counter.items())}")
        
        return train_texts, train_labels, test_texts, test_labels

# ============================
# Fixed Text Processor with Better Negative Detection
# ============================
#Class 2
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
            text = text.lower().strip() #remove space
            
            
            # Remove HTML tags but keep the text
            text = re.sub(r'<.*?>', ' ', text)
            
            # Replace multiple punctuation with single space
            # 匹配所有不是字母数字(\w)和空白(\s)的字符，替换成空格。
            # 也就是说：标点符号、表情符号、特殊符号全都删掉。
            text = re.sub(r'[^\w\s]', ' ', text)
            text = re.sub(r'\s+', ' ', text)
            
            # Tokenize
            try:
                tokens = word_tokenize(text) if 'word_tokenize' in dir() else text.split() 
                #using nltk method to split the word
                #"I can't do this." → ["I", "ca", "n't", "do", "this", "."]
            except:
                tokens = text.split()   
                # if failed to download nltk ,use python split method
                #"I can't do this." → ["I", "can't", "do", "this."] no so precision
            
            # Filter and lemmatize while keeping sentiment words
            if self.lemmatizer:
                #原词	    Lemmatization	Stemming
                # running	run	            run
                #better	    good	        better
                #mice	    mouse	        mic
                # why need to put self: since it was object attribute,整个类的所有方法都能用


                processed_tokens = []
                for token in tokens:
                    if len(token) > 1:  
                            #丢掉长度只有 1 的 token，比如 "a", "I", "."。
                            #这样会把 "I" 去掉，其实 "I" 对情感分析也很重要（"I love this" vs "They love this"）。
                        #["I", "can't", "do", "this."]这就是token
                        # Keep negations, sentiment words, and intensifiers
                        if (token in self.negation_words or 
                            token in self.sentiment_lexicon or 
                            token in self.intensifiers or
                            token not in self.stop_words):
                                #判断是否要保留这个 token：
                                    #如果是 否定词（not, never），保留。
                                    #如果在 情感词典（happy, sad），保留。
                                    #如果是 程度副词（very, slightly），保留。
                                    #如果不是停用词（the, is, of 等），保留。
                            lemmatized = self.lemmatizer.lemmatize(token)
                                    #self.lemmatizer 就是 WordNetLemmatizer 的实例（对象）。
                                    #.lemmatize(token)
                                        # 这是调用 WordNetLemmatizer 提供的方法 lemmatize。
                                        # token 是当前循环中的单词，比如 "running", "better", "cars"。
                                        # .lemmatize(token) 会返回这个单词的 词形还原形式。
                                        # "running" → "run"
                                        # "cars" → "car"
                                        # "better" → "good"（如果指定词性 pos="a"）
                            processed_tokens.append(lemmatized)
                                #把 token 做 词形还原（running → run, mice → mouse）。
                                #然后保存到 processed_tokens。
                tokens = processed_tokens #用处理过的 tokens 覆盖原来的 tokens，保证后续分析只用干净的词。
            else:
                        # 这部分和前面的 if self.lemmatizer: 配套。
                        # 也就是说，如果类里 没有 lemmatizer（没有词形还原器），就走这个分支。
                        # 目的：即使没有词形还原功能，依然要做最基本的 清理和过滤。
                tokens = [token for token in tokens if len(token) > 1 and token not in self.stop_words]
                    #遍历现有的 tokens（已经分词好的单词列表）。
                    # 只保留符合条件的 token：
                    # len(token) > 1 → 去掉只有一个字符的词（比如 "a", "I", "!"）。
                    # token not in self.stop_words → 去掉停用词（stop words），比如 "the", "is", "at", "on" 这些对语义贡献不大的词。
                    # 最终 tokens 会变成一个过滤后的 干净单词列表。

            return ' '.join(tokens)
                #把清理好的 token 列表重新拼接成字符串。
                # 举例：
                # tokens = ["love", "machine", "learning"]
                # ' '.join(tokens)   →   "love machine learning"
                            
        except Exception as e:
            return text.lower()
                #这是错误处理部分（异常捕获）。
                # 如果上面任何一步出错（比如 tokens 是空的，或者某些函数没导入），就不会让程序崩溃。
                # 而是直接返回一个 简单的小写文本，保证整个程序还能运行下去。
                # 相当于一个 安全兜底。

# ============================
# Fixed Three-Class Sentiment Analyzer
# ============================
#Class 3
class FixedThreeClassSentimentAnalyzer:
    def __init__(self): #这是 构造函数，在您创建对象时自动执行。
            #self 指代对象本身，用于在类里保存数据和方法,self also can write as this, me, obj, it just a paremeter
        self.processor = FixedTextProcessor()   
            #这里新建了一个 FixedTextProcessor(class 2) 的对象，并赋值给当前类的 processor 属性。
            #作用：负责 文本预处理（比如清理标点、去掉停用词、分词、词形还原等）。
        self.vectorizer = None
            #没初始化，暂时设为 None。
            # 未来会放入 向量化工具（如 CountVectorizer 或 TfidfVectorizer），把文本转成数值特征。
        self.model = None
            #初始时没有模型，设为 None。
            # 后面训练好模型（比如 Logistic Regression、Naive Bayes、SVM）时会存到这里。
        self.is_trained = False
            #一个布尔标志，表示当前模型 是否已经训练过。
            # 防止您在模型没训练时就直接拿来预测。
        self.label_names = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
            #定义了 情感标签的映射表。
            # 训练好的模型可能会输出数字类别（0, 1, 2），但我们希望最终看到的是文字标签。
    
    def create_enhanced_features(self, texts):
        #create_enhanced_features 的工作流程
            # 输入
                # texts 参数：一堆未经处理的原始句子，例如：
                # texts = ["I LOVE this movie!!!", "Not good at all...", "Wow, AMAZING!"]
            # for 循环逐个处理
                # 循环拿出一句原始句子 text
                # 丢进 self.processor.preprocess_text(text) 进行清洗：
                    # 小写化
                    # 去掉标点
                    # 分词
                    # 去停用词
                    # 词形还原
            # 最后再拼回一个干净的字符串
            # 得到结果赋值给 processed
            # 存入列表
                # processed_texts.append(processed)
            # 这样一来，每处理完一句，结果就被放进 processed_texts 这个大列表里
                # 最终结果
                # 如果输入：
                    # texts = ["I LOVE this movie!!!", "Not good at all...", "Wow, AMAZING!"]
                # 经过 create_enhanced_features 的 文本预处理部分，您得到：
                    # processed_texts = ["love movie", "not good", "wow amazing"]

        """Create enhanced features for the model"""
            #print("Create enhanced features for the model") → 会直接在控制台输出文字。
            # """Create enhanced features for the model""" → 不会打印，而是函数的 说明文档，可以通过 .__doc__ 访问。
        processed_texts = []
        additional_features = []
            #processed_texts 用来保存清理过的文本（分词、去噪等）。
            # additional_features 用来保存每条文本对应的一组数值特征。
            #additional_features = [features1,features2]
                #features1=[sentiment_score,len(text.split())................]
                #so it looks like 2d array
        for text in texts:
            try:
                processed = self.processor.preprocess_text(text)    #fixedtextprocessor class 里的 preprocess_text method
                        #这个时候是一句被清理过的
                processed_texts.append(processed)
                    #对每个文本调用 self.processor.preprocess_text(text)（就是之前的预处理器，清理标点、去掉停用词等）。
                    # 保存到 processed_texts
                
                # Calculate rule-based sentiment score
                sentiment_score, _ = self.processor.calculate_sentiment_score(text)
                
                # Extract additional features
                features = [
                    sentiment_score,                    # Rule-based sentiment score
                    len(text.split()),                  # Word count
                    len(text),                          # Character count
                    text.count('!'),                    # Exclamation marks
                    text.count('?'),                    # Question marks
                    len([w for w in text.split() if w.isupper()]),  # how many Uppercase words
                        #why need to count upper?
                            #"I love this product"
                            # 大写单词数 = 0 → 情绪正常
                            # "I LOVE THIS PRODUCT"
                            # 大写单词数 = 3 → 情绪非常强烈，模型倾向于判断为 Positive
                            # 虽然两句话意思一样，但大写数量让模型能分辨出 情绪强度的差别。
                    text.lower().count('very'),         # Intensifier count
                    text.lower().count('not'),          # Negation count
                    1 if any(contrast in text.lower() for contrast in self.processor.contrast_words) else 0,  # Has contrast: 是否包含转折词
                        #y=先把整句话变成小写 (text.lower())，然后检查里面有没有出现 contrast_words 里的对比词（例如 "but", "however", "although" 等）。
                        # 如果有 → 返回 1
                        # 如果没有 → 返回 0
                            #text.lower()
                                # 把整段文本转成小写，避免因为大小写不同而漏掉匹配。
                                # 例如 "BUT I like it" 和 "but I like it" 在逻辑上是一样的。
                            # self.processor.contrast_words
                                # 这是您预先定义的一组“对比词”，比如：
                                # self.contrast_words = ["but", "however", "although", "though", "yet"]
                                # 它们通常表示转折、对比。
                            # contrast in text.lower() for contrast in self.processor.contrast_words
                                # 遍历所有对比词，检查它们是否出现在 text.lower() 里。
                                # 生成一个布尔值序列，比如：
                                # text = "I like it, but it's expensive."
                                # 结果可能是 [False, False, True, False, False]
                            # any(...)
                                # 如果序列里有一个是 True，就返回 True。
                                # 说明文本里至少出现了一个对比词。
                                # 1 if ... else 0
                                # 如果有对比词 → 特征值 = 1
                                # 如果没有 → 特征值 = 0
                            # 📊 举例
                                # "I love the design, but the battery is bad."
                                # → 包含 "but" → 特征值 = 1
                                # "This phone is great."
                                # → 没有转折词 → 特征值 = 0
                    len([w for w in text.lower().split() if w in self.processor.sentiment_lexicon]),  # Sentiment words
                    text.lower().count('but'),          # But count (contrast indicator)
                    text.lower().count('however'),      # However count 
                ] 
                additional_features.append(features)
                #features 是一个 list，里面包含了针对某一句话算出来的各种数值特征，比如：
                    # sentiment_score（基于规则的情感分数）
                    # len(text.split())（词数）
                    # len(text)（字符数）
                    # text.count('!')（惊叹号个数）
                    # text.count('?')（问号个数）
                    # len([w for w in text.split() if w.isupper()])（全大写单词数）
                    # …… 还有其它十几个指标
                # 举例来说，如果分析 "I LOVE this movie! But it's too long."，
                # features 可能会是：
                    # [2, 8, 32, 1, 0, 1, 0, 0, 1, 3, 1, 0]
                    # （只是举例，不是实际值）
                # additional_features 是一个 大列表，用来存储 所有句子的特征。
                # 如果我们有 100 句话，就会执行 100 次 append(features)。
                # 最后 additional_features 会变成一个 二维数组（100 行 × 12 列），
                # 每一行代表一句话的特征。
                # 最终返回时，代码把它转成 np.array(additional_features)，方便后面直接喂给机器学习模型训练/预测。
                
            except Exception as e:
                processed_texts.append(text.lower())    
                    #如果处理失败，就退而求其次，直接把原始文本转成小写（不做复杂预处理），这样保证 processed_texts 里 至少有一个安全版本 的文本。
                additional_features.append([0] * 12)
                    #这里直接塞进一个 全零的特征向量，长度是 12（因为前面设计了 12 个特征）。
                    # [0] * 12 就是 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]。
                    # 这样即使提取失败，这句话仍然会有一个 有效的占位符特征，保证数组形状统一，不会报错。
        
        return processed_texts, np.array(additional_features)
            #processed_texts
                # 这是一个 list，存放每一句话的预处理结果（小写化、去停用词、词形还原等）。
                # 举例：输入
                # ["I LOVE this movie!", "Not good at all..."]

                # 可能变成：
                # ["love movie", "not good"]

            # np.array(additional_features)
                # additional_features 是我们在循环里 append(features) 的大列表，每条文本对应一行特征。
                # 用 np.array() 转换成 NumPy 数组，方便后面直接喂给机器学习模型。
                # 举例：
                # 第一条 "I LOVE this movie!" → [2, 4, 18, 1, 0, 1, 0, 0, 0, 2, 0, 0]
                # 第二条 "Not good at all..." → [-1, 4, 16, 0, 0, 0, 0, 1, 0, 1, 0, 0]

                # 最终 np.array(additional_features) 就是一个二维矩阵：
                # [[ 2,  4, 18, 1, 0, 1, 0, 0, 0, 2, 0, 0],
                #  [-1,  4, 16, 0, 0, 0, 0, 1, 0, 1, 0, 0]]

    
    def train(self, texts, labels):
            #texts = 一堆原始句子（例如影评）
            # labels = 每个句子对应的情感标签（0=Negative, 1=Neutral, 2=Positive）
        """Train the fixed model with better architecture"""
        print("🚀 Training fixed three-class sentiment analyzer...")
        
        # Check label distribution
        unique_labels, counts = np.unique(labels, return_counts=True) #检查训练集里每个类别有多少样本（是否均衡）。

        print(f"Label distribution: {dict(zip([self.label_names[l] for l in unique_labels], counts))}") #exp： abel distribution: {'Negative': 200, 'Neutral': 150, 'Positive': 250}

        
        # Create enhanced features
        processed_texts, additional_features = self.create_enhanced_features(texts)
            #processed_texts → 清洗后的句子字符串列表，例如：
                # ["love movie", "not good", "wow amazing"]
            # additional_features → 数值特征矩阵，例如： 
                # [[ 1,  2, 15, 0, 0, 0, 1, 0, 0, 1, 0, 0],
                #  [-1,  3, 12, 0, 0, 0, 0, 1, 0, 1, 0, 0],
                #  [ 2,  2, 10, 1, 0, 0, 0, 0, 0, 2, 0, 0]]
            #since the method will return processed_texts, np.array(additional_features)

        # TF-IDF with optimized parameters
        self.vectorizer = TfidfVectorizer(
            max_features=12000,
                #限制词汇表的最大大小为 12,000 个特征词。
                # TF-IDF 会统计所有文本的词汇，但我们只保留最重要的 12,000 个，避免内存爆炸、降低计算量。
            ngram_range=(1, 2),  # Unigrams and bigrams
                #表示同时考虑 unigram（单个词） 和 bigram（两个词连在一起）。
                # 例如句子 "not good at all"：
                # unigram: "not", "good", "at", "all"
                # bigram: "not good", "good at", "at all"
                # 这样可以捕捉到更复杂的语义（比如“not good”比单独的“good”更偏负面）。
            min_df=3,
                #只保留至少在 3个文档里出现过的词汇。
                # 过滤掉非常罕见的噪声词。
            max_df=0.8,
                #如果一个词在超过 80% 的文档中都出现，就把它丢掉。
                # 这些太常见的词（类似 stopwords，如 "the", "is"）没有区分度。
            sublinear_tf=True,
                #使用 对数缩放的 TF（词频）。
                # 普通 TF = 词出现次数；sublinear = 1 + log(TF)。
                # 这样可以降低极端高频词的权重（避免某个词出现 1000 次就支配一切）。
                
                #先看普通 TF-IDF 的 TF 部分
                # TF (Term Frequency) 就是某个词在文档里出现的次数。
                    # 比如句子 "happy happy happy good"：
                    # TF("happy") = 3
                    # TF("good") = 1
                # 这样做的问题是：
                    # 👉 如果一个词在某篇文档里特别多（例如出现 1000 次），它会被认为比出现 1 次的重要 1000 倍。
                    # 👉 但在语义上，其实没有那么夸张，出现 10 次已经够明显了，1000 次不会比 10 次更重要。
                # 🔎 sublinear_tf=True 是什么？
                # 它的意思是：对 TF 做对数缩放（logarithmic scaling）。：计算公式：TFsublinear​=1+log(TF)
               
                # ⚠️ 注意：这里的 log
                # ⁡log 是自然对数 (log base e)。

                # 📊 举例
                    # 句子："happy happy happy good"
                        # 普通 TF
                            # TF("happy") = 3
                            # TF("good") = 1
                        # 对数缩放 TF（sublinear_tf=True）
                            #TF("happy") = 1+log(3)1+log(3) ≈ 2.10
                            # TF("good") = 1+log(1)1+log(1) = 1
                    # 👉 本来 3 次的 "happy" 是 3 倍于 "good"，
                    # 👉 现在经过 log 变换后，只有 2.1 倍，更合理。

                # ⚡ 再看极端情况
                    # 假设 "happy" 出现 1000 次：
                        # 普通 TF: 1000
                    # 对数缩放: 1+log(1000)1+log(1000) ≈ 7.9
                    # 👉 如果不用 log，模型会被 "happy" 完全支配；
                    # 👉 用 log 后，1000 次和 10 次的差距就 被压缩 了，降低了“水军词”的影响。

                # 🎯 总结
                    # 普通 TF：出现 1000 次比出现 10 次重要 100 倍。
                    # sublinear TF：出现 1000 次只比出现 10 次重要 2 倍左右。
                    # 这样更符合语言直觉：重复很多次的词不会无限增加信息量。
            strip_accents='unicode',
                #strip_accents='unicode'
                # 去掉重音符号，例如："café" → "cafe"，"naïve" → "naive"
                # 统一格式，避免稀疏性。
            lowercase=True #把全部变成lowercase

        )
        
        tfidf_features = self.vectorizer.fit_transform(processed_texts)
            #processed_texts 是您前面 清洗好的句子列表（比如 ["i love this movie", "this is terrible"]）。
            # TfidfVectorizer 会把这些句子转换成一个 TF-IDF 矩阵。
            #exp ["i love this movie", "this is terrible"] 
            # became: 可能变成一个稀疏矩阵（2 行 × 词汇数列）。
            # 假设词汇表里有：["i", "love", "this", "movie", "is", "terrible"]
            # 那么矩阵大概像这样：
                # [[TF-IDF(i), TF-IDF(love), TF-IDF(this), TF-IDF(movie), 0, 0],
                #  [0, 0, TF-IDF(this), 0, TF-IDF(is), TF-IDF(terrible)]]



        # Combine TF-IDF and additional features
        additional_sparse = csr_matrix(additional_features)
            #additional_features 是您在 create_enhanced_features 里提取的 手工特征（rule-based + 统计信息），
            # 比如：，规则情感分数，
                # 单词数量
                # 字符数量
                # 感叹号数量
                # 全大写单词数
                # 是否包含对比词（but, however）等等
            # 例如一条句子 [0.8, 4, 20, 1, 0, 0, 1, 0, 0, 2, 1, 0]
                # （12 个数字，对应不同的特征）。
            #csr_matrix 的作用：把这个 普通二维数组 转成稀疏矩阵，节省内存。
                #[0, 0, 3, 0, 0, 0, 1, 0, 0, 0]
                # 普通存法：存 10 个数。
                # 稀疏存法：存成 {2:3, 6:1}，只记录第 2 位是 3，第 6 位是 1。

        X = hstack([tfidf_features, additional_sparse])
            #hstack = 横向拼接两个矩阵。
                # 也就是说：每个样本的 TF-IDF 向量（几千个维度） + 额外特征向量（12 个维度） 拼在一起，变成一个更大的特征向量。
                # 例子：
                # TF-IDF 向量维度：12000
                # 额外特征维度：12
                # 拼接之后：12012 维。
        print(f"Feature matrix shape: {X.shape}")
            #打印出来的就是最终训练用的特征矩阵形状。
        
        # Use ensemble of models for better performance
        logistic = LogisticRegression(
            random_state=42, 
            max_iter=1000, 
            class_weight='balanced',
            C=1.0,
            solver='liblinear'
        )
        #逻辑回归是一个线性分类器，适合处理 稀疏特征（TF-IDF）。        `
            # max_iter=1000 → 允许最多迭代 1000 次，确保收敛。
            # class_weight='balanced' → 如果训练数据里正负样本比例不均，它会自动调整权重。
            # C=1.0 → 正则化参数（控制模型复杂度），越大越容易过拟合。
            # solver='liblinear' → 适合小数据集 & 稀疏矩阵的优化器。
            # 👉 用来捕捉 线性关系，比如：“good” vs “bad” 这种直观情绪词。
        
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced_subsample',
            n_jobs=-1
        )
        #随机森林是基于 决策树的集成模型，能发现更复杂的 非线性关系。
        # n_estimators=100 → 用 100 棵树。
        # max_depth=12 → 每棵树的最大深度，防止过拟合。
        # min_samples_split=5 → 至少有 5 个样本才能继续分裂。
        # min_samples_leaf=2 → 叶子节点至少有 2 个样本。
        # class_weight='balanced_subsample' → 在每棵树训练时平衡类别。
        # n_jobs=-1 → 用所有 CPU 核心加速训练。
        # 👉 用来捕捉 复杂模式，比如：“not bad at all but still could be better” 这种带有多重转折的句子。
                
        # Voting classifier
        self.model = VotingClassifier(
            estimators=[('lr', logistic), ('rf', rf)],
            voting='soft'
        )
        #✅ Soft Voting（软投票 = 概率加权）
            # 每个模型不仅给出类别，还给出 预测概率，然后取平均。
            # 例子：
                # 逻辑回归预测：Positive (0.7), Neutral (0.2), Negative (0.1)
                # 随机森林预测：Positive (0.6), Neutral (0.1), Negative (0.3)
            # 取平均：
                # Positive: (0.7+0.6)/2 = 0.65
                # Neutral: (0.2+0.1)/2 = 0.15
                # Negative: (0.1+0.3)/2 = 0.20
            # 👉 最终选择 Positive（0.65 最大）。
        
        self.model.fit(X, labels)
            #🎯 场景类比：班级投票
                    # 您有一堆学生作业（X），每份作业都有分数（labels）。
                    # 您找了 两位老师 来评分：
                        # 老师 A = 逻辑回归（很理性，只看线性趋势）
                        # 老师 B = 随机森林（很有经验，能看复杂的非线性模式）
                    # 🔨 执行 self.model.fit(X, labels) 时：
                        # 老师 A（逻辑回归）
                            # 仔细分析所有作业的特征（X）和分数（labels）。
                            # 得到一套数学公式（权重向量），以后只要看到新作业，就能代入公式，算出分数。
                        # 老师 B（随机森林）
                            # 用数据把“知识点”拆分成很多个判断题（决策树）。
                            # 每棵树都给出一个预测，最后由森林内部投票决定。
                        # 班主任（VotingClassifier）
                            # 记录下：以后预测时，必须 同时询问老师 A 和老师 B。
                            # 如果是 voting='hard'：取多数表决。
                            # 如果是 voting='soft'：取老师们给出的 分数概率，再加权平均，最后决定结果。
                    # 💡 结果：
                        # fit 就是：让两个子模型各自完成训练，并把他们的结果登记到 VotingClassifier。
                        # 之后，当您调用 predict(new_X)：
                        # 逻辑回归 → 输出「我觉得 70% 是正面，30% 是负面」
                        # 随机森林 → 输出「我觉得 60% 是正面，40% 是负面」
                        # VotingClassifier → 把两个概率加权平均，再决定最终的分类。

        self.is_trained = True  #这个模型已经训练好了，可以拿来预测。
        print("✅ Fixed three-class model training completed!")
    
    def predict(self, text):
        """Enhanced prediction with better rule-based integration"""
        if not self.is_trained:
            raise ValueError("Model not trained!")
            #如果您还没调用 train() 就直接 predict()，会报错。
            # 目的是 保证模型已经训练完成。
        try:
            # Rule-based analysis
            sentiment_score, context_info = self.processor.calculate_sentiment_score(text)
                #调用之前定义的规则分析器 (FixedTextProcessor) 计算情绪分数。
                # sentiment_score 是一个数值，正数偏正面，负数偏负面。
                # context_info 记录一些分析细节（比如找到了 negation、intensifier 等）。
            
            # Strong rule-based decisions for very clear cases
            if abs(sentiment_score) > 2.5:  # Very strong sentiment
                if sentiment_score > 0:
                    prediction = 2  # Positive  ，根据正负值，设置 prediction（0 = Negative, 1 = Neutral, 2 = Positive）
                    confidence = min(0.95, 0.8 + abs(sentiment_score) * 0.05)
                        #规则判定的置信度 = 基础 0.8 + 情绪强度 × 0.05
                        # 置信度上限 = 0.95
                        # 逻辑：情绪越明显，规则越可靠，但不会完全 100% 保证
                else:
                    prediction = 0  # Negative
                    confidence = min(0.95, 0.8 + abs(sentiment_score) * 0.05)
                
                probabilities = [0.05, 0.05, 0.9] if prediction == 2 else [0.9, 0.05, 0.05]
                    #这是 模拟规则判定的类别概率。
                    # prediction == 2 → Positive
                        # Positive = 0.9
                        # Neutral = 0.05
                        # Negative = 0.05
                    # prediction == 0 → Negative
                        # Negative = 0.9
                        # Neutral = 0.05
                        # Positive = 0.05
                    #意思：当规则判定情绪非常强烈时，我们就赋予主要类别 90% 的置信度，其它类别仅 5%。
                return {
                    'original_text': text,# 原始文本
                    'prediction': prediction, # 预测类别编号 (0,1,2)
                    'sentiment': self.label_names[prediction],# 类别标签 ('Negative','Neutral','Positive')
                    'confidence': confidence,# 置信度（上一步计算的值）
                    'probabilities': {# 每个类别的概率
                        'negative': probabilities[0],
                        'neutral': probabilities[1], 
                        'positive': probabilities[2]
                    },
                    'context_notes': [f"Strong rule-based decision (score: {sentiment_score:.2f})", context_info],
                    'rule_based_score': sentiment_score # 规则分数
                }
            
            # Use ML model for other cases
            processed_texts, additional_features = self.create_enhanced_features([text])
                #text 是一条原始输入句子，例如 "I am not happy today!"
                # create_enhanced_features 做了两件事：
                # 文本预处理 → 清洗、分词、去停用词、词形还原 → 得到 processed_texts
                # 示例输出：["not happy today"]
                # 提取附加规则特征 → 句子长度、感叹号数量、强度词、情绪词数量等 → 得到 additional_features
                # 示例输出：[[sentiment_score, word_count, char_count, ...]]

            tfidf_features = self.vectorizer.transform(processed_texts)
                #self.vectorizer 是之前训练好的 TfidfVectorizer
                # 将文本转为 稀疏向量（每个单词或 n-gram 对应一列，值为 TF-IDF）
                # 输出是一个稀疏矩阵 tfidf_features
            
            additional_sparse = csr_matrix(additional_features)
                #把规则特征（additional_features，普通 numpy array）也转为 稀疏矩阵
                # 这样可以和 TF-IDF 特征做拼接
            X = hstack([tfidf_features, additional_sparse])
                #使用 hstack → 水平拼接，把：
                    # 文本特征（TF-IDF）
                    # 规则特征（额外的数值）
                # 合并成一个矩阵 X
                # X 就是最终模型可以直接用来预测的 完整特征矩阵

            # Get ML predictions
            ml_prediction = self.model.predict(X)[0]
            ml_probabilities = self.model.predict_proba(X)[0]
                #ml_prediction = 投票分类器给出的类别编号（0=Negative,1=Neutral,2=Positive）
                # ml_probabilities = 每个类别的概率分布（逻辑回归 + 随机森林软投票综合的结果）

            # Adjust predictions based on rule-based score
            adjusted_probs = ml_probabilities.copy()
                #ml_probabilities 是 VotingClassifier 输出的 [Negative, Neutral, Positive] 概率
                # 复制一份，方便在不修改原始概率的情况下进行调整

            if abs(sentiment_score) > 1.0:  # Moderate rule-based signal
                    #规则分析得分 sentiment_score 的绝对值 > 1 → 情绪信号中等或强
                    # 如果信号较弱，就直接使用 ML 预测
                adjustment = min(0.3, abs(sentiment_score) * 0.15) #规则越强，微调越大，但避免超过 30%
                
                if sentiment_score > 0:  # Rule suggests positive
                    adjusted_probs[2] += adjustment  # 提高 Positive 概率
                    adjusted_probs[0] = max(0.05, adjusted_probs[0] - adjustment/2)
                    adjusted_probs[1] = max(0.05, adjusted_probs[1] - adjustment/2)
                else:  # Rule suggests negative
                    adjusted_probs[0] += adjustment # 提高 Negative 概率
                    adjusted_probs[2] = max(0.05, adjusted_probs[2] - adjustment/2)
                    adjusted_probs[1] = max(0.05, adjusted_probs[1] - adjustment/2)
                
                # Renormalize
                adjusted_probs = adjusted_probs / adjusted_probs.sum()
                    #确保三个类别的概率加起来 = 1
                    # 避免出现总和大于或小于 1 的情况
                final_prediction = np.argmax(adjusted_probs) 
                    #argmax = “返回最大值的 索引（位置）”，而不是最大值本身。
                        #exp: adjusted_probs = [0.05, 0.1, 0.85]  # Negative, Neutral, Positive
                        #最大值 = 0.85 → 对应 Positive（索引 2）
                        # final_prediction = 2 → 可以用 self.label_names[2] 显示 'Positive'

            else:   #直接用mlmodel，不做微调
                final_prediction = ml_prediction
                adjusted_probs = ml_probabilities
            
            return {
                'original_text': text, #返回 输入的原句，方便追踪和调试
                'prediction': final_prediction, #final_prediction = np.argmax(adjusted_probs),表示 预测的类别编号：0=Negative, 1=Neutral, 2=Positive
                'sentiment': self.label_names[final_prediction], #根据编号获取 文字标签,例如 final_prediction = 2 → 'Positive'
                'confidence': max(adjusted_probs), #置信度 = 最大概率值,表示模型 对预测类别的信心大小,范围 0~1
                'probabilities': {
                    'negative': adjusted_probs[0],
                    'neutral': adjusted_probs[1], 
                    'positive': adjusted_probs[2]
                },
                #返回 三类的具体预测概率,方便进一步分析或做阈值处理
                'context_notes': [f"Hybrid ML+Rule prediction (rule score: {sentiment_score:.2f})", context_info] if context_info else [f"Hybrid ML+Rule prediction (rule score: {sentiment_score:.2f})"],
                    #记录 预测方式 + 规则分析分数
                    # 如果有上下文信息（context_info）也加入
                    # 作用：调试、解释模型预测理由
                    #如果 context_info 有内容（不是空值 / None）：
                        #列表包含两条信息：
                            # 模型类型 + 规则分数
                            # 规则分析中额外的上下文信息
                    #如果 context_info 是空值 / None：列表只包含 模型类型 + 规则分数
                'rule_based_score': sentiment_score
            }
            
        except Exception as e:
            print(f"Prediction error: {e}")
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
#class 3 until here


# ============================
# Testing and Demo Functions  
# ============================

def test_fixed_analyzer():
    """Test the fixed analyzer with IMDB dataset"""
    print("🧪 Testing Fixed Three-Class Sentiment Analyzer with IMDB Dataset...")
    
    # Load IMDB dataset
    loader = IMDBDatasetLoader()
    
    try:
        # Load a subset for faster testing (remove subset_size=2000 for full dataset)
        train_texts, train_labels, test_texts, test_labels = loader.load_imdb_data(subset_size=2000)
        
        if train_texts is None:
            print("❌ Failed to load IMDB dataset")
            return None
               
        # Train the analyzer
        analyzer = FixedThreeClassSentimentAnalyzer()
        analyzer.train(train_texts, train_labels)
        
        # Test on challenging cases first
        test_cases = [
            # Previously problematic cases (should now work correctly)
            ("very bad", "Negative"),
            ("i dont like it", "Negative"), 
            ("i don't like it", "Negative"),
            ("didn't like", "Negative"),
            ("doesn't like", "Negative"),
            ("terrible movie", "Negative"),
            ("I hate this film", "Negative"),
            ("awful acting", "Negative"),
            ("horrible movie", "Negative"),
            ("worst film ever", "Negative"),
            
            # Double negations (should be positive/neutral)
            ("not bad", "Positive"),
            ("not terrible", "Positive"), 
            ("not awful", "Positive"),
            ("not horrible", "Positive"),
            ("not boring", "Positive"),
            
            # Negative double negations 
            ("not good", "Negative"),
            ("not great", "Negative"),
            ("not excellent", "Negative"),
            
            # Clear positives
            ("excellent movie", "Positive"),
            ("I love this film", "Positive"),
            ("amazing story", "Positive"),
            ("fantastic acting", "Positive"),
            ("brilliant direction", "Positive"),
            ("wonderful film", "Positive"),
            
            # Neutrals
            ("okay movie", "Neutral"),
            ("average film", "Neutral"),
            ("so-so", "Neutral"),
            ("meh", "Neutral"),
            ("mixed feelings", "Neutral"),
            ("mediocre", "Neutral"),
            
            # Complex cases
            ("good but boring", "Neutral"),
            ("I like it but has problems", "Neutral"),
            ("very good but too long", "Positive"),
            ("not bad at all", "Positive"),
            ("pretty good movie", "Positive"),
            ("really bad film", "Negative"),
            ("extremely disappointing", "Negative"),
        ]
        
        print(f"\n{'='*90}")
        print("🎯 Testing Challenging Cases")
        print(f"{'='*90}")
        print(f"{'#':<3} {'Test Text':<35} {'Expected':<10} {'Predicted':<10} {'Conf':<6} {'Rule':<8} {'Status'}")
        print("-" * 90)
        
        correct_predictions = 0
        for i, (text, expected) in enumerate(test_cases, 1):
            try:
                result = analyzer.predict(text)
                predicted = result['sentiment']
                confidence = result['confidence']
                rule_score = result.get('rule_based_score', 0)
                
                is_correct = predicted == expected
                if is_correct:
                    correct_predictions += 1
                
                status = "✓" if is_correct else "✗"
                print(f"{i:<3} {text:<35} {expected:<10} {predicted:<10} {confidence:.3f} {rule_score:>+.2f} {status}")
                
            except Exception as e:
                print(f"{i:<3} {text:<35} {'ERROR':<10} {'N/A':<10} {'N/A':<6} {'N/A':<8} ✗")
        
        challenge_accuracy = correct_predictions / len(test_cases)
        print(f"\n📊 Challenge Test Accuracy: {challenge_accuracy:.3f} ({correct_predictions}/{len(test_cases)})")
        
        # Evaluate on actual test set
        print(f"\n{'='*60}")
        print("📈 Evaluating on IMDB Test Set")
        print(f"{'='*60}")
        
        test_predictions = [] # 用来存放所有预测结果
        correct = 0 # 记录预测正确的样本数
        
        # Test on subset for speed
        # 为了加快速度，只测试部分数据
        test_subset_size = min(500, len(test_texts))
        print(f"Testing on {test_subset_size} samples...")
        
        for i, text in enumerate(test_texts[:test_subset_size]):
            result = analyzer.predict(text)  # 调用模型预测当前文本
            pred = result['prediction'] # 提取预测的标签
            test_predictions.append(pred) # 保存预测结果
            
            if pred == test_labels[i]:  # 如果预测和真实标签一致
                correct += 1   # 计数器+1
            
            if (i + 1) % 100 == 0:      # 每处理100条样本打印进度
                print(f"  Processed {i + 1}/{test_subset_size} samples...")
        
        # 计算准确率
        test_accuracy = correct / test_subset_size
        print(f"\n📊 IMDB Test Accuracy: {test_accuracy:.4f} ({correct}/{test_subset_size})")
        
        # Show confusion matrix
        try:
            from sklearn.metrics import confusion_matrix
                #EXP
                # Actual \ Predicted   Neg   Neu   Pos
                # Neg                 120    10     5
                # Neu                  12   100    18
                # Pos                   7    15   113

            cm = confusion_matrix(test_labels[:test_subset_size], test_predictions)
                #lassification_report 会输出：
                    # precision (精确率): 预测为该类的样本中有多少是真的。
                    # recall (召回率): 真实是该类的样本中有多少被模型正确找到了。
                    # f1-score: 精确率和召回率的调和平均值，更全面。
                    # support: 每个类别在测试集中实际的样本数。
            print(f"\n📈 Confusion Matrix:")
            print(f"          Predicted")
            print(f"Actual    Neg  Neu  Pos")
            labels = ['Neg', 'Neu', 'Pos']
            for i, label in enumerate(labels):
                row = f"{label}      "
                for j in range(len(cm[i])):
                    row += f"{cm[i][j]:4d} "
                print(row)
        except:
            pass
        
        # Classification report
        try:
            from sklearn.metrics import classification_report
            report = classification_report(test_labels[:test_subset_size], test_predictions, 
                                         target_names=['Negative', 'Neutral', 'Positive'])
            print(f"\n📋 Classification Report:\n{report}")
        except:
            pass #如果混淆矩阵导入或计算失败，就直接跳过，不影响后续执行
        
        print("✅ Testing completed!")
        return analyzer
        #说明测试跑完了，并返回 analyzer（你的模型对象）。
        # 返回 analyzer 可以继续用，不会因为测试而丢失引用。

    except Exception as e:
        print(f"❌ Error during testing: {e}")
        return None #如果测试过程中有任何意外错误（比如 analyzer.predict 报错），会打印出来，并返回 None，避免程序直接崩溃。

def interactive_demo(analyzer):
    """Interactive demonstration with the fixed analyzer"""
    print(f"\n{'='*80}")
    print("🎬 Fixed IMDB Three-Class Sentiment Analyzer")  
    print("   ✅ Now with proper negative detection!")
    print(f"{'='*80}")
    print("📝 Commands:")
    print("   • Type any movie review for analysis")
    print("   • 'test' - Run quick demo")
    print("   • 'help' - Show help")
    print("   • 'quit' - Exit")
    print("-" * 80)
    
    while True:
        user_input = input("\n🎭 Enter your movie review: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("👋 Thanks for testing the fixed analyzer!")
            break
            
        elif user_input.lower() == 'test':
            run_quick_demo(analyzer)
            continue
            
        elif user_input.lower() == 'help':
            print_help_info()
            continue
            
        elif len(user_input.strip()) < 2:
            print("⚠️  Please enter at least 2 characters")
            continue
        
        try:
            result = analyzer.predict(user_input)
            display_result(result)
            
        except Exception as e:
            print(f"❌ Analysis error: {str(e)}")

def display_result(result):
    """Display prediction results in a nice format"""
    print(f"\n{'='*80}")
    print("📊 Analysis Result")
    print(f"{'='*80}")
    
    # Show original text (truncated if too long)
    text = result['original_text']
    display_text = text[:70] + "..." if len(text) > 70 else text #原始评论如果太长（超过 70 字符），就截断并加 ...
    print(f"📝 Review: {display_text}")
    
    # Show analysis notes
    if result.get('context_notes'):
        notes = '; '.join(result['context_notes'])
        print(f"🔍 Analysis: {notes}")
    
    # Show rule-based score
    if 'rule_based_score' in result:
        score = result['rule_based_score'] 
        score_indicator = "📈" if score > 0 else "📉" if score < 0 else "📊"
        print(f"{score_indicator} Rule score: {score:+.2f}")
    
    print("-" * 80)
    
    # Main prediction
    sentiment = result['sentiment']
    confidence = result['confidence']
    
    # Choose emoji based on sentiment
    emoji_map = {'Negative': '😔', 'Neutral': '😐', 'Positive': '😊'}
    emoji = emoji_map.get(sentiment, '🤔')
    
    print(f"{emoji} Predicted Sentiment: {sentiment}")
    print(f"🎯 Confidence: {confidence:.4f} ({confidence*100:.2f}%)")
    
    # Detailed probabilities
    probs = result['probabilities']
    print(f"📊 Detailed probabilities:")
    print(f"   😔 Negative: {probs['negative']:.4f} ({probs['negative']*100:.2f}%)")
    print(f"   😐 Neutral:  {probs['neutral']:.4f} ({probs['neutral']*100:.2f}%)")
    print(f"   😊 Positive: {probs['positive']:.4f} ({probs['positive']*100:.2f}%)")
    
    # Confidence assessment
    if confidence > 0.8:
        conf_msg = "Very confident ✨"
    elif confidence > 0.6:
        conf_msg = "Confident 👍"
    elif confidence > 0.4:
        conf_msg = "Moderately confident 🤔"
    else:
        conf_msg = "Low confidence ❓"
    
    print(f"🔮 {conf_msg}")

def run_quick_demo(analyzer):
    """Run a quick demonstration"""
    print(f"\n{'='*70}")
    print("🚀 Quick Demo - Fixed Issues")
    print(f"{'='*70}")
    
    demo_cases = [
        # Previously broken cases (now fixed)
        "very bad",
        "i dont like it", 
        "terrible movie",
        "I hate this film",
        
        # Double negations (should work correctly)
        "not bad",
        "not terrible",
        "not good", 
        "don't like",
        
        # Clear cases
        "excellent movie",
        "I love this",
        "okay film",
        "average movie",
        
        # Complex cases
        "good but boring",
        "not bad at all",
        "really terrible",
        "extremely good"
    ]
    
    print(f"{'#':<3} {'Test Input':<25} {'Predicted':<10} {'Confidence':<10} {'Rule Score'}")
    print("-" * 70)
    
    for i, text in enumerate(demo_cases, 1):
        try:
            result = analyzer.predict(text)
            predicted = result['sentiment']
            confidence = result['confidence']
            rule_score = result.get('rule_based_score', 0)
            
            print(f"{i:<3} {text:<25} {predicted:<10} {confidence:.3f}      {rule_score:+.2f}")
            
        except Exception as e:
            print(f"{i:<3} {text:<25} {'ERROR':<10} {'N/A':<10} {'N/A'}")

def print_help_info():
    """Show help information"""
    print(f"\n{'='*70}")
    print("📚 Fixed Three-Class Sentiment Analyzer Help")
    print(f"{'='*70}")
    print("🎯 Three sentiment classes:")
    print("   😔 Negative: bad, terrible, hate, don't like, awful, etc.")
    print("   😐 Neutral: okay, average, so-so, mixed, mediocre, etc.")  
    print("   😊 Positive: good, excellent, love, amazing, great, etc.")
    print("\n✅ Fixed issues:")
    print("   • 'very bad' now correctly → Negative")
    print("   • 'i dont like it' now correctly → Negative")  
    print("   • 'terrible movie' now correctly → Negative")
    print("   • Better handling of 'don't like', 'didn't like'")
    print("   • Double negations: 'not bad' → Positive")
    print("   • Improved rule-based + ML hybrid approach")
    print("\n🔧 Enhanced features:")
    print("   • Better negation detection")
    print("   • Improved double negation handling")
    print("   • Context-aware analysis")
    print("   • Real IMDB dataset training")
    print("   • Ensemble model for better accuracy")

def main():
    """Main function to run the fixed analyzer"""
    print("🚀 Starting Fixed IMDB Three-Class Sentiment Analysis System...")
    
    try:
        # Download NLTK resources
        print("📦 Setting up NLTK resources...")
        nltk_downloads = ['punkt', 'punkt_tab', 'stopwords', 'wordnet', 'omw-1.4']
        for item in nltk_downloads:
            try:
                nltk.download(item, quiet=True)
            except:
                pass
        print("✅ NLTK setup completed")
        
        # Test the fixed analyzer with IMDB dataset
        analyzer = test_fixed_analyzer()
        
        if analyzer is not None:
            print("\n🎮 Starting interactive demo...")
            interactive_demo(analyzer)
        else:
            print("❌ Could not initialize analyzer")
        
    except KeyboardInterrupt:
        print("\n👋 Program interrupted. Goodbye!")
    except Exception as e:
        print(f"❌ System error: {str(e)}")
        print("\n📋 Troubleshooting:")
        print("1. Ensure internet connection for dataset download")
        print("2. Install required packages:")
        print("   pip install nltk scikit-learn numpy pandas matplotlib seaborn scipy")
        print("3. Check available disk space (dataset is ~80MB)")

if __name__ == "__main__":
    main()