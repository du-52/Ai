import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Page configuration
st.set_page_config(
    page_title="Sentiment Analysis App",
    page_icon="😊",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
.sentiment-positive {
    background-color: #d4edda;
    border: 1px solid #c3e6cb;
    border-radius: 0.25rem;
    padding: 0.75rem;
    color: #155724;
}
.sentiment-negative {
    background-color: #f8d7da;
    border: 1px solid #f5c6cb;
    border-radius: 0.25rem;
    padding: 0.75rem;
    color: #721c24;
}
.sentiment-neutral {
    background-color: #fff3cd;
    border: 1px solid #ffeaa7;
    border-radius: 0.25rem;
    padding: 0.75rem;
    color: #856404;
}
</style>
""", unsafe_allow_html=True)

st.title("😊 Sentiment Analysis Tool")
st.write("Analyze text sentiment: **Positive**, **Negative**, or **Neutral**")

# Model file paths - adjust these to match your actual file names
MODEL_FILES = {
    "Multinomial Naive Bayes": "sentiment_analysis_pipeline.joblib",
    "Logistic Regression": "logistic_regression_pipeline.joblib",
    # "Random Forest": "random_forest_pipeline.joblib"  # adjust if different
}

# Check which models are available
available_models = {}
missing_models = []

for model_name, file_path in MODEL_FILES.items():
    if os.path.exists(file_path):
        available_models[model_name] = file_path
    else:
        missing_models.append(f"{model_name} ({file_path})")

# Display available models
if available_models:
    st.success(f"✅ Found {len(available_models)} models: {', '.join(available_models.keys())}")
else:
    st.error("❌ No model files found! Please make sure your .joblib files are in the same directory.")
    st.stop()

if missing_models:
    st.warning(f"⚠️ Missing models: {', '.join(missing_models)}")

# Sidebar for model selection
st.sidebar.header("🎯 Model Selection")
selected_model_name = st.sidebar.selectbox(
    "Choose a sentiment analysis model:",
    options=list(available_models.keys()),
    help="Select which model to use for sentiment prediction"
)

# Load selected model
@st.cache_resource
def load_model(model_path):
    """Load model with caching to improve performance"""
    try:
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Load the selected model
model_path = available_models[selected_model_name]
model = load_model(model_path)

if model is None:
    st.error(f"Failed to load {selected_model_name}")
    st.stop()

st.sidebar.success(f"✅ {selected_model_name} loaded!")

# Helper function to get sentiment emoji and color
def get_sentiment_display(sentiment):
    """Return emoji, color class, and description for sentiment"""
    sentiment_lower = str(sentiment).lower()
    
    if 'positive' in sentiment_lower or sentiment == '1' or sentiment == 1:
        return "😊", "sentiment-positive", "Positive", "#28a745"
    elif 'negative' in sentiment_lower or sentiment == '-1' or sentiment == -1:
        return "😞", "sentiment-negative", "Negative", "#dc3545"
    elif 'neutral' in sentiment_lower or sentiment == '0' or sentiment == 0:
        return "😐", "sentiment-neutral", "Neutral", "#ffc107"
    else:
        # Handle any other cases
        return "🤔", "sentiment-neutral", str(sentiment), "#6c757d"

# Main content area
st.header(f"🔍 Analyzing with: {selected_model_name}")

# Create tabs for different testing options
tab1, tab2, tab3 = st.tabs(["💬 Text Analysis", "📄 Batch Analysis", "📊 Model Info"])

# Tab 1: Single Text Testing
with tab1:
    st.subheader("Analyze Single Text")
    
    # Example texts for quick testing
    st.markdown("**Quick Test Examples:**")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("😊 Positive Example"):
            st.session_state.example_text = "I absolutely love this product! It's amazing and works perfectly."
    
    with col2:
        if st.button("😞 Negative Example"):
            st.session_state.example_text = "This is terrible. I hate it and it doesn't work at all."
    
    with col3:
        if st.button("😐 Neutral Example"):
            st.session_state.example_text = "The weather today is cloudy with a chance of rain."
    
    # Text input
    default_text = st.session_state.get('example_text', '')
    user_text = st.text_area(
        "Enter text to analyze:",
        value=default_text,
        placeholder="Type or paste your text here... (e.g., 'I love this movie!' or 'This product is terrible')",
        height=120,
        help="Enter any text and the model will predict if it's positive, negative, or neutral"
    )
    
    col1, col2 = st.columns([1, 4])
    
    with col1:
        analyze_button = st.button("🔍 Analyze Sentiment", type="primary")
    
    if analyze_button and user_text.strip():
        try:
            # Make prediction
            prediction = model.predict([user_text])[0]
            emoji, css_class, sentiment_name, color = get_sentiment_display(prediction)
            
            # Display main result
            st.markdown(f"""
            <div class="{css_class}">
                <h2 style="margin: 0;">{emoji} {sentiment_name}</h2>
                <p style="margin: 0; margin-top: 10px;"><strong>Predicted Sentiment:</strong> {sentiment_name}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Get prediction probabilities if available
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba([user_text])[0]
                classes = model.classes_ if hasattr(model, 'classes_') else range(len(probabilities))
                
                # Create probability dataframe with proper sentiment names
                prob_data = []
                for i, (class_label, prob) in enumerate(zip(classes, probabilities)):
                    emoji, _, sentiment_name, color = get_sentiment_display(class_label)
                    prob_data.append({
                        'Sentiment': f"{emoji} {sentiment_name}",
                        'Probability': prob,
                        'Percentage': f"{prob*100:.1f}%",
                        'Color': color
                    })
                
                prob_df = pd.DataFrame(prob_data).sort_values('Probability', ascending=False)
                
                # Show confidence
                max_prob = prob_df.iloc[0]['Probability']
                confidence_level = "High" if max_prob > 0.7 else "Medium" if max_prob > 0.5 else "Low"
                st.info(f"**Confidence Level:** {confidence_level} ({max_prob*100:.1f}%)")
                
                # Show probabilities table
                st.write("**Detailed Probabilities:**")
                display_df = prob_df[['Sentiment', 'Percentage']].reset_index(drop=True)
                st.dataframe(display_df, use_container_width=True, hide_index=True)
                
                # Visualization
                fig, ax = plt.subplots(figsize=(10, 6))
                bars = ax.barh(prob_df['Sentiment'], prob_df['Probability'])
                
                # Color bars according to sentiment
                for i, (bar, color) in enumerate(zip(bars, prob_df['Color'])):
                    bar.set_color(color)
                    # Add percentage text
                    width = bar.get_width()
                    ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                           f'{width*100:.1f}%', ha='left', va='center', fontweight='bold')
                
                ax.set_xlabel('Probability')
                ax.set_title('Sentiment Analysis Results', fontsize=16, fontweight='bold')
                ax.set_xlim(0, 1)
                
                # Add grid for better readability
                ax.grid(axis='x', alpha=0.3)
                
                st.pyplot(fig)
                plt.close()
                
            else:
                st.success(f"**Prediction: {sentiment_name}**")
                
        except Exception as e:
            st.error(f"Error analyzing sentiment: {str(e)}")
            st.write("Please check that your model is trained for sentiment analysis.")
    
    elif analyze_button:
        st.warning("Please enter some text to analyze.")

# Tab 2: Batch Testing
with tab2:
    st.subheader("Batch Sentiment Analysis")
    
    # Sample data download
    st.markdown("**Need sample data?** Download our example CSV:")
    sample_data = pd.DataFrame({
        'text': [
            "I love this product so much!",
            "This is the worst experience ever.",
            "The weather is okay today.",
            "Amazing service, highly recommended!",
            "Not good, not bad, just average.",
            "Terrible quality, very disappointed."
        ],
        'true_sentiment': ['positive', 'negative', 'neutral', 'positive', 'neutral', 'negative']
    })
    
    sample_csv = sample_data.to_csv(index=False)
    st.download_button(
        label="📥 Download Sample CSV",
        data=sample_csv,
        file_name="sample_sentiment_data.csv",
        mime='text/csv'
    )
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload a CSV file with text data:",
        type=['csv'],
        help="CSV should have a column with text data for sentiment analysis"
    )
    
    if uploaded_file is not None:
        try:
            # Read the CSV file
            df = pd.read_csv(uploaded_file)
            st.write("**Uploaded Data Preview:**")
            st.dataframe(df.head(), use_container_width=True)
            
            # Let user select text column
            text_column = st.selectbox(
                "Select the text column:",
                options=df.columns.tolist(),
                help="Choose which column contains the text to analyze"
            )
            
            # Optional: Select true labels column for evaluation
            has_labels = st.checkbox("File contains true sentiment labels for evaluation")
            label_column = None
            
            if has_labels:
                label_column = st.selectbox(
                    "Select the sentiment label column:",
                    options=[col for col in df.columns.tolist() if col != text_column]
                )
            
            if st.button("🚀 Analyze All Texts", type="primary"):
                try:
                    # Make predictions
                    with st.spinner("Analyzing sentiments..."):
                        predictions = model.predict(df[text_column])
                        
                        # Add predictions to dataframe
                        result_df = df.copy()
                        result_df['Predicted_Sentiment'] = predictions
                        
                        # Add sentiment names and emojis
                        sentiment_info = [get_sentiment_display(pred) for pred in predictions]
                        result_df['Sentiment_Display'] = [f"{emoji} {name}" for emoji, _, name, _ in sentiment_info]
                        
                        # Add probabilities if available
                        if hasattr(model, 'predict_proba'):
                            probabilities = model.predict_proba(df[text_column])
                            classes = model.classes_ if hasattr(model, 'classes_') else range(len(probabilities[0]))
                            
                            # Add probability columns for each sentiment
                            for i, class_label in enumerate(classes):
                                _, _, sentiment_name, _ = get_sentiment_display(class_label)
                                result_df[f'{sentiment_name}_Probability'] = probabilities[:, i]
                            
                            # Add confidence level
                            max_probs = np.max(probabilities, axis=1)
                            result_df['Confidence'] = ['High' if p > 0.7 else 'Medium' if p > 0.5 else 'Low' 
                                                     for p in max_probs]
                    
                    st.success("✅ Sentiment analysis completed!")
                    
                    # Show results summary
                    sentiment_counts = result_df['Predicted_Sentiment'].value_counts()
                    
                    col1, col2, col3 = st.columns(3)
                    
                    for sentiment in sentiment_counts.index:
                        emoji, _, sentiment_name, color = get_sentiment_display(sentiment)
                        count = sentiment_counts[sentiment]
                        percentage = (count / len(result_df)) * 100
                        
                        if sentiment_name == "Positive":
                            col1.metric(f"{emoji} Positive", f"{count} ({percentage:.1f}%)")
                        elif sentiment_name == "Negative":
                            col2.metric(f"{emoji} Negative", f"{count} ({percentage:.1f}%)")
                        else:
                            col3.metric(f"{emoji} Neutral", f"{count} ({percentage:.1f}%)")
                    
                    # Show detailed results
                    st.write("**Detailed Results:**")
                    st.dataframe(result_df, use_container_width=True)
                    
                    # Download results
                    csv = result_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv,
                        file_name=f"sentiment_analysis_{selected_model_name.lower().replace(' ', '_')}.csv",
                        mime='text/csv'
                    )
                    
                    # Visualization of results
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                    
                    # Sentiment distribution pie chart
                    sentiment_display_counts = result_df['Sentiment_Display'].value_counts()
                    colors = []
                    for sentiment in sentiment_display_counts.index:
                        if "Positive" in sentiment:
                            colors.append("#28a745")
                        elif "Negative" in sentiment:
                            colors.append("#dc3545")
                        else:
                            colors.append("#ffc107")
                    
                    ax1.pie(sentiment_display_counts.values, labels=sentiment_display_counts.index, 
                           autopct='%1.1f%%', colors=colors, startangle=90)
                    ax1.set_title('Sentiment Distribution')
                    
                    # Confidence distribution
                    if 'Confidence' in result_df.columns:
                        confidence_counts = result_df['Confidence'].value_counts()
                        bars = ax2.bar(confidence_counts.index, confidence_counts.values, 
                                      color=['#28a745', '#ffc107', '#dc3545'])
                        ax2.set_title('Prediction Confidence Distribution')
                        ax2.set_ylabel('Count')
                        
                        # Add value labels on bars
                        for bar in bars:
                            height = bar.get_height()
                            ax2.text(bar.get_x() + bar.get_width()/2., height,
                                   f'{int(height)}', ha='center', va='bottom')
                    
                    st.pyplot(fig)
                    plt.close()
                    
                    # Evaluation metrics if labels are provided
                    if has_labels and label_column:
                        st.write("**Model Performance Evaluation:**")
                        
                        accuracy = accuracy_score(df[label_column], predictions)
                        st.metric("Overall Accuracy", f"{accuracy:.4f}")
                        
                        # Classification report
                        report = classification_report(df[label_column], predictions, output_dict=True)
                        report_df = pd.DataFrame(report).transpose()
                        st.write("**Detailed Classification Report:**")
                        st.dataframe(report_df, use_container_width=True)
                        
                        # Confusion Matrix
                        cm = confusion_matrix(df[label_column], predictions)
                        fig, ax = plt.subplots(figsize=(8, 6))
                        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                        ax.set_title('Confusion Matrix')
                        ax.set_ylabel('True Sentiment')
                        ax.set_xlabel('Predicted Sentiment')
                        st.pyplot(fig)
                        plt.close()
                    
                except Exception as e:
                    st.error(f"Error during batch analysis: {str(e)}")
                    
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
    else:
        st.info("📤 Upload a CSV file to start batch sentiment analysis. Make sure it has a column with text data.")

# Tab 3: Model Information
with tab3:
    st.subheader("📊 Model Information")
    
    try:
        # Display model type
        if hasattr(model, 'named_steps'):
            # It's a pipeline
            st.write("**🔧 Model Type:** Machine Learning Pipeline")
            st.write("**📋 Pipeline Components:**")
            for name, step in model.named_steps.items():
                st.write(f"- **{name}**: {type(step).__name__}")
                
            # Get the classifier
            if 'classifier' in model.named_steps:
                classifier = model.named_steps['classifier']
                st.write(f"**🎯 Main Classifier:** {type(classifier).__name__}")
        else:
            st.write(f"**🔧 Model Type:** {type(model).__name__}")
            
        # Show classes if available
        if hasattr(model, 'classes_'):
            st.write("**🏷️ Sentiment Classes:**")
            classes_info = []
            for class_label in model.classes_:
                emoji, _, sentiment_name, _ = get_sentiment_display(class_label)
                classes_info.append({
                    'Class Label': str(class_label),
                    'Sentiment': f"{emoji} {sentiment_name}"
                })
            classes_df = pd.DataFrame(classes_info)
            st.dataframe(classes_df, use_container_width=True, hide_index=True)
        
        # Feature information if it's a pipeline with vectorizer
        if hasattr(model, 'named_steps'):
            if 'tfidf' in model.named_steps:
                vectorizer = model.named_steps['tfidf']
                st.write("**📝 Text Processing Information:**")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if hasattr(vectorizer, 'vocabulary_'):
                        vocab_size = len(vectorizer.vocabulary_)
                        st.metric("Vocabulary Size", f"{vocab_size:,}")
                    
                    if hasattr(vectorizer, 'max_features'):
                        max_feat = vectorizer.max_features or "No Limit"
                        st.metric("Max Features", max_feat)
                
                with col2:
                    if hasattr(vectorizer, 'ngram_range'):
                        ngram_range = vectorizer.ngram_range
                        st.metric("N-gram Range", f"{ngram_range[0]} to {ngram_range[1]}")
                    
                    if hasattr(vectorizer, 'min_df'):
                        st.metric("Min Document Frequency", vectorizer.min_df)
                    
    except Exception as e:
        st.error(f"Error displaying model information: {str(e)}")

# Footer
st.markdown("---")
st.markdown("### 💡 Tips for Better Results:")
st.markdown("""
- **Clear text**: Remove unnecessary characters and formatting
- **Context matters**: Longer texts usually give better predictions
- **Mixed sentiments**: The model predicts the overall sentiment of the entire text
- **Domain specifics**: Models work best on text similar to their training data
""")

st.markdown("**📁 Note:** Make sure your .joblib model files are in the same directory as this app.")

# Instructions for running
st.sidebar.markdown("---")
st.sidebar.markdown("### 🚀 How to Run")
st.sidebar.markdown("""
1. Save this code as `app.py`
2. Put your .joblib files in the same folder
3. Install required packages:
   ```
   pip install streamlit pandas scikit-learn seaborn matplotlib
   ```
4. Run: `streamlit run app.py`
5. Open in browser and start analyzing!
""")