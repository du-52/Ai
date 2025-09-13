import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time

# Page configuration
st.set_page_config(
    page_title="Movie Review Sentiment Analyzer",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FF6B6B;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .model-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .prediction-positive {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #c3e6cb;
        margin: 1rem 0;
    }
    .prediction-negative {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #f5c6cb;
        margin: 1rem 0;
    }
    .confidence-high { color: #28a745; font-weight: bold; }
    .confidence-medium { color: #ffc107; font-weight: bold; }
    .confidence-low { color: #dc3545; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    """Load the trained models"""
    models = {}
    try:
        models['Multinomial Naive Bayes'] = joblib.load('sentiment_analysis_pipeline.joblib')
        st.success("✅ Multinomial Naive Bayes model loaded successfully!")
    except FileNotFoundError:
        st.warning("⚠️ sentiment_analysis_pipeline.joblib not found")
        models['Multinomial Naive Bayes'] = None
    
    try:
        models['Logistic Regression'] = joblib.load('logistic_regression_pipeline.joblib')
        st.success("✅ Logistic Regression model loaded successfully!")
    except FileNotFoundError:
        st.warning("⚠️ logistic_regression_pipeline.joblib not found")
        models['Logistic Regression'] = None

    try:
        models['Support Vector Machine'] = joblib.load('linear_svm_pipeline.joblib')
        st.success("✅ Support Vector Machine model loaded successfully!")
    except FileNotFoundError:
        st.warning("⚠️ linear_svm_pipeline.joblib not found")
        models['Support Vector Machine'] = None
    
    return models

def get_confidence_level(prob):
    """Determine confidence level based on probability"""
    max_prob = max(prob)
    if max_prob > 0.8:
        return "High", "confidence-high"
    elif max_prob > 0.6:
        return "Medium", "confidence-medium"
    else:
        return "Low", "confidence-low"

def predict_sentiment(text, model, model_name):
    """Predict sentiment using the given model"""
    try:
        # Debug: Show the input text
        st.write(f"**Debug - {model_name}:** Analyzing text: '{text[:100]}...' (length: {len(text)})")
        
        # Validate input
        if not text or len(text.strip()) == 0:
            st.warning(f"{model_name}: Empty text input!")
            return None
            
        # Get prediction
        prediction = model.predict([text])[0]
        st.write(f"**Debug - {model_name}:** Raw prediction: {prediction}")
        
        # Get prediction probabilities if available
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba([text])[0]
            st.write(f"**Debug - {model_name}:** Probabilities: {proba}")
            confidence = max(proba)
            neg_prob = proba[0]
            pos_prob = proba[1]
        else:
            # For models without predict_proba, use decision function or default
            if hasattr(model, 'decision_function'):
                decision = model.decision_function([text])[0]
                st.write(f"**Debug - {model_name}:** Decision function: {decision}")
                # Convert decision function to pseudo-probability
                confidence = min(abs(decision) / 2, 0.95)  # Cap at 95%
                if decision > 0:
                    pos_prob = 0.5 + confidence/2
                    neg_prob = 0.5 - confidence/2
                else:
                    neg_prob = 0.5 + confidence/2
                    pos_prob = 0.5 - confidence/2
            else:
                confidence = 0.75  # Default confidence
                pos_prob = 0.75 if prediction == 1 else 0.25
                neg_prob = 0.25 if prediction == 1 else 0.75
        
        sentiment = "Positive" if prediction == 1 else "Negative"
        
        return {
            'sentiment': sentiment,
            'confidence': confidence,
            'positive_prob': pos_prob,
            'negative_prob': neg_prob,
            'model': model_name
        }
    except Exception as e:
        st.error(f"Error with {model_name}: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return None

def create_confidence_chart(results):
    """Create a confidence comparison chart"""
    if not results:
        return None
    
    models = [r['model'] for r in results if r is not None]
    confidences = [r['confidence'] for r in results if r is not None]
    sentiments = [r['sentiment'] for r in results if r is not None]
    
    colors = ['#FF6B6B' if s == 'Negative' else '#4ECDC4' for s in sentiments]
    
    fig = go.Figure(data=[
        go.Bar(x=models, y=confidences, marker_color=colors, text=[f"{c:.2%}" for c in confidences], textposition='auto')
    ])
    
    fig.update_layout(
        title="Model Confidence Comparison",
        yaxis_title="Confidence Level",
        xaxis_title="Models",
        yaxis=dict(range=[0, 1]),
        showlegend=False,
        height=400
    )
    
    return fig

def create_probability_chart(result):
    """Create a probability distribution chart for a single model"""
    if not result:
        return None
    
    labels = ['Negative', 'Positive']
    values = [result['negative_prob'], result['positive_prob']]
    colors = ['#FF6B6B', '#4ECDC4']
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.3,
        marker_colors=colors,
        textinfo='label+percent',
        textfont_size=12
    )])
    
    fig.update_layout(
        title=f"{result['model']} - Probability Distribution",
        showlegend=True,
        height=300
    )
    
    return fig

# Main app
def main():
    # Header
    st.markdown('<h1 class="main-header">🎬 Movie Review Sentiment Analyzer</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Load models
    with st.spinner("Loading AI models..."):
        models = load_models()
    
    # Sidebar
    st.sidebar.title("🔧 Model Settings")
    
    # Model selection
    available_models = [name for name, model in models.items() if model is not None]
    
    if not available_models:
        st.error("❌ No models found! Please ensure the .joblib files are in the same directory as this script.")
        st.stop()
    
    selected_models = st.sidebar.multiselect(
        "Select Models to Use:",
        available_models,
        default=available_models
    )
    
    if not selected_models:
        st.warning("Please select at least one model from the sidebar.")
        return
    
    # Model information
    st.sidebar.markdown("### 📊 Model Information")
    model_info = {
        'Multinomial Naive Bayes': {
            'accuracy': '86.33%',
            'description': 'Probabilistic classifier based on Bayes theorem'
        },
        'Logistic Regression': {
            'accuracy': '89.25%',
            'description': 'Linear model for binary classification'
        }
    }
    
    for model_name in selected_models:
        if model_name in model_info:
            with st.sidebar.expander(f"{model_name}"):
                st.write(f"**Accuracy:** {model_info[model_name]['accuracy']}")
                st.write(f"**Description:** {model_info[model_name]['description']}")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📝 Enter Your Movie Review")
        
        # Text input options
        input_method = st.radio("Choose input method:", ["Type Review", "Use Sample Reviews"])
        
        if input_method == "Type Review":
            user_input = st.text_area(
                "Write your movie review here:",
                placeholder="e.g., This movie was absolutely amazing! The plot was engaging and the acting was superb...",
                height=150
            )
        else:
            # Sample reviews
            sample_reviews = [
                "This movie was absolutely terrible. The plot made no sense and the acting was horrible.",
                "Amazing film! Great story, excellent performances, and beautiful cinematography.",
                "It was okay, not great but not terrible either. Pretty average movie.",
                "I loved every minute of it! One of the best movies I've ever seen.",
                "Boring and predictable. I fell asleep halfway through.",
                "The movie was good, but the ending was disappointing.",
                "Incredible action sequences and great character development!",
                "Not worth the money. Poor script and bad direction."
            ]
            
            selected_sample = st.selectbox("Choose a sample review:", [""] + sample_reviews)
            user_input = st.text_area(
                "Selected review (you can edit it):",
                value=selected_sample,
                height=100
            )
    
    with col2:
        st.subheader("⚡ Quick Actions")
        
        # Batch analysis
        if st.button("🔄 Analyze Multiple Reviews", help="Analyze several reviews at once"):
            st.session_state.show_batch = True
        
        # Clear button
        if st.button("🗑️ Clear All", help="Clear all inputs and results"):
            st.session_state.clear()
            st.rerun()
    
    # Analysis button
    if st.button("🚀 Analyze Sentiment", type="primary", use_container_width=True):
        if user_input.strip():
            # Perform analysis
            with st.spinner("Analyzing sentiment..."):
                results = []
                
                # Create columns for results
                cols = st.columns(len(selected_models))
                
                for i, model_name in enumerate(selected_models):
                    with cols[i]:
                        model = models[model_name]
                        result = predict_sentiment(user_input, model, model_name)
                        if result:
                            results.append(result)
                            
                            # Display result card
                            sentiment = result['sentiment']
                            confidence = result['confidence']
                            confidence_level, confidence_class = get_confidence_level([result['negative_prob'], result['positive_prob']])
                            
                            if sentiment == "Positive":
                                st.markdown(f"""
                                <div class="prediction-positive">
                                    <h3>🎉 {model_name}</h3>
                                    <h2>😊 Positive</h2>
                                    <p><strong>Confidence:</strong> <span class="{confidence_class}">{confidence:.1%} ({confidence_level})</span></p>
                                    <p><strong>Positive:</strong> {result['positive_prob']:.1%} | <strong>Negative:</strong> {result['negative_prob']:.1%}</p>
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.markdown(f"""
                                <div class="prediction-negative">
                                    <h3>📊 {model_name}</h3>
                                    <h2>😞 Negative</h2>
                                    <p><strong>Confidence:</strong> <span class="{confidence_class}">{confidence:.1%} ({confidence_level})</span></p>
                                    <p><strong>Positive:</strong> {result['positive_prob']:.1%} | <strong>Negative:</strong> {result['negative_prob']:.1%}</p>
                                </div>
                                """, unsafe_allow_html=True)
                
                # Comparison charts
                if len(results) > 1:
                    st.markdown("---")
                    st.subheader("📈 Model Comparison")
                    
                    # Confidence comparison
                    conf_chart = create_confidence_chart(results)
                    if conf_chart:
                        st.plotly_chart(conf_chart, use_container_width=True)
                
                # Individual probability charts
                if results:
                    st.markdown("---")
                    st.subheader("🎯 Detailed Analysis")
                    
                    chart_cols = st.columns(len(results))
                    for i, result in enumerate(results):
                        with chart_cols[i]:
                            prob_chart = create_probability_chart(result)
                            if prob_chart:
                                st.plotly_chart(prob_chart, use_container_width=True)
        else:
            st.warning("Please enter a movie review to analyze!")
    
    # Batch analysis section
    if st.session_state.get('show_batch', False):
        st.markdown("---")
        st.subheader("📊 Batch Analysis")
        
        batch_reviews = st.text_area(
            "Enter multiple reviews (one per line):",
            placeholder="Review 1\nReview 2\nReview 3...",
            height=200
        )
        
        if st.button("Analyze Batch"):
            if batch_reviews.strip():
                reviews_list = [review.strip() for review in batch_reviews.split('\n') if review.strip()]
                
                if reviews_list:
                    batch_results = []
                    progress_bar = st.progress(0)
                    
                    for i, review in enumerate(reviews_list):
                        review_results = {}
                        for model_name in selected_models:
                            model = models[model_name]
                            result = predict_sentiment(review, model, model_name)
                            if result:
                                review_results[model_name] = result['sentiment']
                        
                        batch_results.append({
                            'Review': review[:100] + '...' if len(review) > 100 else review,
                            **review_results
                        })
                        
                        progress_bar.progress((i + 1) / len(reviews_list))
                    
                    # Display batch results
                    df = pd.DataFrame(batch_results)
                    st.dataframe(df, use_container_width=True)
                    
                    # Summary statistics
                    summary_data = []
                    for model_name in selected_models:
                        if model_name in df.columns:
                            positive_count = (df[model_name] == 'Positive').sum()
                            negative_count = (df[model_name] == 'Negative').sum()
                            summary_data.append({
                                'Model': model_name,
                                'Positive': positive_count,
                                'Negative': negative_count,
                                'Total': len(df)
                            })
                    
                    if summary_data:
                        st.subheader("📋 Summary")
                        summary_df = pd.DataFrame(summary_data)
                        st.dataframe(summary_df, use_container_width=True)
    

if __name__ == "__main__":
    main()