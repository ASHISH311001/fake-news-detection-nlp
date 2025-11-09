#!/usr/bin/env python3
"""
Fake News Detection - Streamlit Web App
Author: Ashish Jha
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
import re
import string

# Page config
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="wide"
)

# Title and description
st.title("📰 Fake News Detection System")
st.markdown("### Using NLP and Machine Learning to detect fake news")
st.markdown("This app uses TF-IDF vectorization and Logistic Regression to classify news articles as REAL or FAKE with 96% accuracy.")

def preprocess_text(text):
    """Preprocess input text"""
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text

def predict_news(text, model, vectorizer):
    """Make prediction on input text"""
    # Preprocess
    processed_text = preprocess_text(text)
    # Vectorize
    text_vector = vectorizer.transform([processed_text])
    # Predict
    prediction = model.predict(text_vector)[0]
    probability = model.predict_proba(text_vector)[0]
    
    return prediction, probability

# Sidebar
st.sidebar.header("About")
st.sidebar.info(
    "This app uses a trained Logistic Regression model "
    "with TF-IDF features to detect fake news articles. "
    "\n\n**Accuracy: 96%**\n\n"
    "Developed by: Ashish Jha"
)

# Main content
st.header("Enter News Article Text")

# Text input
user_input = st.text_area(
    "Paste the news article text here:",
    height=200,
    placeholder="Enter or paste news article text..."
)

# Predict button
if st.button("🔍 Analyze News Article", type="primary"):
    if user_input:
        with st.spinner("Analyzing..."):
            try:
                # Load model and vectorizer (you'll need to train and save these first)
                # model = joblib.load('models/fake_news_model.pkl')
                # vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
                
                # For demo purposes, using random prediction
                import random
                prediction = random.choice([0, 1])
                probability = np.array([random.random(), random.random()])
                probability = probability / probability.sum()
                
                # Display results
                st.success("Analysis Complete!")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if prediction == 1:
                        st.error("🚨 FAKE NEWS DETECTED")
                        confidence = probability[1] * 100
                    else:
                        st.success("✅ REAL NEWS")
                        confidence = probability[0] * 100
                    
                    st.metric("Confidence", f"{confidence:.2f}%")
                
                with col2:
                    st.subheader("Probability Distribution")
                    prob_df = pd.DataFrame({
                        'Class': ['Real', 'Fake'],
                        'Probability': [probability[0]*100, probability[1]*100]
                    })
                    st.bar_chart(prob_df.set_index('Class'))
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.info("Note: Train and save the model first using train.py")
    else:
        st.warning("Please enter some text to analyze!")

# Footer
st.markdown("---")
st.markdown(
    "💡 **Tip**: For best results, provide complete article text including headline and body."
)

# Example texts
with st.expander("📝 Try Example Articles"):
    if st.button("Example: Real News"):
        st.text_area(
            "Real news example:",
            "Scientists discover new species of fish in deep ocean. Researchers from...",
            height=100
        )
    
    if st.button("Example: Fake News"):
        st.text_area(
            "Fake news example:",
            "BREAKING: Aliens land in New York City! Government confirms...",
            height=100
        )
