import streamlit as st
import joblib
import os
from train_model import preprocess_text
import pandas as pd

# Page config
st.set_page_config(
    page_title="Emotion Detection App",
    page_icon="🎭",
    layout="centered"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stTextInput > div > div > input {
        font-size: 1.2rem;
    }
    .emotion-result {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

def load_model():
    """Load the trained model"""
    model_path = 'models/emotion_classifier.pkl'
    if not os.path.exists(model_path):
        st.error("Model not found! Please run train_model.py first.")
        return None
    return joblib.load(model_path)

def get_emotion_color(emotion):
    """Return color code for each emotion"""
    colors = {
        'happy': '#FFD700',    # Gold
        'sad': '#4169E1',      # Royal Blue
        'angry': '#FF4500',    # Red-Orange
        'fear': '#800080',     # Purple
        'surprise': '#32CD32', # Lime Green
        'love': '#FF69B4'      # Hot Pink
    }
    return colors.get(emotion, '#808080')

def main():
    st.title("🎭 Emotion Detection App")
    st.write("Enter any text, and I'll detect the emotion behind it!")

    # Load model
    model = load_model()
    if model is None:
        return

    # Text input
    text_input = st.text_area(
        "Type your text here:",
        height=100,
        placeholder="Example: I am so happy today!"
    )

    if text_input:
        # Preprocess the input
        cleaned_text = preprocess_text(text_input)
        
        # Make prediction
        prediction = model.predict([cleaned_text])[0]
        probability = max(model.predict_proba([cleaned_text])[0]) * 100

        # Display results
        st.markdown(f"""
            <div style="
                background-color: {get_emotion_color(prediction)};
                padding: 20px;
                border-radius: 10px;
                margin: 10px 0;
                color: white;
                text-align: center;
                font-size: 24px;
                font-weight: bold;
            ">
                Detected Emotion: {prediction.upper()}
                <br>
                <span style="font-size: 18px;">Confidence: {probability:.2f}%</span>
            </div>
        """, unsafe_allow_html=True)

        # Show preprocessing details
        if st.checkbox("Show text preprocessing details"):
            st.write("Original text:", text_input)
            st.write("Preprocessed text:", cleaned_text)

    # Add information about supported emotions
    with st.expander("ℹ️ Supported Emotions"):
        st.write("""
        This app can detect the following emotions:
        - 😊 Happy
        - 😢 Sad
        - 😠 Angry
        - 😨 Fear
        - 😲 Surprise
        - ❤️ Love
        """)

    # Footer
    st.markdown("---")
    st.markdown(
        "Built with ❤️ using Python, Streamlit, and Machine Learning"
    )

if __name__ == "__main__":
    main()