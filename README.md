# Emotion Detection Web App

A machine learning-based web application that detects emotions from text input using Python and Streamlit.

## Features

- Real-time emotion detection from text input
- Clean text preprocessing using neattext
- Machine learning pipeline using CountVectorizer + Naive Bayes
- Local model deployment (no external API required)
- Simple and intuitive Streamlit UI

## Setup Instructions

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the training script:
   ```bash
   python train_model.py
   ```
4. Launch the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Project Structure

- `app.py`: Main Streamlit application
- `train_model.py`: Script for training and saving the emotion detection model
- `data/`: Directory containing the emotion dataset
- `models/`: Directory containing the trained model file

## How it Works

1. The application uses a machine learning model trained on emotion-labeled text data
2. Text input is preprocessed using neattext for cleaning
3. The cleaned text is vectorized using CountVectorizer
4. A Naive Bayes classifier predicts the emotion
5. Results are displayed in real-time through the Streamlit interface

## Supported Emotions

- Happy
- Sad
- Angry
- Fear
- Surprise
- Love

## Requirements

See `requirements.txt` for a full list of dependencies.

## Local Development

This application runs entirely locally and doesn't require any external API calls.#   E m o t i o n - D e t e c t i o n - A p p 
 
 
