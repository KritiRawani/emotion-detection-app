import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import joblib
import neattext.functions as nfx
import os
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('wordnet')

def load_data():
    """
    Load and prepare the emotion dataset
    Enhanced dataset with more examples and better balance
    """
    data = {
        'text': [
            # Happy examples
            "I am so happy today!",
            "This is the best day ever!",
            "I'm feeling incredibly joyful",
            "What a wonderful day",
            "I can't stop smiling",
            "This makes me so happy",
            "I'm thrilled with the results",
            "Such a delightful experience",
            "Feeling blessed and happy",
            "Today was amazing",
            "I am so happy",
            "This is great news",
            "I'm really happy about this",
            "Happiness fills my heart",
            "What a joyous occasion",
            
            # Sad examples
            "I feel sad and lonely",
            "I miss my family so much",
            "This is heartbreaking",
            "I'm feeling down today",
            "Nothing seems to go right",
            "I'm really disappointed",
            "This loss hurts deeply",
            "I feel empty inside",
            "Such a tragic situation",
            "Everything feels gloomy",
            "I'm so sad about this",
            "This makes me feel depressed",
            "Feeling blue today",
            "My heart is heavy",
            "I can't stop crying",
            
            # Angry examples
            "This makes me so angry",
            "I'm furious about this",
            "This is absolutely infuriating",
            "I can't stand this anymore",
            "This is completely unacceptable",
            "I'm fed up with this situation",
            "This drives me crazy",
            "I'm really frustrated",
            "How dare they do this",
            "This is outrageous",
            "I'm so mad right now",
            "This is making me angry",
            "My blood is boiling",
            "I'm enraged by this",
            "This is so frustrating",
            
            # Fear examples
            "I'm really scared right now",
            "This is terrifying",
            "I'm afraid of what might happen",
            "This situation is frightening",
            "I have a bad feeling about this",
            "I'm getting anxious",
            "This is making me nervous",
            "I'm worried about tomorrow",
            "That was a scary experience",
            "Fear is taking over",
            "I'm so scared",
            "This is really frightening",
            "I'm terrified of this",
            "My fears are overwhelming",
            "This is so scary",
            
            # Surprise examples
            "Wow, what a surprise!",
            "I can't believe this happened!",
            "This is unexpected",
            "I'm completely shocked",
            "That came out of nowhere",
            "I'm amazed by this",
            "This is unbelievable",
            "What an incredible surprise",
            "I never saw this coming",
            "This is mind-blowing",
            "Oh my goodness!",
            "This is such a surprise",
            "I'm totally surprised",
            "What a shock!",
            "I wasn't expecting this",
            
            # Love examples
            "I love you so much",
            "You mean everything to me",
            "My heart is full of love",
            "I'm deeply in love",
            "You make me so happy",
            "I cherish every moment with you",
            "My love grows stronger each day",
            "You're the love of my life",
            "I'm blessed to have your love",
            "Forever in love with you",
            "I love this so much",
            "This fills me with love",
            "My heart is yours",
            "Love is in the air",
            "I'm in love with this"
        ],
        'emotion': [
            # Happy
            'happy', 'happy', 'happy', 'happy', 'happy',
            'happy', 'happy', 'happy', 'happy', 'happy',
            'happy', 'happy', 'happy', 'happy', 'happy',
            
            # Sad
            'sad', 'sad', 'sad', 'sad', 'sad',
            'sad', 'sad', 'sad', 'sad', 'sad',
            'sad', 'sad', 'sad', 'sad', 'sad',
            
            # Angry
            'angry', 'angry', 'angry', 'angry', 'angry',
            'angry', 'angry', 'angry', 'angry', 'angry',
            'angry', 'angry', 'angry', 'angry', 'angry',
            
            # Fear
            'fear', 'fear', 'fear', 'fear', 'fear',
            'fear', 'fear', 'fear', 'fear', 'fear',
            'fear', 'fear', 'fear', 'fear', 'fear',
            
            # Surprise
            'surprise', 'surprise', 'surprise', 'surprise', 'surprise',
            'surprise', 'surprise', 'surprise', 'surprise', 'surprise',
            'surprise', 'surprise', 'surprise', 'surprise', 'surprise',
            
            # Love
            'love', 'love', 'love', 'love', 'love',
            'love', 'love', 'love', 'love', 'love',
            'love', 'love', 'love', 'love', 'love'
        ]
    }
    return pd.DataFrame(data)

def preprocess_text(text):
    """Enhanced text preprocessing with lemmatization"""
    # Convert to string and lowercase
    text = str(text).lower()
    
    # Basic cleaning
    text = nfx.remove_special_characters(text)
    
    # Tokenization and Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    text = ' '.join([lemmatizer.lemmatize(token) for token in tokens])
    
    return text

def train_model():
    """Train and save the emotion detection model"""
    # Create models directory if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')

    # Load and preprocess data
    print("Loading and preprocessing data...")
    df = load_data()
    df['cleaned_text'] = df['text'].apply(preprocess_text)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        df['cleaned_text'],
        df['emotion'],
        test_size=0.2,
        random_state=42,
        stratify=df['emotion']  # Ensure balanced split
    )

    # Create pipeline with TF-IDF
    print("Training model...")
    model = Pipeline([
        ('vectorizer', TfidfVectorizer(ngram_range=(1, 2), max_features=5000)),
        ('classifier', MultinomialNB())
    ])

    # Train the model
    model.fit(X_train, y_train)

    # Save the model
    print("\nSaving model...")
    joblib.dump(model, 'models/emotion_classifier.pkl')

    # Calculate and print accuracy
    accuracy = model.score(X_test, y_test)
    print(f"\nModel accuracy: {accuracy:.2f}")

    return model

if __name__ == "__main__":
    print("Training emotion detection model...")
    model = train_model()
    print("Model training completed and saved to models/emotion_classifier.pkl")