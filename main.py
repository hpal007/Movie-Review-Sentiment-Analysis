import numpy as np
import tensorflow
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

# Load model
model = load_model('simple_rnn_imdb.h5')
model.summary()


## Mapping of word index for understanding 
word_index = imdb.get_word_index()
reversed_word_index = {v:k for k,v in word_index.items()}


def decoding_the_review(encoded_review):
    return " ".join([reversed_word_index.get(i-3, '?') for i in encoded_review])

# Function to preprocess the user input 

def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review],maxlen=500)
    return padded_review


# Prediction Function

def predict_sentiment(review):
    preprocessed_input = preprocess_text(review)
    prediction = model.predict(preprocessed_input)
    sentiment = 'Positive' if prediction[0][0]> 0.5 else 'Negative'
    return sentiment, prediction[0][0]


# Streamlit APP
import streamlit as st

st.title("Review Sentiment Analysis")
st.write("Enter a review for movie to classify it as Positive or Negative")

user_input = st.text_area('Movie Review')

if st.button('Classify'):
    preprocess_input = preprocess_text(user_input)
    prediction = model.predict(preprocess_input)
    sentiment = 'Positive' if prediction[0][0]> 0.5 else 'Negative'

    # Display the result 
    st.write(f'Sentiment: {sentiment}')
    st.write(f'Prediction Score: {prediction[0][0]}')
else:
    st.write('Please enter a movie review')

