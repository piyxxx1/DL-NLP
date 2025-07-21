import streamlit as st
import numpy as np
import random
import pickle
from tensorflow.keras.models import load_model # type: ignore
from nltk.tokenize import RegexpTokenizer

# --- Load data and model ---
model = load_model("next_word_model.h5") 
with open("unique_tokens.pkl", "rb") as f:
    unique_tokens = pickle.load(f)
with open("token_index.pkl", "rb") as f:
    unique_token_index = pickle.load(f)

tokenizer = RegexpTokenizer(r'\w+')
n_words = 10

# --- Prediction Logic ---
def predict_next_word(input_text, n_best):
    input_text = input_text.lower()
    x = np.zeros((1, n_words, len(unique_tokens)))
    words = input_text.split()
    for i, word in enumerate(words):
        if word in unique_token_index:
            x[0, i, unique_token_index[word]] = 1
    predictions = model.predict(x)[0]
    return np.argpartition(predictions, -n_best)[-n_best:]

def generate_text(input_text, text_length, creativity=3):
    word_sequence = input_text.split()
    current = 0
    for _ in range(text_length):
        sub_sequence = " ".join(tokenizer.tokenize(" ".join(word_sequence).lower())[current:current+n_words])
        try:
            choice = unique_tokens[random.choice(predict_next_word(sub_sequence, creativity))]
        except:
            choice = random.choice(unique_tokens)
        word_sequence.append(choice)
        current += 1
    return " ".join(word_sequence)

# --- Streamlit UI ---
st.title("🧠 Next Word Predictor")
st.write("Generate text using an LSTM model trained on news data.")

seed_text = st.text_input("Enter the seed text (at least 10 words):")
length = st.slider("Number of words to generate", 10, 200, 50)
creativity = st.slider("Creativity (Top-N predictions)", 1, 10, 5)

if st.button("Generate Text"):
    if len(seed_text.split()) < n_words:
        st.warning(f"Please enter at least {n_words} words.")
    else:
        with st.spinner("Generating..."):
            result = generate_text(seed_text, length, creativity)
        st.success("Here is your generated text:")
        st.write(result)
