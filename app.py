import streamlit as st
import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
# Load the trained model and tokenizer
model = tf.keras.models.load_model("model1.h5")  # Replace 'your_model_path' with the actual path to your trained model
tokenizer = ("Tokenizer.pickle")# Load the tokenizer (you need to load it as it was during training)

file_path = 'adele.txt'

# Open the file in read mode
with open(file_path, 'r') as file:
    # Read the contents of the file
    data = file.read()
tk = Tokenizer()
data = data.splitlines()
encoded_text = tk.texts_to_sequences(data)
tk.fit_on_texts(data)
vocab_size = len(tk.word_counts) + 1
datalist = []
for d in encoded_text:
  if len(d)>1:
    for i in range(2, len(d)):
      datalist.append(d[:i])
      print(d[:i])

max_length = 20
sequences = pad_sequences(datalist, maxlen=max_length, padding='pre')
X = sequences[:, :-1]
y = sequences[:, -1]
y = to_categorical(y, num_classes=vocab_size)
seq_length = X.shape[1]

# Define the maximum length for generated sequences
max_length = 20
token = tokenizer

# Function to generate poetry
poetry_length = 10
seq_length = X.shape[1]
def generate_poetry(seed_text, n_lines):
  for i in range(n_lines):
    text = []
    for i in range(poetry_length):
      encoded = tk.texts_to_sequences([seed_text])
      encoded = pad_sequences(encoded, maxlen=seq_length, padding='pre')

      y_pred = np.argmax(model.predict(encoded), axis=-1)

      predicted_word = ""
      for word, index in tk.word_index.items():
        if index == y_pred:
          predicted_word = word
          break

      seed_text = seed_text + ' ' + predicted_word
      text.append(predicted_word)

    seed_text = text[-1]
    text = ' '.join(text)
    st.write(text)

# Streamlit App
st.title("Poetry Generation App")

# User input for seed text
seed_text = st.text_input("Enter seed text:")

# Generate poetry on button click
if st.button("Generate Poetry"):
    generated_poetry = generate_poetry(seed_text, 5)