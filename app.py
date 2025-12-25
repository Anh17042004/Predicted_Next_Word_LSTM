import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model 
from tensorflow.keras.preprocessing.sequence import pad_sequences


model = load_model("predicted_word_lstm.h5")

# load tokenizer
with open("tokenizer.pickle", "rb") as handle:
    tokenizer = pickle.load(handle)


def predict_next_word(model, tokenizer, text, max_sequence_len):
  # tokenizer chuoi
  token_list = tokenizer.texts_to_sequences([text])
  if len(token_list[0]) >= max_sequence_len:
    token_list = token_list[0][-max_sequence_len:]

  # padding chỉ nhận 2D, phải them token_list vào 1 list
  token_list = pad_sequences(token_list, maxlen=max_sequence_len-1, padding="pre")
  predicted = model.predict(token_list, verbose=0)

  # tìm index từ có xác suất cao nhất
  predicted_index = np.argmax(predicted, axis=1)

  #tìm từ có xác suất cao nhất trong bộ từ vựng
  for word, index in tokenizer.word_index.items():
    if index == predicted_index:
      return word
  return None


# streamlit app
st.title("Next word predict LSTM")
input_text = st.text_input("Hãy nhập chuỗi từ", "to be or not to be")
if st.button("Dự đoán"):
    input_text  = input_text.lower()

    max_sequence_len = model.input_shape[1] + 1
    next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)

    st.write(f"Next word: {next_word}")

