import streamlit as st
import tensorflow as tf
import numpy as np

st.title("TensorFlow Test")

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28,28,1)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10, activation="softmax")
])

x = np.random.rand(1,28,28,1)

st.write("Before")

pred = model.predict(x, verbose=0)

st.write("After")
st.write(pred)