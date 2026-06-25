import streamlit as st
import numpy as np
import pandas as pd

from PIL import Image


from tensorflow.keras.models import load_model

# Load model
model = load_model("models/digit_model.keras")

st.sidebar.success("Model Accuracy: 99%")

st.set_page_config(
    page_title="Handwritten Digit Recognizer",
    page_icon="🔢",
    layout="wide"
)

st.title("🔢 Handwritten Digit Recognizer")

col1, col2, col3 = st.columns(3)

col1.metric("Classes", "10")
col2.metric("Input Size", "28×28")
col3.metric("Accuracy", "99%")

uploaded_file = st.file_uploader(
    "Upload a digit image",
    type=["png", "jpg", "jpeg"]
)


    
if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Image",
        width=200
    )

    image = image.convert("L")

    image = image.resize((28, 28))

    image_array = np.array(image, dtype=np.float32)
  
    image_array = 255.0 - image_array
    image_array = image_array / 255.0

    image_array = np.expand_dims(image_array, axis=0)
    image_array = np.expand_dims(image_array, axis=-1)


    try:

        with st.spinner("Predicting..."):
            prediction = model(image_array, training=False).numpy()

        st.write(prediction)


        digit = np.argmax(prediction)
        confidence = np.max(prediction) * 100

        st.success(f"Predicted Digit: {digit}")
        st.metric("Confidence", f"{confidence:.2f}%")

        prob_df = pd.DataFrame({
            "Digit": list(range(10)),
            "Probability": prediction[0]
        })

        st.subheader("Prediction Probabilities")
        st.bar_chart(prob_df.set_index("Digit"))

    except Exception as e:
        st.error(f"Error: {e}")

    