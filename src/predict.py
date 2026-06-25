from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist
import numpy as np

# Load model
model = load_model("models/digit_model.keras")

# Load test data
(_, _), (X_test, y_test) = mnist.load_data()

# Normalize
X_test = X_test / 255.0

# Select sample
sample = X_test[0]

# Reshape
sample_input = sample.reshape(1, 28, 28, 1)

# Predict
prediction = model.predict(sample_input)

predicted_digit = np.argmax(prediction)

print("Actual Digit:", y_test[0])
print("Predicted Digit:", predicted_digit)
print("Confidence:", np.max(prediction) * 100)

import matplotlib.pyplot as plt 
plt.imshow(sample, cmap="gray") 
plt.title(f"Predicted: {predicted_digit}") 
plt.axis("off") 
plt.show()
