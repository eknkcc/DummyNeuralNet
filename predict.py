import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import load_model

# Load and preprocess the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Load the saved model
model = load_model("mnist_model.h5")

# Make predictions on a sample image from the test set
sample_index = 0
sample_image = x_test[sample_index]
sample_label = y_test[sample_index]

prediction = model.predict(sample_image.reshape(1, 28, 28))
predicted_label = np.argmax(prediction)

print(f"Predicted label: {predicted_label}")
print(f"Actual label: {sample_label}")