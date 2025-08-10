import tensorflow as tf
import numpy as np
from PIL import Image

# Load class labels
with open("labels.txt", "r") as f:
    class_labels = [line.strip() for line in f.readlines()]

# Load the TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path="tflite_model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Preprocess image
def preprocess_image(image_path, img_size=128):
    img = Image.open(image_path).convert("RGB").resize((img_size, img_size))
    img = np.array(img, dtype=np.float32) / 255.0
    img = np.expand_dims(img, axis=0)  # Shape: (1, 128, 128, 3)
    return img

# Predict function
def predict(image_path):
    input_data = preprocess_image(image_path)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])[0]

    predicted_index = np.argmax(output_data)
    predicted_label = class_labels[predicted_index]
    confidence = output_data[predicted_index]

    print(f"\n Prediction: {predicted_label} ({confidence:.2f} confidence)")

# Replace with your image path
predict("data/organic/O_11.jpg")

