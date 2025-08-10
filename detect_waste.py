import tensorflow as tf
import numpy as np
import cv2
import time

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="tflite_model.tflite")
interpreter.allocate_tensors()

# Get model input/output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Dynamically get model input size
input_shape = input_details[0]['shape']
img_height, img_width = input_shape[1], input_shape[2]

# Class labels
labels = ['glass', 'metal', 'organic', 'paper', 'plastic']

# Preprocessing function
def preprocess(img):
    img = cv2.resize(img, (img_width, img_height))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Start webcam and set resolution
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("Error: Cannot open camera.")
    exit()

prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Get frame size
    frame_height, frame_width, _ = frame.shape

    # Define centered square ROI
    roi_size = 200  # Fixed size for consistency
    center_x, center_y = frame_width // 2, frame_height // 2
    x1 = center_x - roi_size // 2
    y1 = center_y - roi_size // 2
    x2 = center_x + roi_size // 2
    y2 = center_y + roi_size // 2

    # Draw ROI rectangle and instruction
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Extract ROI
    roi = frame[y1:y2, x1:x2]

    # Preprocess ROI and run inference
    input_data = preprocess(roi)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])[0]

    # Apply softmax and get prediction
    probabilities = tf.nn.softmax(output_data).numpy()
    predicted_index = np.argmax(probabilities)
    predicted_class = labels[predicted_index]
    confidence = probabilities[predicted_index]

    # Calculate FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    label_text = f"{predicted_class}: {confidence:.2f}"
    cv2.putText(frame, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2)

    # FPS and exit instruction
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 0), 2)
    cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # Display frame
    cv2.imshow('Real-Time Waste Classification', frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()







"""import tensorflow as tf
import numpy as np
import cv2
import time

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="tflite_model.tflite")
interpreter.allocate_tensors()

# Get model input/output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Dynamically get model input size
input_shape = input_details[0]['shape']
img_height, img_width = input_shape[1], input_shape[2]

# Class labels — update if you have more/less or different classes
labels = ['glass', 'metal', 'organic', 'paper','plastic']

# Preprocessing function
def preprocess(img):
    img = cv2.resize(img, (img_width, img_height))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)  # Shape: (1, h, w, 3)
    return img

# Start webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot open camera.")
    exit()

prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Preprocess and infer

    # Draw a centered ROI box on the frame
    frame_height, frame_width, _ = frame.shape
    roi_size = min(frame_width, frame_height) // 3  # Adjust size as needed
    x1 = frame_width // 2 - roi_size // 2
    y1 = frame_height // 2 - roi_size // 2
    x2 = x1 + roi_size
    y2 = y1 + roi_size

    # Draw rectangle on frame
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Extract ROI from frame
    roi = frame[y1:y2, x1:x2]

    # Preprocess ROI and infer
    input_data = preprocess(roi)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])[0]

    # Apply softmax if needed (common for classification outputs)
    probabilities = tf.nn.softmax(output_data).numpy()
    predicted_index = np.argmax(probabilities)
    predicted_class = labels[predicted_index]
    confidence = probabilities[predicted_index]

    # Calculate FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    label_text = f"{predicted_class}: {confidence:.2f}"
    cv2.putText(frame, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
            1, (0, 255, 0), 2)
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 0), 2)
    cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # Show frame
    cv2.imshow('Real-Time Waste Classification', frame)

    # Break loop on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()







import tensorflow as tf
import numpy as np
import cv2

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="tflite_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def preprocess(img):
    img = cv2.resize(img, (128, 128))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

cap = cv2.VideoCapture(0)

labels = ['plastic', 'paper', 'metal', 'organic']  # adjust based on your folders

while True:
    ret, frame = cap.read()
    if not ret:
        break

    input_data = preprocess(frame)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_class = labels[np.argmax(output_data)]

    cv2.putText(frame, predicted_class, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Waste Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
 """