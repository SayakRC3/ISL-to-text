import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp

# Load trained model and labels
MODEL_PATH = r"D:\Project x\model\model.h5"
LABELS_PATH = r"D:\Project x\model\labels.npy"

model = tf.keras.models.load_model(MODEL_PATH)
labels = np.load(LABELS_PATH)

# Confidence threshold
CONFIDENCE_THRESHOLD = 80  # Only add letters if confidence is above this

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# List to store the word being formed
word_list = []  # Use a list instead of a string

# Function to extract exactly 42 keypoints (21 per hand) with padding
def extract_keypoints(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    keypoints = []

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            hand_keypoints = []
            for i in range(21):  # Extract exactly 21 keypoints per hand
                hand_keypoints.append([
                    hand_landmarks.landmark[i].x, 
                    hand_landmarks.landmark[i].y, 
                    hand_landmarks.landmark[i].z
                ])
            keypoints.append(hand_keypoints)

    # Ensure exactly 42 keypoints (21 per hand)
    while len(keypoints) < 2:  # If only one hand is detected, pad with zero keypoints for the second hand
        keypoints.append([[0, 0, 0]] * 21)

    # If no hands detected, return None
    if len(keypoints) == 0:
        return None

    # Flatten the keypoints list
    keypoints = np.array(keypoints).flatten()

    # Ensure the final shape is exactly (126,)
    if keypoints.shape[0] != 126:
        return None  # Skip frames with incorrect shape

    return keypoints  # Shape (126,)

# Start webcam for real-time gesture recognition
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the image for a mirror effect
    frame = cv2.flip(frame, 1)
    
    # Extract keypoints
    keypoints = extract_keypoints(frame)

    if keypoints is None:
        # If no hand detected, do not display anything
        pass  
    else:
        # Reshape keypoints for model input
        keypoints = keypoints.reshape(1, 1, 126)  # Ensure correct shape

        # Predict gesture
        prediction = model.predict(keypoints)
        predicted_index = np.argmax(prediction)
        confidence = np.max(prediction) * 100  # Convert confidence to percentage

        # Only add the letter if confidence is above the threshold
        if confidence >= CONFIDENCE_THRESHOLD:
            predicted_label = labels[predicted_index]

            # Avoid duplicate letters (only add when a change is detected)
            if len(word_list) == 0 or (word_list[-1] != predicted_label):
                word_list.append(predicted_label)

            # Display the current gesture
            text = f"Gesture: {predicted_label} ({confidence:.2f}%)"
            cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the formed word
    word = "".join(word_list)  # Convert list to string for display
    cv2.putText(frame, f"Word: {word}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

    # Draw hand landmarks
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Show the frame
    cv2.imshow("ISL Gesture Recognition", frame)

    # Handle keyboard input
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):  # Quit program
        break
    elif key == ord('c'):  # Clear word
        word_list.clear()  # Reset the word list
    elif key == ord(' '):  # Add space between words
        word_list.append(" ")
    elif key == 8:  # Backspace to remove last letter
        if word_list:
            word_list.pop()

# Release resources
cap.release()
cv2.destroyAllWindows()
