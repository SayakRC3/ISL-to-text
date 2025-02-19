import os
import cv2
import numpy as np
import mediapipe as mp

# Set dataset path
DATASET_PATH = r"D:\Project x\Dataset"
SAVE_PATH = r"D:\Project x\Processed_Keypoints"

# Ensure output directory exists
os.makedirs(SAVE_PATH, exist_ok=True)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Function to extract keypoints from an image
def extract_keypoints(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    keypoints = []

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for landmark in hand_landmarks.landmark:
                keypoints.append([landmark.x, landmark.y, landmark.z])
    
    # If only one hand is detected, pad the array with zeros for the second hand
    if len(keypoints) == 21:
        keypoints += [[0, 0, 0]] * 21  # Add 21 zero-points for missing hand
    
    return np.array(keypoints).flatten() if keypoints else np.zeros(42 * 3)

# Process all images in the dataset
for class_name in os.listdir(DATASET_PATH):
    class_path = os.path.join(DATASET_PATH, class_name)
    
    if not os.path.isdir(class_path):
        continue
    
    keypoints_list = []
    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)
        
        # Read image
        image = cv2.imread(img_path)
        if image is None:
            print(f"Skipping invalid image: {img_path}")
            continue

        # Extract keypoints
        keypoints = extract_keypoints(image)
        keypoints_list.append(keypoints)

    # Save extracted keypoints as .npy file
    np.save(os.path.join(SAVE_PATH, f"{class_name}.npy"), np.array(keypoints_list))
    print(f"Saved {class_name}.npy with {len(keypoints_list)} samples.")

print("Preprocessing complete! Keypoints saved in:", SAVE_PATH)
