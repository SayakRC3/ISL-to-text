import tensorflow as tf
import numpy as np
import os

# Ensure the "model" directory exists
MODEL_PATH = r"D:\Project x\model"
os.makedirs(MODEL_PATH, exist_ok=True)

# Load trained model from memory (if it exists in Python)
try:
    model = tf.keras.models.load_model("model.h5")  # Try to reload the model from memory
except:
    print("⚠️ No model found in memory. Retraining might be needed.")
    exit()

# Save model properly
model.save(os.path.join(MODEL_PATH, "model.h5"))

# Save labels again
labels = np.load("labels.npy")  # Reload labels
np.save(os.path.join(MODEL_PATH, "labels.npy"), labels)

print(f"✅ Model saved at: {MODEL_PATH}\model.h5")
print(f"✅ Labels saved at: {MODEL_PATH}\labels.npy")
