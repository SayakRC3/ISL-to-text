import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Load the training history from the trained model
MODEL_PATH = r"D:\Project x\model\model.h5"

# Load the model
model = tf.keras.models.load_model(MODEL_PATH)

# Extract training history from model
history = model.history.history

# Plot Accuracy
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history['accuracy'], label="Training Accuracy")
plt.plot(history['val_accuracy'], label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Model Accuracy")
plt.legend()

# Plot Loss
plt.subplot(1, 2, 2)
plt.plot(history['loss'], label="Training Loss")
plt.plot(history['val_loss'], label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Model Loss")
plt.legend()

# Show the plots
plt.show()
