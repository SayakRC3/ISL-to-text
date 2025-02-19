import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Set processed keypoints directory
DATASET_PATH = r"D:\Project x\Processed_Keypoints"

# Load keypoints and labels
X, y, labels = [], [], []

for idx, file in enumerate(os.listdir(DATASET_PATH)):
    if file.endswith(".npy"):
        class_name = file.split(".npy")[0]
        data = np.load(os.path.join(DATASET_PATH, file))

        # Append data and corresponding labels
        X.extend(data)
        y.extend([idx] * len(data))
        labels.append(class_name)

# Convert to NumPy arrays
X = np.array(X, dtype=np.float32)
y = np.array(y)

# Reshape X for LSTM (Adding time dimension)
X = X.reshape(X.shape[0], 1, X.shape[1])  # Shape: (samples, time_steps=1, features)

# Convert labels to categorical (one-hot encoding)
y = to_categorical(y, num_classes=len(labels))

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the LSTM model
model = Sequential([
    LSTM(64, return_sequences=True, activation='relu', input_shape=(X.shape[1], X.shape[2])),
    Dropout(0.2),
    LSTM(128, return_sequences=False, activation='relu'),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dense(len(labels), activation='softmax')
])

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
history = model.fit(X_train, y_train, epochs=200, batch_size=32, validation_data=(X_test, y_test))

# Create "model" directory if not exists
os.makedirs("model", exist_ok=True)

# Save model and labels in "model" folder
model.save("model/model.h5")
np.save("model/labels.npy", np.array(labels))

# Predictions for error calculations
y_pred = model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)
y_test_labels = np.argmax(y_test, axis=1)

# Calculate error metrics
mse = mean_squared_error(y_test_labels, y_pred_labels)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_labels, y_pred_labels)

# Convert accuracy to percentage
accuracy = history.history['accuracy'][-1] * 100
val_accuracy = history.history['val_accuracy'][-1] * 100

# Print results
print(f"✅ Final Training Accuracy: {accuracy:.2f}%")
print(f"✅ Final Validation Accuracy: {val_accuracy:.2f}%")
print(f"📉 Mean Squared Error (MSE): {mse:.4f}")
print(f"📉 Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"📉 Mean Absolute Error (MAE): {mae:.4f}")

# Plot training accuracy & loss
plt.figure(figsize=(12, 5))

# Accuracy Plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label="Training Accuracy")
plt.plot(history.history['val_accuracy'], label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Model Accuracy")
plt.legend()

# Loss Plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label="Training Loss")
plt.plot(history.history['val_loss'], label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Model Loss")
plt.legend()

plt.show()
