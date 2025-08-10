# model_training_cnn.py

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
import os

# 📁 Dataset directories (adjust if needed)
TRAIN_DIR = '../data/train'
VAL_DIR = '../data/val'
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10

# 🔁 Data generators
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    color_mode='rgb',
    shuffle=True
)
import json
with open("../models/class_indices.json", "w") as f:
    json.dump(train_generator.class_indices, f)
print("✅ Saved class indices to ../models/class_indices.json")


val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    color_mode='rgb',
    shuffle=False
)

# 🧱 CNN Model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(2, 2),
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Flatten(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dense(train_generator.num_classes, activation='softmax')  # Output layer
])

# ⚙️ Compile
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 💾 Callbacks
os.makedirs('../models', exist_ok=True)

checkpoint = ModelCheckpoint('../models/cnn_model_best.h5', save_best_only=True, monitor='val_accuracy')
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# 🏋️‍♀️ Train
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=[checkpoint, early_stop]
)

# 📊 Plot Accuracy
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Val')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# ✅ Save final model
model.save('../models/cnn_model_final.h5')
print("✅ Model training complete and saved.")
