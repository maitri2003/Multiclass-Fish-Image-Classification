# train_transfer_learning.py

import tensorflow as tf
from tensorflow.keras.applications import VGG16, ResNet50, MobileNet, InceptionV3, EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import os

# ✅ Constants
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
TRAIN_DIR = '../data/train'
VAL_DIR = '../data/val'
MODELS_DIR = '../models'

# ✅ Load Data
train_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
val_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    color_mode='rgb',
    shuffle=True
)
import json
with open("../models/class_indices.json", "w") as f:
    json.dump(train_data.class_indices, f)
print("✅ Saved class indices to ../models/class_indices.json")


val_data = val_gen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    color_mode='rgb',
    shuffle=False
)

# ✅ Pretrained model dictionary
pretrained_models = {
    "VGG16": VGG16,
    "ResNet50": ResNet50,
    "MobileNet": MobileNet,
    "InceptionV3": InceptionV3,
    "EfficientNetB0": EfficientNetB0
}

# ✅ Train each model
for name, model_fn in pretrained_models.items():
    print(f"\n🔁 Training with {name}...")

    try:
        # 🧠 Load base model
        base_model = model_fn(
            weights='imagenet',
            include_top=False,
            input_shape=(224, 224, 3)
        )
        base_model.trainable = False  # Freeze all convolutional layers

        # 🔧 Add custom classification head
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.3)(x)
        predictions = Dense(train_data.num_classes, activation='softmax')(x)

        model = Model(inputs=base_model.input, outputs=predictions)

        # ⚙️ Compile
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # 💾 Callbacks
        os.makedirs(MODELS_DIR, exist_ok=True)
        model_path = os.path.join(MODELS_DIR, f'{name}_best.h5')
        checkpoint = ModelCheckpoint(model_path, save_best_only=True, monitor='val_accuracy')
        early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

        # 🏋️‍♂️ Train
        history = model.fit(
            train_data,
            epochs=EPOCHS,
            validation_data=val_data,
            callbacks=[checkpoint, early_stop]
        )

        print(f"✅ {name} training complete. Model saved to: {model_path}")

    except Exception as e:
        print(f"❌ Failed to train {name}: {e}")
