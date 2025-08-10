# save_class_indices.py

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import json
import os

# ✅ Paths & settings
TRAIN_DIR = '../data/train'
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
MODELS_DIR = '../models'

# ✅ Load dataset structure
gen = ImageDataGenerator(rescale=1.0 / 255)

train_data = gen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

# ✅ Save mapping
os.makedirs(MODELS_DIR, exist_ok=True)
mapping_path = os.path.join(MODELS_DIR, "class_indices.json")

with open(mapping_path, "w") as f:
    json.dump(train_data.class_indices, f)

print(f"✅ Saved class indices to {mapping_path}")
