# Step 2: Data Preparation (Excluding Unzipping)

# 📦 Imports
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 📐 Set Image Size and Batch Size
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
TRAIN_DIR = '../data/train'
VAL_DIR = '../data/val'  # Make sure this folder contains fish species folders

# 🧪 1. Create ImageDataGenerator with Augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # 80% training, 20% validation
)

# Load train data
train_generator = datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    color_mode='rgb',
    shuffle=True
)


# Load val data
val_generator = datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    color_mode='rgb',
    shuffle=False
)

# 🏷️ 4. Show Class Labels
print("Class Labels:", train_generator.class_indices)

# 🖼️ 5. Visualize Sample Images with Labels
images, labels = next(train_generator)

plt.figure(figsize=(12, 6))
for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.imshow(images[i])
    label_index = labels[i].argmax()
    class_name = list(train_generator.class_indices.keys())[label_index]
    plt.title(class_name)
    plt.axis('off')

plt.tight_layout()
plt.show()
