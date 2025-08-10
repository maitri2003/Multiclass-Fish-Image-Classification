# Multiclass Fish Image Classification

This project is a deep learning-based image classification system designed to identify different fish species from images. It uses Convolutional Neural Networks (CNN) and Transfer Learning models such as VGG16, ResNet50, InceptionV3, MobileNet, and EfficientNetB0.

## 1. Dataset

The dataset contains images of multiple fish species, organized into folders (one folder per class).

### 1.1 Download Dataset

You can download the dataset from:https://drive.google.com/drive/folders/1iKdOs4slf3XvNWkeSfsszhPRggfJ2qEd

After downloading, place it inside:
`project_root/data/`

Folder structure should look like:

```
data/
 ├── train/
 ├── val/
```

* `train/` contains training images organized in subfolders by class name
* `val/` contains validation images organized in subfolders by class name

---

## 2. Pre-trained Models

We have trained multiple models and saved them in `.h5` format. Due to their large size, they are stored using **Git Large File Storage (Git LFS)** instead of being stored directly in the GitHub repository history.

### 2.1 Download Trained Models with Git LFS

To automatically download models when cloning the repo:

1. Install Git LFS:

```
git lfs install
```

2. Clone the repository:

```
git clone https://github.com/maitri2003/Multiclass-Fish-Image-Classification.git
```

3. If models are not downloaded automatically, run:

```
git lfs pull
```

After downloading, you should see large `.h5` files in:

```
project_root/models/
```

If you see small text files instead of large `.h5` files, it means the LFS files haven’t been pulled yet. Use `git lfs pull` to fetch them.

---

## 3. Requirements

This project uses Python and TensorFlow/Keras for model training and evaluation.

### 3.1 Install Dependencies

You can install all dependencies using:

```
pip install -r requirements.txt
```

If `requirements.txt` is not present, you can create it from your current environment using:

```
pip freeze > requirements.txt
```

---

## 4. How to Run the Project

### 4.1 Clone the Repository

```
git clone https://github.com/maitri2003/Multiclass-Fish-Image-Classification.git
cd Multiclass-Fish-Image-Classification
```

### 4.2 Install Requirements

```
pip install -r requirements.txt
```

### 4.3 Train Models

Train the CNN model:

```
python train_cnn.py
```

Train Transfer Learning models:

```
python train_transfer_learning.py
```

### 4.4 Evaluate Models

```
python evaluate_models.py
```

---

Do you want me to also **add the `.gitattributes` content for Git LFS tracking** of your `.h5` model files so this README matches your repo setup perfectly? That way, anyone cloning will get models without needing separate download links.

