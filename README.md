# Emotion Classifier

This project is a Convolutional Neural Network (CNN) based classifier to recognize human emotions from grayscale facial images. It uses the [FER 2013 dataset](https://www.kaggle.com/datasets/msambare/fer2013) from Kaggle, which contains 48x48 pixel grayscale images categorized into seven emotions: **angry**, **disgust**, **fear**, **happy**, **neutral**, **sad**, and **surprise**.

---

## Table of Contents

1. [Directory Structure](#directory-structure)
2. [Dataset](#dataset)
3. [Model Architecture](#model-architecture)
4. [Installation](#installation)
5.  [Results](#results)

---

## Directory Structure

The project is organized as follows:

```
emotion-classifier/
│
├── manual_test/                 # Test images for manual inference
│   ├── test1.jpg
│   └── test2.jpg
│
├── model/                       # Trained model files
│   ├── emotion_model.h5         # Saved Keras model
│
├── README.md                    # Documentation
├── Emotion-Recognition.ipynb      source code
└── 
```

```
You will need to download and set up the FER 2013 dataset to train and validate the model.
```
## Dataset
```
The FER 2013 dataset is available for download from Kaggle:

FER 2013 Dataset on Kaggle
Steps to Download the Dataset
Visit the Kaggle dataset page: FER 2013 Dataset.
Download the dataset file (fer2013.csv).
Extract and organize it into the following structure under the images/ directory:
emotion-classifier/
├── images/
    ├── train/                   # Training images
    │   ├── angry/
    │   ├── disgust/
    │   ├── fear/
    │   ├── happy/
    │   ├── neutral/
    │   ├── sad/
    │   └── surprise/
    ├── validation/              # Validation images
    │   ├── angry/
    │   ├── disgust/
    │   ├── fear/
    │   ├── happy/
    │   ├── neutral/
    │   ├── sad/
    │   └── surprise/
```

## Model Architecture
```
The model uses the following architecture:

Conv2D: 32 filters, kernel size 3x3, ReLU activation
MaxPooling2D: Pool size 2x2
Conv2D: 64 filters, kernel size 3x3, ReLU activation
MaxPooling2D: Pool size 2x2
Conv2D: 128 filters, kernel size 3x3, ReLU activation
MaxPooling2D: Pool size 2x2
Conv2D: 64 filters, kernel size 3x3, ReLU activation
Flatten: Converts 2D matrix to 1D vector
Dense: 64 neurons, ReLU activation
Dense: 7 neurons, softmax activation
Model Summary
Total trainable parameters: 183,367
```
## Installation
```
Prerequisites
Python 3.6+
TensorFlow/Keras
OpenCV
NumPy
Matplotlib
scikit-learn
```

## Results
```
Accuracy
Training Accuracy: 66.34%
Validation Accuracy: 55.84%
Classification Report
Emotion	Precision	Recall	F1-Score
Angry	0.51	0.38	0.43
Disgust	0.54	0.29	0.38
Fear	0.47	0.28	0.35
Happy	0.72	0.81	0.76
Neutral	0.55	0.43	0.48
Sad	0.38	0.62	0.47
Surprise	0.68	0.72	0.70



```
![image](https://github.com/user-attachments/assets/33fe1f5c-d05a-4302-ae6f-ed0f8a54fa6c)

License
This project is licensed under the MIT License. See the LICENSE file for details.
