# Emotion Classifier

This project is a Convolutional Neural Network (CNN) based classifier to recognize human emotions from grayscale facial images. It uses the [FER 2013 dataset](https://www.kaggle.com/datasets/msambare/fer2013) from Kaggle, which contains 48x48 pixel grayscale images categorized into seven emotions: **angry**, **disgust**, **fear**, **happy**, **neutral**, **sad**, and **surprise**.

---

## Table of Contents

1. [Directory Structure](#directory-structure)
2. [Dataset](#dataset)
3. [Model Architecture](#model-architecture)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Results](#results)
7. [Future Work](#future-work)

---

## Directory Structure

The project is organized as follows:

```plaintext
emotion-classifier/
│
├── manual_test/                 # Test images for manual inference
│   ├── test1.jpg
│   └── test2.jpg
│
├── model/                       # Trained model files
│   ├── emotion_model.h5         # Saved Keras model
│
├── output/                      # Outputs generated (e.g., plots)
│   ├── confusion_matrix.png
│   └── training_accuracy_loss.png
│
├── scripts/                     # Helper scripts
│   ├── train_emotion_classifier.py
│   ├── test_emotion_classifier.py
│   └── preprocess_images.py
│
├── README.md                    # Documentation
├── requirements.txt             # Python dependencies
└── LICENSE                      # License file
You will need to download and set up the FER 2013 dataset to train and validate the model.

Dataset
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
You can use the scripts/preprocess_images.py script to preprocess and organize the dataset into training and validation directories.

Model Architecture
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
Installation
Prerequisites
Python 3.6+
TensorFlow/Keras
OpenCV
NumPy
Matplotlib
scikit-learn
Steps
Clone this repository:
git clone https://github.com/your-repo/emotion-classifier.git
cd emotion-classifier
Install dependencies:
pip install -r requirements.txt
Usage
Preprocessing the Dataset
After downloading the dataset, run the following script to preprocess and organize images:

python scripts/preprocess_images.py --input_path fer2013.csv --output_dir ./images/
Training the Model
Ensure the dataset is in the correct structure under the ./images/ directory, then run:

python scripts/train_emotion_classifier.py
Testing the Model
You can test the model on individual images:

from scripts.test_emotion_classifier import getPrediction

getPrediction('./manual_test/test1.jpg')
Inference Example
Input Image:Test Image
Predicted Emotion: happy
Results
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
Confusion Matrix

Future Work
Optimize Hyperparameters: Tune the learning rate, batch size, and number of epochs.
Data Augmentation: Introduce augmentation to balance the dataset and improve generalization.
Improve Model: Experiment with advanced architectures like ResNet or MobileNet.
Real-Time Prediction: Integrate with a webcam for real-time emotion detection.
Contributing
Contributions are welcome! Please submit a pull request or open an issue to discuss your idea.

License
This project is licensed under the MIT License. See the LICENSE file for details.