# Animal Body Condition Score(BCS) Classification using AP-10K Dataset

This project implements an animal weight classification system using deep learning and traditional machine learning approaches. The system classifies animals into three weight categories: **Underweight**, **Healthy**, and **Overweight** based on their body proportions(Body length) extracted from the AP-10K animal pose dataset.

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation &amp; Setup](#installation--setup)
- [Dataset Download](#dataset-download)
- [Project Structure](#project-structure)
- [Usage Instructions](#usage-instructions)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

## 🎯 Project Overview

This project uses two complementary approaches for animal weight classification:

1. **Traditional ML Approach**: Histogram of Oriented Gradients (HOG) features + Random Forest
2. **Deep Learning Approach**: Custom CNN

The classification is based on body length measurements calculated from animal keypoint annotations, creating proxy labels for weight categories.

## ✨ Features

- **Multi-approach Classification**: Both traditional ML and deep learning methods
- **Robust Data Processing**: Handles missing data and various image formats
- **Comprehensive Evaluation**: Detailed metrics, confusion matrices, and visualizations
- **Transfer Learning**: Uses custom CNN for the model training
- **HOG Feature Extraction**: Traditional computer vision features for comparison
- **Data Augmentation**: Improves model generalization
- **Model Persistence**: Save and load trained models

## 📋 Prerequisites

- **Python 3.8+**
- **Anaconda or Miniconda**
- **Jupyter Notebook**
- **8GB+ RAM recommended**
- **GPU support optional but recommended for faster training**

## 🚀 Installation & Setup

### 1. Create Conda Environment

```bash
conda create -n animal_classification python=3.9
conda activate animal_classification
```

### 2. Install Jupyter Notebook

```bash
conda install jupyter notebook
```

### 3. Clone/Download Project

Download this project and navigate to the project directory.

```
git clone https://github.com/Dkplucas/model.git
```

### 4. Install Dependencies

```bash
pip install -r requirements_ml.txt
```

### 5. Launch Jupyter Notebook

```bash
jupyter notebook
```

## 📥 Dataset Download

### Download AP-10K Dataset

1. **Visit the official repository**: [AP-10K Dataset](https://github.com/AlexTheBad/AP-10K?tab=readme-ov-file)
2. **Download the dataset** following the instructions in the repository
3. **Extract the dataset** to your Jupyter notebook home directory with the following structure:

```
data/
└── ap-10K/
    ├── annotations/
    │   ├── ap10k-train-split1.json
    │   ├── ap10k-train-split2.json
    │   ├── ap10k-train-split3.json
    │   ├── ap10k-val-split1.json
    │   ├── ap10k-val-split2.json
    │   ├── ap10k-val-split3.json
    │   ├── ap10k-test-split1.json
    │   ├── ap10k-test-split2.json
    │   └── ap10k-test-split3.json
    └── data/
        ├── 000000000001.jpg
        ├── 000000000002.jpg
        └── ... (all image files)
```

**Important**: Ensure the dataset is placed in the `data/ap-10K/` directory relative to your Jupyter notebook working directory.

## 📁 Project Structure

```
animal_classification/
├── dataprocess.py          # Data preprocessing and label creation
├── hogfeatures.py          # HOG feature extraction
├── model.py                # Deep learning model training
├── requirements_ml.txt     # Project dependencies
├── README.md              # This file
└── data/                  # Dataset directory (to be created)
    └── ap-10K/
        ├── annotations/   # JSON annotation files
        └── data/         # Image files
```

## 🔄 Usage Instructions

**⚠️ Important**: Run the scripts in the following order for successful execution:

### Step 1: Data Preprocessing

Open and run `dataprocess.py` in Jupyter Notebook:

```python
%run dataprocess.py
```

**What this does:**

- Loads AP-10K annotations from JSON files
- Calculates body length from keypoint coordinates (nose to tail distance)
- Creates weight classification labels (Underweight/Healthy/Overweight)
- Splits data into train/validation/test sets
- Saves processed data as CSV files

**Expected outputs:**

- `data/train_split.csv`
- `data/val_split.csv`
- `data/test_split.csv`

### Step 2: HOG Feature Extraction

Run `hogfeatures.py` in Jupyter Notebook:

```python
%run hogfeatures.py
```

**What this does:**

- Extracts Histogram of Oriented Gradients (HOG) features from images
- Processes images in batches for memory efficiency
- Handles various image path structures in the dataset
- Prepares traditional ML features for classification

**Expected outputs:**

- HOG feature arrays for train/validation/test sets
- Progress logs showing successful feature extractions

### Step 3: Model Training

Run `model.py` in Jupyter Notebook:

```python
%run model.py
```

**What this does:**

- Builds and trains custom CNN model
- Implements data augmentation and callbacks
- Evaluates model performance on test set
- Generates confusion matrices and classification reports
- Saves trained model and training history

**Expected outputs:**

- `best_weights.h5` - Best model weights
- `animal_weight_classifier_weights.h5` - Final model weights
- `animal_weight_classifier_architecture.json` - Model architecture
- `training_log.csv` - Training history
- `confusion_matrix.png` - Confusion matrix visualization
- `training_history.png` - Training/validation curves

## 🏗️ Model Architecture

### Deep Learning Model (Custon CNN)

- **Base Model**: Custom CNN
- **Input**: 224×224×3 (Width × Height × RGB Channels)
- **Architecture**:
  - Conv2D ((222, 222, 32), activation='relu')
  - MaxPooling2D (111, 111, 32)
  - Conv2D ((109, 109, 64), activation='relu')
  - MaxPooling2D ()(2, 2)
  - Conv2D (64, (3, 3), activation='relu')
  - GobalAveragePooling2D ()
  - Dropout (0.5)
  - Dense(64, activation='relu')
  - Dropout(0.3)
  - Dense(3, activation='softmax')
- **Optimizer**:Adam(learning_rate=1e-4)
- **Loss**: Sparse Categorical Crossentropy

### Traditional ML Approach

- **Feature Extraction**: HOG (Histogram of Oriented Gradients)
- **Parameters**:
  - 9 orientations
  - 16×16 pixels per cell
  - 2×2 cells per block
  - L2-Hys normalization
- **Classifier**: Random Forest (can be extended)

## 📊 Results

The model provides:

- **Classification metrics**: Precision, Recall, F1-score for each class
- **Confusion matrix**: Visual representation of classification performance
- **Training curves**: Loss and accuracy over training epochs
- **Class distribution analysis**: Understanding of dataset balance

## 🔧 Troubleshooting

### Common Issues:

1. **"Image not found" errors**:

   - Ensure dataset is extracted to correct `data/ap-10K/` directory
   - Check that both annotation files and images are present
2. **Memory errors during training**:

   - Reduce batch size in `model.py` (default is 8-16)
   - Close other applications to free up RAM
3. **Slow training**:

   - Consider using GPU acceleration with tensorflow-gpu
   - Reduce image resolution if necessary
4. **Import errors**:

   - Ensure all dependencies are installed: `pip install -r requirements_ml.txt`
   - Activate the correct conda environment
5. **TensorFlow/Keras errors**:

   - Update TensorFlow: `pip install tensorflow --upgrade`
   - Check CUDA compatibility for GPU usage

### Dependencies Issues:

```bash
# If sklearn package fails, install scikit-learn instead:
pip install scikit-learn

# If skimage package fails, install scikit-image instead:
pip install scikit-image
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is for educational purposes. Please refer to the AP-10K dataset license for data usage terms.

## 🙏 Acknowledgments

- [AP-10K Dataset](https://github.com/AlexTheBad/AP-10K) for providing the animal pose dataset
- TensorFlow and scikit-learn communities for the frameworks

---

**Note**: This project is designed for research and educational purposes. For production use, consider additional validation and testing on diverse datasets.
