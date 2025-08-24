"""This file was created by Lucasdkp"""

# Model training 
# Please make sure to install scikit-learn and tensorflow or any other package in the requirements_ml.txt
# Training a model on your computer may be resource-intensive and time-consuming. Use Anaconda(Jupyter notebook) or a similar tool to manage your environment.

# import necessary packages
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, applications
from tensorflow.keras.models import Model, model_from_json
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
import seaborn as sns
import pandas as pd
import json
import pickle
from sklearn.model_selection import train_test_split
import io
import sys
import time

# Load dataset function (added to fix missing data issue)
def load_dataset_from_csv():
    """Load dataset from CSV files created by dataprocess.py"""
    try:
        # Try to load the preprocessed data
        df_train = pd.read_csv('data/train_split.csv')
        df_val = pd.read_csv('data/val_split.csv')
        df_test = pd.read_csv('data/test_split.csv')
        
        X_train = df_train['file_name'].tolist()
        y_train = df_train['weight_class'].tolist()
        X_val = df_val['file_name'].tolist()
        y_val = df_val['weight_class'].tolist()
        X_test = df_test['file_name'].tolist()
        y_test = df_test['weight_class'].tolist()
        
        print(f"Loaded dataset from CSV files: Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        return X_train, y_train, X_val, y_val, X_test, y_test
        
    except FileNotFoundError:
        print("CSV files not found. Running dataprocess.py to create dataset...")
        # Import and run dataprocess.py
        import subprocess
        import sys
        
        try:
            # Run dataprocess.py
            result = subprocess.run([sys.executable, "dataprocess.py"], capture_output=True, text=True)
            if result.returncode == 0:
                print("dataprocess.py executed successfully")
                # Try to load the CSV files again
                return load_dataset_from_csv()
            else:
                print(f"Error running dataprocess.py: {result.stderr}")
                # Create a dummy dataset as fallback
                return create_dummy_dataset()
        except:
            print("Failed to run dataprocess.py. Creating dummy dataset...")
            return create_dummy_dataset()

def create_dummy_dataset():
    """Create a small dummy dataset for testing when no real data is found"""
    print("Creating dummy dataset for testing...")
    
    # Create some dummy file paths and labels
    dummy_files = [
        'dummy_image_1.jpg', 'dummy_image_2.jpg', 'dummy_image_3.jpg',
        'dummy_image_4.jpg', 'dummy_image_5.jpg', 'dummy_image_6.jpg',
        'dummy_image_7.jpg', 'dummy_image_8.jpg', 'dummy_image_9.jpg',
        'dummy_image_10.jpg', 'dummy_image_11.jpg', 'dummy_image_12.jpg',
        'dummy_image_13.jpg', 'dummy_image_14.jpg', 'dummy_image_15.jpg',
        'dummy_image_16.jpg', 'dummy_image_17.jpg', 'dummy_image_18.jpg',
        'dummy_image_19.jpg', 'dummy_image_20.jpg', 'dummy_image_21.jpg',
        'dummy_image_22.jpg', 'dummy_image_23.jpg', 'dummy_image_24.jpg'
    ]
    
    dummy_labels = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2]
    
    # Split into train, validation, and test sets
    X_temp, X_test, y_temp, y_test = train_test_split(
        dummy_files, dummy_labels, test_size=0.2, random_state=42, stratify=dummy_labels
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp  # 0.25 * 0.8 = 0.2
    )
    
    print(f"Dummy dataset created: Train: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test

# find_image_path function
def find_image_path(image_filename, data_base_dir='data/ap-10K/'):
    """Search for an image file recursively in the dataset directory."""
    for root, dirs, files in os.walk(data_base_dir):
        if image_filename in files:
            return os.path.join(root, image_filename)
    return None

# Load and preprocess image for deep learning model - ENSURES 3 CHANNELS
def load_and_preprocess_image_for_dl(file_path, target_size=(224, 224), data_base_dir='data/ap-10K/'):
    """Load and preprocess image for deep learning model - ENSURES 3 CHANNELS"""
    try:
        # For dummy images, create a random image
        if file_path.startswith('dummy_image_'):
            # Create a random image for testing
            image = np.random.rand(*target_size, 3).astype(np.float32)
            image = applications.efficientnet.preprocess_input(image)
            return image
        
        # Find the image file
        image_filename = os.path.basename(file_path)
        full_path = find_image_path(image_filename, data_base_dir)
        
        if full_path is None:
            # If not found, try using the provided path directly
            full_path = file_path
            if not os.path.exists(full_path):
                print(f"Warning: Image file not found: {file_path}")
                return None
        
        # Load image using TensorFlow
        image = tf.io.read_file(full_path)
        image = tf.image.decode_jpeg(image, channels=3)  # FORCE 3 CHANNELS
        
        # Convert to float32 and resize
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, target_size)
        
        # Apply EfficientNet preprocessing (normalization)
        image = applications.efficientnet.preprocess_input(image)
        
        return image.numpy()
        
    except Exception as e:
        print(f"Error loading image {file_path}: {e}")
        return None

# Load and preprocess image for Random Forest (flattened features)
def load_and_preprocess_image_for_rf(file_path, target_size=(64, 64), data_base_dir='data/ap-10K/'):
    """Load and preprocess image for Random Forest classifier"""
    try:
        # For dummy images, create a random image
        if file_path.startswith('dummy_image_'):
            # Create a random image for testing and flatten it
            image = np.random.rand(*target_size, 3).astype(np.float32)
            return image.flatten()
        
        # Find the image file
        image_filename = os.path.basename(file_path)
        full_path = find_image_path(image_filename, data_base_dir)
        
        if full_path is None:
            # If not found, try using the provided path directly
            full_path = file_path
            if not os.path.exists(full_path):
                print(f"Warning: Image file not found: {file_path}")
                return None
        
        # Load image using TensorFlow
        image = tf.io.read_file(full_path)
        image = tf.image.decode_jpeg(image, channels=3)  # FORCE 3 CHANNELS
        
        # Convert to float32 and resize to smaller size for RF
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, target_size)
        
        # Flatten the image for Random Forest
        return image.numpy().flatten()
        
    except Exception as e:
        print(f"Error loading image {file_path} for RF: {e}")
        return None

# Build a simpler model from scratch to avoid serialization issues
def build_simple_model(input_shape=(224, 224, 3), num_classes=3):
    """Build a simpler CNN model that won't have serialization issues"""
    # Fix the warning by using Input layer instead of input_shape parameter
    model = keras.Sequential([
        layers.Input(shape=input_shape),  # Add Input layer to fix the warning
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.5),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

# Train Random Forest classifier
def train_random_forest(X_train, y_train, X_test, y_test):
    """Train and evaluate a Random Forest classifier"""
    print("\n" + "="*60)
    print("TRAINING RANDOM FOREST CLASSIFIER")
    print("="*60)
    
    start_time = time.time()
    
    # Preload images for Random Forest (smaller size for efficiency)
    print("Preloading images for Random Forest...")
    X_train_rf = []
    y_train_rf = []
    failed_count = 0
    
    for i, (file_path, label) in enumerate(zip(X_train, y_train)):
        if i % 50 == 0:
            print(f"Loading RF image {i}/{len(X_train)}")
        features = load_and_preprocess_image_for_rf(file_path, target_size=(64, 64))
        if features is not None:
            X_train_rf.append(features)
            y_train_rf.append(label)
        else:
            failed_count += 1
    
    X_test_rf = []
    y_test_rf = []
    test_failed_count = 0
    
    for i, (file_path, label) in enumerate(zip(X_test, y_test)):
        features = load_and_preprocess_image_for_rf(file_path, target_size=(64, 64))
        if features is not None:
            X_test_rf.append(features)
            y_test_rf.append(label)
        else:
            test_failed_count += 1
    
    X_train_rf = np.array(X_train_rf)
    y_train_rf = np.array(y_train_rf)
    X_test_rf = np.array(X_test_rf)
    y_test_rf = np.array(y_test_rf)
    
    print(f"RF Training samples: {len(X_train_rf)} (failed: {failed_count})")
    print(f"RF Test samples: {len(X_test_rf)} (failed: {test_failed_count})")
    
    if len(X_train_rf) == 0 or len(X_test_rf) == 0:
        print("Not enough data for Random Forest training")
        return None
    
    # Create and train Random Forest classifier
    rf_classifier = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        random_state=42,
        n_jobs=-1  # Use all available cores
    )
    
    print("Training Random Forest...")
    rf_classifier.fit(X_train_rf, y_train_rf)
    
    # Make predictions
    print("Making predictions with Random Forest...")
    y_pred_rf = rf_classifier.predict(X_test_rf)
    y_pred_proba_rf = rf_classifier.predict_proba(X_test_rf)
    
    # Calculate accuracy
    accuracy = np.mean(y_pred_rf == y_test_rf)
    
    # Generate classification report
    print("\nRandom Forest Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    report = classification_report(
        y_test_rf, 
        y_pred_rf, 
        target_names=['Underweight', 'Healthy', 'Overweight'],
        zero_division=0
    )
    print(report)
    
    # Confusion matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test_rf, y_pred_rf)
    
    class_names = ['Underweight', 'Healthy', 'Overweight']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Random Forest - Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('random_forest_confusion_matrix.png')
    plt.show()
    
    end_time = time.time()
    training_time = end_time - start_time
    print(f"Random Forest training and evaluation time: {training_time:.2f} seconds")
    
    return {
        'classifier': rf_classifier,
        'accuracy': accuracy,
        'predictions': y_pred_rf,
        'probabilities': y_pred_proba_rf,
        'training_time': training_time,
        'train_samples': len(X_train_rf),
        'test_samples': len(X_test_rf)
    }

# Create the simpler model
model = build_simple_model(input_shape=(224, 224, 3), num_classes=3)

# Use a lower learning rate
optimizer = keras.optimizers.Adam(learning_rate=1e-4)

model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Callbacks for training
def create_callbacks():
    return [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            'best_weights.weights.h5',  # Fixed file extension
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=True,
            verbose=1
        ),
        keras.callbacks.CSVLogger('training_log.csv')
    ]

# Training function with improved data handling
def train_model(X_train, y_train, X_val, y_val, epochs=50, batch_size=16):
    """Train the model with the given data"""
    # Preload all data first
    print("Preloading training images...")
    X_train_loaded = []
    y_train_loaded = []
    train_failed_count = 0
    
    for i, (file_path, label) in enumerate(zip(X_train, y_train)):
        if i % 50 == 0:
            print(f"Loading training image {i}/{len(X_train)}")
        image = load_and_preprocess_image_for_dl(file_path)
        if image is not None:
            X_train_loaded.append(image)
            y_train_loaded.append(label)
        else:
            train_failed_count += 1
    
    X_train_loaded = np.array(X_train_loaded)
    y_train_loaded = np.array(y_train_loaded)
    
    print("Preloading validation images...")
    X_val_loaded = []
    y_val_loaded = []
    val_failed_count = 0
    
    for i, (file_path, label) in enumerate(zip(X_val, y_val)):
        if i % 50 == 0:
            print(f"Loading validation image {i}/{len(X_val)}")
        image = load_and_preprocess_image_for_dl(file_path)
        if image is not None:
            X_val_loaded.append(image)
            y_val_loaded.append(label)
        else:
            val_failed_count += 1
    
    X_val_loaded = np.array(X_val_loaded)
    y_val_loaded = np.array(y_val_loaded)
    
    print(f"Training samples: {len(X_train_loaded)} (failed: {train_failed_count})")
    print(f"Validation samples: {len(X_val_loaded)} (failed: {val_failed_count})")
    
    if len(X_train_loaded) == 0:
        print("No training images loaded successfully!")
        return None, 0
    
    # Create simple dataset instead of using generator for small datasets
    print("Starting training...")
    
    callbacks = create_callbacks()
    
    start_time = time.time()
    
    history = model.fit(
        X_train_loaded, y_train_loaded,
        batch_size=min(batch_size, len(X_train_loaded)),
        epochs=epochs,
        validation_data=(X_val_loaded, y_val_loaded),
        callbacks=callbacks,
        verbose=1,
        shuffle=True
    )
    
    end_time = time.time()
    training_time = end_time - start_time
    print(f"Neural Network training time: {training_time:.2f} seconds")
    
    return history, training_time

# Evaluation function
def evaluate_model(X_test, y_test):
    """Evaluate the model on test data"""
    # Preload all test data for proper evaluation
    X_test_actual = []
    y_test_actual = []
    test_failed_count = 0
    
    for i, (file_path, label) in enumerate(zip(X_test, y_test)):
        if i % 50 == 0:
            print(f"Loading test image {i}/{len(X_test)}")
        image = load_and_preprocess_image_for_dl(file_path)
        if image is not None:
            X_test_actual.append(image)
            y_test_actual.append(label)
        else:
            test_failed_count += 1
    
    X_test_actual = np.array(X_test_actual)
    y_test_actual = np.array(y_test_actual)
    
    print(f"Test samples loaded: {len(X_test_actual)} (failed: {test_failed_count})")
    
    if len(X_test_actual) == 0:
        print("No test images found!")
        return None
    
    print("Evaluating model...")
    results = model.evaluate(X_test_actual, y_test_actual, verbose=1)
    
    print(f"\nTest Results:")
    print(f"Loss: {results[0]:.4f}")
    print(f"Accuracy: {results[1]:.4f}")
    
    return results

# Prediction and analysis function
def predict_and_analyze(X_test, y_test, model_name="Neural Network"):
    """Make predictions and provide detailed analysis"""
    # Preload all test data
    X_test_actual = []
    y_test_actual = []
    failed_count = 0
    
    for i, (file_path, label) in enumerate(zip(X_test, y_test)):
        if i % 50 == 0:
            print(f"Loading prediction image {i}/{len(X_test)}")
        image = load_and_preprocess_image_for_dl(file_path)
        if image is not None:
            X_test_actual.append(image)
            y_test_actual.append(label)
        else:
            failed_count += 1
    
    X_test_actual = np.array(X_test_actual)
    y_test_actual = np.array(y_test_actual)
    
    print(f"Prediction samples loaded: {len(X_test_actual)} (failed: {failed_count})")
    
    if len(X_test_actual) == 0:
        print("No test images found!")
        return None, None
    
    # Get predictions
    y_pred_proba = model.predict(X_test_actual, verbose=1)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Check if any classes are missing in predictions
    unique_preds = np.unique(y_pred)
    unique_actual = np.unique(y_test_actual)
    
    print(f"Actual classes present: {unique_actual}")
    print(f"Predicted classes: {unique_preds}")
    
    # Classification report with zero_division parameter to avoid warnings
    print(f"\n{model_name} - Classification Report:")
    report = classification_report(
        y_test_actual, 
        y_pred, 
        target_names=['Underweight', 'Healthy', 'Overweight'],
        zero_division=0
    )
    print(report)
    
    # Confusion matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test_actual, y_pred)
    
    class_names = ['Underweight', 'Healthy', 'Overweight']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title(f'{model_name} - Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'{model_name.lower().replace(" ", "_")}_confusion_matrix.png')
    plt.show()
    
    return y_pred, y_pred_proba

# Plot training history
def plot_training_history(history):
    """Plot training history"""
    if history is None:
        print("No training history to plot")
        return
        
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

# Compare model performance
def compare_models(nn_accuracy, rf_results, nn_time):
    """Compare the performance of Neural Network and Random Forest"""
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    
    if rf_results is not None and nn_accuracy is not None:
        rf_accuracy = rf_results['accuracy']
        rf_time = rf_results['training_time']
        
        print(f"Neural Network Accuracy: {nn_accuracy:.4f}")
        print(f"Random Forest Accuracy: {rf_accuracy:.4f}")
        print(f"\nNeural Network Training Time: {nn_time:.2f} seconds")
        print(f"Random Forest Training Time: {rf_time:.2f} seconds")
        
        # Determine which model performed better
        if nn_accuracy > rf_accuracy:
            print("\nNeural Network performed better!")
            improvement = ((nn_accuracy - rf_accuracy) / rf_accuracy) * 100
            print(f"Improvement: {improvement:.2f}%")
        elif rf_accuracy > nn_accuracy:
            print("\nRandom Forest performed better!")
            improvement = ((rf_accuracy - nn_accuracy) / nn_accuracy) * 100
            print(f"Improvement: {improvement:.2f}%")
        else:
            print("\nBoth models performed equally!")
    else:
        if nn_accuracy is not None:
            print(f"Neural Network Accuracy: {nn_accuracy:.4f}")
        if rf_results is not None:
            print(f"Random Forest Accuracy: {rf_results['accuracy']:.4f}")
        print("Some results not available for comparison")

# Main execution
if __name__ == "__main__":
    # Load the dataset from CSV files created by dataprocess.py
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset_from_csv()
    
    # First, check your dataset sizes
    print(f"\nDataset sizes from CSV:")
    print(f"Train: {len(X_train)}")
    print(f"Validation: {len(X_val)}")
    print(f"Test: {len(X_test)}")
    
    # Check class distribution
    print("\nClass distribution in training set:")
    train_counts = pd.Series(y_train).value_counts().sort_index()
    print(train_counts)
    
    # Train Random Forest first (usually faster)
    rf_results = train_random_forest(X_train, y_train, X_test, y_test)
    
    # Train the neural network with a smaller batch size for small datasets
    batch_size = min(8, len(X_train) // 2)  # Ensure we have at least 2 batches
    if batch_size < 2:
        batch_size = 2
    
    print("\n" + "="*60)
    print("TRAINING NEURAL NETWORK")
    print("="*60)
    
    print("Starting neural network training...")
    history, nn_time = train_model(X_train, y_train, X_val, y_val, epochs=50, batch_size=batch_size)
    
    if history is not None:
        # Plot training history
        plot_training_history(history)
        
        # Evaluate on test set
        test_results = evaluate_model(X_test, y_test)
        
        # Detailed analysis for neural network
        y_pred, y_pred_proba = predict_and_analyze(X_test, y_test, "Neural Network")
        
        # Compare model performance
        if test_results is not None:
            compare_models(test_results[1], rf_results, nn_time)
        
        # Save combined results
        combined_results = {
            'neural_network': {
                'accuracy': test_results[1] if test_results else None,
                'training_time': nn_time
            },
            'random_forest': rf_results if rf_results else None,
            'dataset_info': {
                'train_original': len(X_train),
                'val_original': len(X_val),
                'test_original': len(X_test)
            }
        }
        
        with open('model_comparison_results.pkl', 'wb') as f:
            pickle.dump(combined_results, f)
        
        print("\nModel comparison results saved to 'model_comparison_results.pkl'")
    else:
        print("Neural Network training failed!")