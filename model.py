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
import seaborn as sns
import pandas as pd
import json
import pickle
from sklearn.model_selection import train_test_split
import io
import sys

# Load dataset function (added to fix missing data issue)
def load_dataset_from_csv():
    """Load dataset from CSV files created by dataprocess.py"""
    try:
        # Try to load the preprocessed data
        df_train = pd.read_csv('data/train_split.csv')
        df_val = pd.read_csv('data/val_split.csv')
        df_test = pd.read_csv('data/test_split.csv')
        
        X_train = df_train['file_name']
        y_train = df_train['weight_class']
        X_val = df_val['file_name']
        y_val = df_val['weight_class']
        X_test = df_test['file_name']
        y_test = df_test['weight_class']
        
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

# Create a data generator
def create_data_generator(X, y, batch_size=32, shuffle=False):
    """Create a data generator that handles small datasets properly"""
    # Convert to lists if they are pandas Series
    if hasattr(X, 'tolist'):
        X = X.tolist()
    if hasattr(y, 'tolist'):
        y = y.tolist()
    
    # Preload all images to avoid generator issues
    images = []
    labels = []
    
    print("Preloading images...")
    for i, (file_path, label) in enumerate(zip(X, y)):
        if i % 50 == 0:
            print(f"Loading image {i}/{len(X)}")
        image = load_and_preprocess_image_for_dl(file_path)
        if image is not None:
            images.append(image)
            labels.append(label)
    
    images = np.array(images)
    labels = np.array(labels)
    
    print(f"Successfully loaded {len(images)}/{len(X)} images")
    
    # Create data generator
    if shuffle:
        datagen = keras.preprocessing.image.ImageDataGenerator(
            preprocessing_function=applications.efficientnet.preprocess_input,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            fill_mode='nearest'
        )
    else:
        datagen = keras.preprocessing.image.ImageDataGenerator(
            preprocessing_function=applications.efficientnet.preprocess_input
        )
    
    generator = datagen.flow(
        images, labels,
        batch_size=min(batch_size, len(images)),
        shuffle=shuffle
    )
    
    return generator, len(images)

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

# Custom data generator class to fix the warning
class CustomDataGenerator(keras.utils.Sequence):
    def __init__(self, X, y, batch_size=32, shuffle=True, augment=False):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.indexes = np.arange(len(self.X))
        self.on_epoch_end()
        
        # Initialize the base class properly to fix the warning
        super().__init__()
    
    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch_size))
    
    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        X_batch = self.X[batch_indexes]
        y_batch = self.y[batch_indexes]
        
        return X_batch, y_batch
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

# Training function with improved data handling
def train_model(X_train, y_train, X_val, y_val, epochs=50, batch_size=16):
    """Train the model with the given data"""
    # Preload all data first
    print("Preloading training images...")
    X_train_loaded = []
    y_train_loaded = []
    for i, (file_path, label) in enumerate(zip(X_train, y_train)):
        image = load_and_preprocess_image_for_dl(file_path)
        if image is not None:
            X_train_loaded.append(image)
            y_train_loaded.append(label)
    
    X_train_loaded = np.array(X_train_loaded)
    y_train_loaded = np.array(y_train_loaded)
    
    print("Preloading validation images...")
    X_val_loaded = []
    y_val_loaded = []
    for i, (file_path, label) in enumerate(zip(X_val, y_val)):
        image = load_and_preprocess_image_for_dl(file_path)
        if image is not None:
            X_val_loaded.append(image)
            y_val_loaded.append(label)
    
    X_val_loaded = np.array(X_val_loaded)
    y_val_loaded = np.array(y_val_loaded)
    
    print(f"Training samples: {len(X_train_loaded)}")
    print(f"Validation samples: {len(X_val_loaded)}")
    
    # Use the custom data generator
    train_gen = CustomDataGenerator(X_train_loaded, y_train_loaded, batch_size=batch_size, shuffle=True)
    val_gen = CustomDataGenerator(X_val_loaded, y_val_loaded, batch_size=batch_size, shuffle=False)
    
    callbacks = create_callbacks()
    
    print("Starting training...")
    
    # Calculate steps
    train_steps = len(train_gen)
    val_steps = len(val_gen)
    
    print(f"Train steps per epoch: {train_steps}")
    print(f"Validation steps: {val_steps}")
    
    history = model.fit(
        train_gen,
        steps_per_epoch=train_steps,
        epochs=epochs,
        validation_data=val_gen,
        validation_steps=val_steps,
        callbacks=callbacks,
        verbose=1
    )
    
    return history

# Evaluation function
def evaluate_model(X_test, y_test):
    """Evaluate the model on test data"""
    # Preload all test data for proper evaluation
    X_test_actual = []
    y_test_actual = []
    
    for i, (file_path, label) in enumerate(zip(X_test, y_test)):
        image = load_and_preprocess_image_for_dl(file_path)
        if image is not None:
            X_test_actual.append(image)
            y_test_actual.append(label)
    
    X_test_actual = np.array(X_test_actual)
    y_test_actual = np.array(y_test_actual)
    
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
def predict_and_analyze(X_test, y_test):
    """Make predictions and provide detailed analysis"""
    # Preload all test data
    X_test_actual = []
    y_test_actual = []
    
    for i, (file_path, label) in enumerate(zip(X_test, y_test)):
        image = load_and_preprocess_image_for_dl(file_path)
        if image is not None:
            X_test_actual.append(image)
            y_test_actual.append(label)
    
    X_test_actual = np.array(X_test_actual)
    y_test_actual = np.array(y_test_actual)
    
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
    print("\nClassification Report:")
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
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.show()
    
    return y_pred, y_pred_proba

# Plot training history
def plot_training_history(history):
    """Plot training history"""
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

# Save model components
def save_model_completely(model, base_filename='animal_weight_classifier'):
    """Save model architecture, weights, and training history separately"""
    try:
        # 1. Save model architecture as JSON
        model_json = model.to_json()
        with open(f'{base_filename}_architecture.json', 'w', encoding='utf-8') as json_file:
            json_file.write(model_json)
        print(f"Model architecture saved as {base_filename}_architecture.json")
        
        # 2. Save model weights
        model.save_weights(f'{base_filename}_weights.weights.h5')
        print(f"Model weights saved as {base_filename}_weights.weights.h5")
        
        # 3. Save model summary using a different approach
        # Capture the model summary output
        old_stdout = sys.stdout
        new_stdout = io.StringIO()
        sys.stdout = new_stdout
        
        model.summary()
        
        summary_str = new_stdout.getvalue()
        sys.stdout = old_stdout
        
        # Clean the summary string to remove any problematic characters
        clean_summary = summary_str.encode('ascii', 'ignore').decode('ascii')
        
        with open(f'{base_filename}_summary.txt', 'w', encoding='utf-8') as f:
            f.write(clean_summary)
        
        print(f"Model summary saved as {base_filename}_summary.txt")
        
        return True
        
    except Exception as e:
        print(f"Error saving model components: {e}")
        return False

# Load model function (added for completeness)
def load_model_from_files(architecture_file, weights_file):
    """Load a model from architecture and weights files"""
    with open(architecture_file, 'r', encoding='utf-8') as json_file:
        model_json = json_file.read()
    
    model = model_from_json(model_json)
    model.load_weights(weights_file)
    
    # Recompile the model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Main execution
if __name__ == "__main__":
    # Load the dataset from CSV files created by dataprocess.py
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset_from_csv()
    
    # First, check your dataset sizes
    print(f"Dataset sizes:")
    print(f"Train: {len(X_train)}")
    print(f"Validation: {len(X_val)}")
    print(f"Test: {len(X_test)}")
    
    # Check class distribution
    print("\nClass distribution in training set:")
    if hasattr(y_train, 'value_counts'):
        print(y_train.value_counts().sort_index())
    else:
        print(pd.Series(y_train).value_counts().sort_index())
    
    # Train the model with a smaller batch size for small datasets
    batch_size = min(8, len(X_train) // 2)  # Ensure we have at least 2 batches
    if batch_size < 2:
        batch_size = 2
    
    print("Starting training...")
    history = train_model(X_train, y_train, X_val, y_val, epochs=50, batch_size=batch_size)
    
    # Plot training history
    plot_training_history(history)
    
    # Evaluate on test set
    test_results = evaluate_model(X_test, y_test)
    
    # Detailed analysis
    y_pred, y_pred_proba = predict_and_analyze(X_test, y_test)
    
    # Save the model components separately
    save_success = save_model_completely(model)
    
    if save_success:
        print("Model components saved successfully!")
        print("To reload the model later:")
        print("1. Load architecture: model = keras.models.model_from_json(json_string)")
        print("2. Load weights: model.load_weights('animal_weight_classifier_weights.weights.h5')")
        print("3. Compile: model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])")
    else:
        print("Failed to save model components. Using fallback...")
        # Fallback: Save only predictions and results for report
        results = {
            'test_accuracy': test_results[1] if test_results else None,
            'predictions': y_pred.tolist() if y_pred is not None else [],
            'true_labels': y_test.tolist() if hasattr(y_test, 'tolist') else list(y_test)
        }
        with open('training_results.pkl', 'wb') as f:
            pickle.dump(results, f)
        print("Results saved to training_results.pkl")