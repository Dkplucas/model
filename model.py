"""This file was created by Lucasdkp"""

# Model training 
# Please make sure to install scikit-learn and tensorflow or any other package in the requirements_ml.txt
# Training a model on your computer may be resource-intensive and time-consuming. Use Anaconda(Jupyter notebook) or a similar tool to manage your environment.

# import necessary packages
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, applications
from tensorflow.keras.models import Model
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import pandas as pd
import json
import pickle

def find_image_path(image_filename, data_base_dir='data/ap-10K/'):
    """Search for an image file recursively in the dataset directory."""
    for root, dirs, files in os.walk(data_base_dir):
        if image_filename in files:
            return os.path.join(root, image_filename)
    return None

def load_and_preprocess_image_for_dl(file_path, target_size=(224, 224), data_base_dir='data/ap-10K/'):
    """Load and preprocess image for deep learning model - ENSURES 3 CHANNELS"""
    try:
        # Find the image file
        image_filename = os.path.basename(file_path)
        full_path = find_image_path(image_filename, data_base_dir)
        
        if full_path is None:
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
        return None

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
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
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
            'best_weights.h5',
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=True,
            verbose=1
        ),
        keras.callbacks.CSVLogger('training_log.csv')
    ]

def train_model(X_train, y_train, X_val, y_val, epochs=50, batch_size=16):
    """Train the model with the given data"""
    train_gen, train_samples = create_data_generator(X_train, y_train, batch_size, shuffle=True)
    val_gen, val_samples = create_data_generator(X_val, y_val, batch_size, shuffle=False)
    
    callbacks = create_callbacks()
    
    print("Starting training...")
    print(f"Training samples: {train_samples}")
    print(f"Validation samples: {val_samples}")
    
    # Calculate steps
    train_steps = max(1, train_samples // batch_size)
    val_steps = max(1, val_samples // batch_size)
    
    if train_samples <= batch_size:
        train_steps = 1
    if val_samples <= batch_size:
        val_steps = 1
    
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

def save_model_completely(model, base_filename='animal_weight_classifier'):
    """Save model architecture, weights, and training history separately"""
    try:
        # 1. Save model architecture as JSON
        model_json = model.to_json()
        with open(f'{base_filename}_architecture.json', 'w') as json_file:
            json_file.write(model_json)
        print(f"Model architecture saved as {base_filename}_architecture.json")
        
        # 2. Save model weights
        model.save_weights(f'{base_filename}_weights.h5')
        print(f"Model weights saved as {base_filename}_weights.h5")
        
        # 3. Save model summary
        with open(f'{base_filename}_summary.txt', 'w') as f:
            model.summary(print_fn=lambda x: f.write(x + '\n'))
        print(f"Model summary saved as {base_filename}_summary.txt")
        
        return True
        
    except Exception as e:
        print(f"Error saving model components: {e}")
        return False

# Main execution
if __name__ == "__main__":
    # Assuming you have X_train, y_train, X_val, y_val, X_test, y_test from preprocessing
    
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
    
    # Train the model
    print("Starting training...")
    history = train_model(X_train, y_train, X_val, y_val, epochs=50, batch_size=8)
    
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
        print("2. Load weights: model.load_weights('animal_weight_classifier_weights.h5')")
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