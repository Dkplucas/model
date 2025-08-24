"""This file was created by Lucasdkp"""

# Extracting the HOG features from the images
# This is a script to extract HOG features from images in the AP-10K dataset
# The HOG features are extracted from the images and saved in a CSV file
# The CSV file is then used to train the SVM model

# Import packages 
from skimage import io
from skimage.feature import hog
from skimage.io import imread
from skimage.transform import resize
from skimage.color import rgb2gray
import numpy as np
import os
import pandas as pd

# Extract HOG features
def extract_hog_features(image_path, target_size=(224, 224), data_base_dir='data/ap-10K/'):
    """
    Read an image, convert to grayscale, resize, and extract HOG features.
    This handles the AP-10K dataset structure.
    """
    try:
        # AP-10K has images in subdirectories based on the first few digits
        # Example: image '000000007348.jpg' is in 'images/00000/000000007348.jpg'
        image_id = os.path.splitext(os.path.basename(image_path))[0]
        subdir = image_id[:5]  # First 5 digits as subdirectory
        subdir2 = image_id[:7]  # Sometimes nested further
        
        # Try multiple possible paths
        possible_paths = [
            os.path.join(data_base_dir, 'images', image_path),  # Flat structure
            os.path.join(data_base_dir, 'images', subdir, image_path),  # Single subdir
            os.path.join(data_base_dir, 'images', subdir, subdir2, image_path),  # Nested subdir
            os.path.join(data_base_dir, 'train_data', 'images', image_path),  # Alternative structure
        ]
        
        full_path = None
        for path in possible_paths:
            if os.path.exists(path):
                full_path = path
                break
        
        if full_path is None:
            print(f"Image not found in any location: {image_path}")
            return None
        
        # Read the image
        image = imread(full_path)
        
        # Convert to grayscale if it's a color image
        if image.ndim == 3:
            image = rgb2gray(image)
        
        # Resize the image
        image = resize(image, target_size, anti_aliasing=True)
        
        # Extract HOG features
        features, _ = hog(image, 
                         orientations=9, 
                         pixels_per_cell=(16, 16),
                         cells_per_block=(2, 2), 
                         visualize=True, 
                         block_norm='L2-Hys')
        return features
        
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

# Explore dataset structure
def explore_dataset_structure(data_base_dir='data/ap-10K/'):
    """
    Explore the dataset structure to understand where images are stored.
    """
    print("Exploring dataset structure...")
    
    # Check what directories exist
    if not os.path.exists(data_base_dir):
        print(f"Base directory does not exist: {data_base_dir}")
        return None
    
    # List all directories in the base path
    for root, dirs, files in os.walk(data_base_dir):
        level = root.replace(data_base_dir, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 2 * (level + 1)
        for file in files[:5]:  # Show first 5 files
            if file.endswith(('.jpg', '.jpeg', '.png')):
                print(f"{subindent}{file}")
        if len(files) > 5:
            print(f"{subindent}... and {len(files) - 5} more files")
    
    return None

# File discovery system
def find_image_path(image_filename, data_base_dir='data/ap-10K/'):
    """
    Search for an image file recursively in the dataset directory.
    """
    for root, dirs, files in os.walk(data_base_dir):
        if image_filename in files:
            return os.path.join(root, image_filename)
    return None

# Feature extraction function
def extract_hog_features_with_search(image_path, target_size=(224, 224), data_base_dir='data/ap-10K/'):
    """
    Extract HOG features with recursive search for the image file.
    """
    try:
        # First try the direct path
        image_filename = os.path.basename(image_path)
        
        # Search for the image recursively
        full_path = find_image_path(image_filename, data_base_dir)
        
        if full_path is None:
            print(f"Image not found anywhere: {image_filename}")
            return None
        
        # Read the image
        image = imread(full_path)
        
        # Convert to grayscale if it's a color image
        if image.ndim == 3:
            image = rgb2gray(image)
        
        # Resize the image
        image = resize(image, target_size, anti_aliasing=True)
        
        # Extract HOG features
        features, _ = hog(image, 
                         orientations=9, 
                         pixels_per_cell=(16, 16),
                         cells_per_block=(2, 2), 
                         visualize=True, 
                         block_norm='L2-Hys')
        return features
        
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

# Batch processing function with better error handling
def extract_hog_features_batch(file_paths, target_size=(224, 224), max_errors=10):
    """
    Extract HOG features for a batch of images with comprehensive error handling.
    """
    hog_features = []
    valid_indices = []
    error_count = 0
    
    for i, file_path in enumerate(file_paths):
        if error_count >= max_errors:
            print(f"Stopping due to too many errors ({max_errors})")
            break
            
        features = extract_hog_features_with_search(file_path, target_size)
        
        if features is not None:
            hog_features.append(features)
            valid_indices.append(i)
        else:
            error_count += 1
            
        # Progress update
        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1}/{len(file_paths)} images, {len(hog_features)} successful")
    
    return np.array(hog_features), valid_indices

# Alternative: Use already loaded images from preprocessing
def extract_hog_from_preloaded_images(preloaded_images):
    """
    Extract HOG features from images that have already been loaded and preprocessed.
    This avoids the file path issues.
    """
    hog_features = []
    
    for i, image in enumerate(preloaded_images):
        if image is not None:
            try:
                # Convert to grayscale if it's a color image
                if image.ndim == 3:
                    gray_image = rgb2gray(image)
                else:
                    gray_image = image
                
                # Extract HOG features
                features, _ = hog(gray_image, 
                                 orientations=9, 
                                 pixels_per_cell=(16, 16),
                                 cells_per_block=(2, 2), 
                                 visualize=True, 
                                 block_norm='L2-Hys')
                hog_features.append(features)
            except Exception as e:
                print(f"Error extracting HOG from preloaded image {i}: {e}")
                hog_features.append(None)
        else:
            hog_features.append(None)
    
    # Filter out None values
    valid_indices = [i for i, feat in enumerate(hog_features) if feat is not None]
    valid_features = [hog_features[i] for i in valid_indices]
    
    return np.array(valid_features), valid_indices

# Usage example
if __name__ == "__main__":
    # First, explore the dataset structure
    explore_dataset_structure()
    
    # Load your preprocessed data
    # Assuming you have X_train, X_val, X_test from your preprocessing
    
    # Option 1: Try with file path search (slower but more robust)
    print("Extracting HOG features with file search...")
    X_train_hog, train_indices = extract_hog_features_batch(X_train, max_errors=50)
    X_val_hog, val_indices = extract_hog_features_batch(X_val, max_errors=20)
    X_test_hog, test_indices = extract_hog_features_batch(X_test, max_errors=20)
    
    print(f"HOG features extracted: Train {len(X_train_hog)}, Val {len(X_val_hog)}, Test {len(X_test_hog)}")
    
    