"""This file was created by Lucasdkp"""

# Data Preprocessing bby importing the necessary packages for the project
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import math

# Load annotations from the data folder dowloaded from the AP-10K repository
with open('data/ap-10K/annotations/ap10k-train-split1.json') as f:
    data = json.load(f)
    
# Proxy label creation to calculate the body length
def calculate_body_length(keypoints, format_type='coco'):
    """
    Calculate body length from keypoints using nose (index 0) and tail (index 16) keypoints.
    In COCO format, keypoints are stored as a flat list: [x1, y1, v1, x2, y2, v2, ...]
    """
    if not isinstance(keypoints, list) or len(keypoints) < 51:  # 17 keypoints * 3 = 51
        return None
    
    # Extract nose (keypoint 0) and tail (keypoint 16) coordinates
    # Nose: indices 0, 1, 2
    nose_x, nose_y, nose_v = keypoints[0], keypoints[1], keypoints[2]
    # Tail: indices 48, 49, 50 (16*3=48, 16*3+1=49, 16*3+2=50)
    tail_x, tail_y, tail_v = keypoints[48], keypoints[49], keypoints[50]
    
    # Check if both keypoints are visible (v > 0 typically means visible)
    if nose_v > 0 and tail_v > 0:
        # Calculate Euclidean distance
        body_length = math.sqrt((tail_x - nose_x)**2 + (tail_y - nose_y)**2)
        return body_length
    
    return None

# Create dataset
dataset = []

# Create mapping from category_id to category_name
category_map = {}
for category in data['categories']:
    category_map[category['id']] = category['name']

# Create mapping from image_id to image info
image_map = {}
for img_info in data['images']:
    image_map[img_info['id']] = img_info

# Process annotations
for ann in data['annotations']:
    category_id = ann['category_id']
    category_name = category_map.get(category_id, '')
    
    if category_name.lower() in ['goat', 'sheep'] and 'keypoints' in ann:
        body_len = calculate_body_length(ann['keypoints'])
        if body_len:
            image_id = ann['image_id']
            img_info = image_map.get(image_id, {})
            
            dataset.append({
                'image_id': image_id,
                'file_name': img_info.get('file_name', f'{image_id}.jpg'),
                'species': category_name,
                'body_length': body_len,
                'bbox': ann.get('bbox', []),
                'keypoints': ann.get('keypoints', [])
            })

# Convert to DataFrame
df = pd.DataFrame(dataset)

# Check if we have enough data
if len(df) == 0:
    raise ValueError("No valid data found. Check your filtering criteria and keypoint format.")

print(f"Found {len(df)} valid images")
print(df['species'].value_counts())

# Create class labels(Underweight, Healthy, Overweight) based on body length percentiles PER SPECIES
df['weight_class'] = np.nan

for species in df['species'].unique():
    species_data = df[df['species'] == species]
    
    if len(species_data) >= 3:  # Need at least 3 samples for quantiles
        try:
            # Use quantiles for binning
            quantiles = species_data['body_length'].quantile([0.33, 0.66])
            def assign_class(length):
                if length <= quantiles.iloc[0]:
                    return 0  # Underweight
                elif length <= quantiles.iloc[1]:
                    return 1  # Healthy
                else:
                    return 2  # Overweight
            
            df.loc[df['species'] == species, 'weight_class'] = species_data['body_length'].apply(assign_class)
        except:
            # Fallback: equal width binning
            df.loc[df['species'] == species, 'weight_class'] = pd.cut(
                species_data['body_length'], 
                bins=3, 
                labels=[0, 1, 2], 
                include_lowest=True
            )
    else:
        # If not enough samples, assign all to class 1 (Healthy)
        df.loc[df['species'] == species, 'weight_class'] = 1

# Drop any remaining NaN values
df = df.dropna(subset=['weight_class'])
df['weight_class'] = df['weight_class'].astype(int)

print("\nClass distribution:")
print(df['weight_class'].value_counts().sort_index())

# Split the data - ensure we have enough samples for stratification
if len(df['weight_class'].unique()) > 1 and df['weight_class'].value_counts().min() >= 2:
    X = df['file_name']  # Use file_name to load images
    y = df['weight_class']
    
    # First split: train vs temp (val + test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    
    # Second split: val vs test (50/50 of temp)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )
else:
    # Fallback: simple split without stratification
    X_train, X_temp, y_train, y_temp = train_test_split(
        df['file_name'], df['weight_class'], test_size=0.3, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

print(f"\nTrain: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

# Save the splits for later use
df_train = df[df['file_name'].isin(X_train)]
df_val = df[df['file_name'].isin(X_val)]
df_test = df[df['file_name'].isin(X_test)]

df_train.to_csv('data/train_split.csv', index=False)
df_val.to_csv('data/val_split.csv', index=False)
df_test.to_csv('data/test_split.csv', index=False)

print("Data preprocessing completed successfully!")