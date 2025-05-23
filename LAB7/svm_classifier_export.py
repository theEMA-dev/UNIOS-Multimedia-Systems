import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import joblib
import os

def load_and_process_features(json_path):
    # Load data from JSON file
    try:
        with open(json_path, 'rb') as fp:
            buffer_size = 64 * 1024 * 1024  # 64MB buffer
            chunks = []
            while True:
                chunk = fp.read(buffer_size)
                if not chunk:
                    break
                chunks.append(chunk)
            json_str = b''.join(chunks).decode('utf-8')
            data = json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        raise
    except Exception as e:
        print(f"Error loading file: {e}")
        raise
    
    X_mfcc = np.array(data["mfcc"])
    X_chroma = np.array(data["chroma"])
    X_centroid = np.array(data["spectral_centroid"])
    X_bandwidth = np.array(data["spectral_bandwidth"])
    X_contrast = np.array(data["spectral_contrast"])
    X_zcr = np.array(data["zcr"])
    y = np.array(data["labels"])
    
    def extract_stats(features):
        mean_vals = np.mean(features, axis=2)
        std_vals = np.std(features, axis=2)
        max_vals = np.max(features, axis=2)
        min_vals = np.min(features, axis=2)
        median_vals = np.median(features, axis=2)
        delta_vals = np.diff(features, axis=2)
        delta_delta_vals = np.diff(delta_vals, axis=2)
        delta_mean = np.mean(delta_vals, axis=2)
        delta_std = np.std(delta_vals, axis=2)
        delta_delta_mean = np.mean(delta_delta_vals, axis=2)
        delta_delta_std = np.std(delta_delta_vals, axis=2)
        return np.column_stack([
            mean_vals, std_vals, max_vals, min_vals, median_vals,
            delta_mean, delta_std, delta_delta_mean, delta_delta_std
        ])
    
    mfcc_processed = extract_stats(X_mfcc)
    chroma_processed = extract_stats(X_chroma)
    centroid_processed = extract_stats(X_centroid)
    bandwidth_processed = extract_stats(X_bandwidth)
    contrast_processed = extract_stats(X_contrast) # Note: Original script had X_contrast shape (4800, 7, 216) vs (4800, 1, 216) for others. Assuming this is intended.
    zcr_processed = extract_stats(X_zcr)
    
    X_combined = np.hstack([
        mfcc_processed,
        chroma_processed,
        centroid_processed,
        bandwidth_processed,
        contrast_processed,
        zcr_processed
    ])
    
    print(f"Final combined shape for training: {X_combined.shape}")
    return X_combined, y

def main():
    # Configuration
    TRAIN_PATH = "d:/Dev/UNIOS-Multimedia-Systems/LAB5/train_v2.json"
    MODEL_DIR = "LAB7/models"
    PIPELINE_PATH = os.path.join(MODEL_DIR, "svm_pipeline.save")

    # Create model directory if it doesn't exist
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Load and process training data
    print("\nLoading training data...")
    X_train, y_train = load_and_process_features(TRAIN_PATH)
    
    # Define preprocessing steps
    scaler = StandardScaler()
    # From previous run: Number of components for 99% variance: 203
    n_components_99 = 203 
    pca = PCA(n_components=n_components_99)
    
    # Define SVM with best parameters
    svm_params = {
        'C': 50,
        'class_weight': 'balanced',
        'degree': 2,
        'gamma': 0.001,
        'kernel': 'rbf',
        'random_state': 42 # for reproducibility
    }
    svm_model = SVC(**svm_params)
    
    # Create the pipeline
    print("\nCreating and training the pipeline (Scaler -> PCA -> SVM)...")
    pipeline = Pipeline([
        ('scaler', scaler),
        ('pca', pca),
        ('svm', svm_model)
    ])
    
    # Train the entire pipeline
    pipeline.fit(X_train, y_train)
    
    # Save the entire pipeline
    joblib.dump(pipeline, PIPELINE_PATH)
    print(f"Trained pipeline (Scaler, PCA, SVM) saved to {PIPELINE_PATH}")

if __name__ == "__main__":
    main()
