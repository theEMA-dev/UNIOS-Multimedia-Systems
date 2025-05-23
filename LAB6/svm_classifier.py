import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_process_features(json_path):
    # Load data from JSON file
    try:
        # Read the file in binary mode with a large buffer
        with open(json_path, 'rb') as fp:
            # Use a large buffer size for reading
            buffer_size = 64 * 1024 * 1024  # 64MB buffer
            chunks = []
            
            while True:
                chunk = fp.read(buffer_size)
                if not chunk:
                    break
                chunks.append(chunk)
            
            # Combine chunks and decode
            json_str = b''.join(chunks).decode('utf-8')
            
            # Parse JSON
            data = json.loads(json_str)
            
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        raise
    except Exception as e:
        print(f"Error loading file: {e}")
        raise
    
    # Convert to numpy arrays
    X_mfcc = np.array(data["mfcc"])           # Shape: (4800, 13, 216)
    X_chroma = np.array(data["chroma"])       # Shape: (4800, 12, 216)
    X_centroid = np.array(data["spectral_centroid"])   # Shape: (4800, 1, 216)
    X_bandwidth = np.array(data["spectral_bandwidth"]) # Shape: (4800, 1, 216)
    X_contrast = np.array(data["spectral_contrast"])   # Shape: (4800, 1, 216)
    X_zcr = np.array(data["zcr"])            # Shape: (4800, 1, 216)
    y = np.array(data["labels"])
    mappings = data["mappings"]
    
    print(f"Original MFCC shape: {X_mfcc.shape}")
    print(f"Original Chroma shape: {X_chroma.shape}")
    print(f"Original Centroid shape: {X_centroid.shape}")
    print(f"Original Bandwidth shape: {X_bandwidth.shape}")
    print(f"Original Contrast shape: {X_contrast.shape}")
    print(f"Original ZCR shape: {X_zcr.shape}")
    
    # Extract statistics for each coefficient/pitch class across time frames
    def extract_stats(features):
        # Features shape: (n_samples, n_features, time_frames)
        mean_vals = np.mean(features, axis=2)  # Mean across time
        std_vals = np.std(features, axis=2)    # Std across time
        max_vals = np.max(features, axis=2)    # Max across time
        min_vals = np.min(features, axis=2)    # Min across time
        median_vals = np.median(features, axis=2)  # Median across time
        
        # Stack all statistics horizontally
        delta_vals = np.diff(features, axis=2)  # First derivative
        delta_delta_vals = np.diff(delta_vals, axis=2)  # Second derivative
        
        delta_mean = np.mean(delta_vals, axis=2)
        delta_std = np.std(delta_vals, axis=2)
        delta_delta_mean = np.mean(delta_delta_vals, axis=2)
        delta_delta_std = np.std(delta_delta_vals, axis=2)
        
        return np.column_stack([
            mean_vals, std_vals, max_vals, min_vals, median_vals,
            delta_mean, delta_std, delta_delta_mean, delta_delta_std
        ])
    
    # Process MFCC features (13 coefficients × 5 stats = 65 features)
    mfcc_processed = extract_stats(X_mfcc)
    
    # Process Chroma features (12 pitch classes × 5 stats = 60 features)
    chroma_processed = extract_stats(X_chroma)
    
    # Process Centroid features (1 value × 5 stats = 5 features)
    centroid_processed = extract_stats(X_centroid)
    
    # Process Bandwidth features (1 value × 5 stats = 5 features)
    bandwidth_processed = extract_stats(X_bandwidth)
    
    # Process Contrast features (1 value × 5 stats = 5 features)
    contrast_processed = extract_stats(X_contrast)
    
    # Process ZCR features (1 value × 5 stats = 5 features)
    zcr_processed = extract_stats(X_zcr)
    
    # Combine all features
    X_combined = np.hstack([
        mfcc_processed,      # 65 features
        chroma_processed,    # 60 features
        centroid_processed,  # 5 features
        bandwidth_processed, # 5 features
        contrast_processed,  # 5 features
        zcr_processed       # 5 features
    ])
    
    print(f"MFCC features shape: {mfcc_processed.shape}")
    print(f"Chroma features shape: {chroma_processed.shape}")
    print(f"Centroid features shape: {centroid_processed.shape}")
    print(f"Bandwidth features shape: {bandwidth_processed.shape}")
    print(f"Contrast features shape: {contrast_processed.shape}")
    print(f"ZCR features shape: {zcr_processed.shape}")
    print(f"Final combined shape: {X_combined.shape}")
    
    return X_combined, y, mappings

def main():
    # Configuration
    TRAIN_PATH = "d:/Dev/UNIOS-Multimedia-Systems/LAB5/train_v2.json"
    TEST_PATH = "d:/Dev/UNIOS-Multimedia-Systems/LAB5/test_v2.json"
    
    # Load and process training data
    print("\nLoading training data...")
    X_train, y_train, mappings = load_and_process_features(TRAIN_PATH)
    
    # Load and process test data
    print("\nLoading test data...")
    X_test, y_test, _ = load_and_process_features(TEST_PATH)
    
    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Apply PCA with variance ratio analysis
    print("\nApplying PCA with component analysis...")
    pca = PCA()
    X_train_pca_full = pca.fit_transform(X_train_scaled)
    
    # Analyze cumulative variance ratio
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    n_components_95 = np.argmax(cumsum >= 0.95) + 1
    n_components_99 = np.argmax(cumsum >= 0.99) + 1
    
    print(f"Number of components for 95% variance: {n_components_95}")
    print(f"Number of components for 99% variance: {n_components_99}")
    
    # Use 99% variance for better feature representation
    pca = PCA(n_components=n_components_99)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    print(f"Reduced dimensions from {X_train_scaled.shape[1]} to {X_train_pca.shape[1]} features")
    
    # Fine-tuned parameter grid based on previous best results
    param_grid = {
        'kernel': ['rbf', 'poly', 'sigmoid'],  # Explore additional kernels
        'C': [10, 50, 75, 100, 125, 150, 200],  # Expand range of C values
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.05, 0.1],  # Add more gamma values
        'class_weight': ['balanced', None],  # Include unbalanced option
        'degree': [2, 3, 4]  # Explore higher polynomial degrees
    }
    
    # Perform Grid Search
    print("\nPerforming Grid Search for optimal parameters...")
    svm = SVC(random_state=42)
    grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
    grid_search.fit(X_train_pca, y_train)
    
    # Get best parameters and model
    best_params = grid_search.best_params_
    best_svm = grid_search.best_estimator_
    
    print("\nBest parameters found:")
    for param, value in best_params.items():
        print(f"{param}: {value}")
    
    print(f"\nBest cross-validation score: {grid_search.best_score_:.4f}")
        
    # Train final model with best parameters
    print("\nTraining final SVM model with best parameters...")
    best_svm.fit(X_train_pca, y_train)
    
    # Calculate genre counts
    genre_counts = pd.Series(y_train).value_counts().sort_index()
    genre_counts.index = mappings
    
    print("\nDataset Summary:")
    print(f"Total number of training samples: {len(y_train)}")
    print(f"Total number of test samples: {len(y_test)}")
    print("\nSamples per genre:")
    for genre, count in genre_counts.items():
        print(f"{genre}: {count}")
    
    # Calculate cross-validation scores with best model
    print("\nCalculating cross-validation scores...")
    cv_scores = cross_val_score(best_svm, X_train_pca, y_train, cv=5, scoring='accuracy')
    
    # Predict on test set
    y_pred = best_svm.predict(X_test_pca)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Classification report
    print("\nTest set performance:")
    class_report = classification_report(y_test, y_pred, target_names=mappings)
    print(class_report)
    
    # Save results to a file
    print("\nSaving results to 'svm_results.txt'...")
    with open("LAB6/svm_results.txt", "w") as f:
        f.write("Music Genre Classification - SVM Results (with PCA and GridSearch)\n")
        f.write("=========================================================\n\n")
        f.write("Dataset Summary:\n")
        f.write(f"Total number of training samples: {len(y_train)}\n")
        f.write(f"Total number of test samples: {len(y_test)}\n\n")
        f.write("Samples per genre:\n")
        for genre, count in genre_counts.items():
            f.write(f"{genre}: {count}\n")
        f.write(f"\nTest Set Accuracy: {accuracy:.4f}\n\n")
        f.write("Best Parameters:\n")
        for param, value in best_params.items():
            f.write(f"{param}: {value}\n")
        f.write(f"\nNumber of features after PCA: {X_train_pca.shape[1]}\n")
        f.write(f"Explained variance ratio: {sum(pca.explained_variance_ratio_):.4f}\n\n")
        f.write(f"Cross-validation scores: {cv_scores}\n")
        f.write(f"Average CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})\n\n")
        f.write("Classification Report:\n")
        f.write(class_report)
    
    # Confusion matrix
    print("\nGenerating confusion matrix...")
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Save confusion matrix
    conf_df = pd.DataFrame(conf_matrix, index=mappings, columns=mappings)
    conf_df.to_csv("LAB6/confusion_matrix.csv")
    print("Confusion matrix saved to 'confusion_matrix.csv'")
    
    # Normalize confusion matrix for plotting only
    conf_matrix_normalized = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
    
    # Plot normalized confusion matrix
    plt.figure(figsize=(16, 14))
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_normalized, display_labels=mappings)
    disp.plot(cmap=plt.cm.Blues, values_format='.2f')  # Show 2 decimal places
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.title("Normalized SVM Confusion Matrix")
    plt.tight_layout()
    plt.savefig("LAB6/normalized_confusion_matrix.png")
    plt.show()
    
    # Seaborn heatmap for confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='YlGnBu', xticklabels=mappings, yticklabels=mappings)
    plt.title("Confusion Matrix Heatmap")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig("LAB6/confusion_matrix_heatmap.png")
    plt.show()
    
    # Seaborn barplot for class-wise accuracy
    class_accuracy = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
    plt.figure(figsize=(10, 6))
    bars = sns.barplot(x=mappings, y=class_accuracy, palette="viridis")
    plt.title("Class-wise Accuracy")
    plt.xlabel("Genres")
    plt.ylabel("Accuracy")
    plt.xticks(rotation=45, ha='right')

    # Add text annotations to each bar
    for bar in bars.patches:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f'{yval:.2f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig("LAB6/class_wise_accuracy.png")
    plt.show()
    
if __name__ == "__main__":
    main()
