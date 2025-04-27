import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def load_and_process_features(json_path):
    # Load data from JSON file
    with open(json_path, "r") as fp:
        data = json.load(fp)
    
    # Convert to numpy arrays
    X_mfcc = np.array(data["mfcc"])    # Shape: (4800, 13, 216)
    X_chroma = np.array(data["chroma"])  # Shape: (4800, 12, 216)
    y = np.array(data["labels"])
    mappings = data["mappings"]
    
    print(f"Original MFCC shape: {X_mfcc.shape}")
    print(f"Original Chroma shape: {X_chroma.shape}")
    
    # Extract statistics for each coefficient/pitch class across time frames
    def extract_stats(features):
        # Features shape: (n_samples, n_features, time_frames)
        mean_vals = np.mean(features, axis=2)  # Mean across time
        std_vals = np.std(features, axis=2)    # Std across time
        max_vals = np.max(features, axis=2)    # Max across time
        min_vals = np.min(features, axis=2)    # Min across time
        median_vals = np.median(features, axis=2)  # Median across time
        
        # Stack all statistics horizontally
        return np.column_stack([
            mean_vals, std_vals, max_vals, min_vals, median_vals
        ])
    
    # Process MFCC features (13 coefficients × 5 stats = 65 features)
    mfcc_processed = extract_stats(X_mfcc)
    
    # Process Chroma features (12 pitch classes × 5 stats = 60 features)
    chroma_processed = extract_stats(X_chroma)
    
    # Combine features
    X_combined = np.hstack([mfcc_processed, chroma_processed])
    
    print(f"MFCC features shape: {mfcc_processed.shape}")
    print(f"Chroma features shape: {chroma_processed.shape}")
    print(f"Final combined shape: {X_combined.shape}")
    
    return X_combined, y, mappings

def main():
    # Configuration
    TRAIN_PATH = "d:/Dev/UNIOS-Multimedia-Systems/LAB5/test.json"
    TEST_PATH = "d:/Dev/UNIOS-Multimedia-Systems/LAB5/test.json"
    
    # Load and process training data
    print("\nLoading training data...")
    X_train, y_train, mappings = load_and_process_features(TRAIN_PATH)
    
    # Load and process test data
    print("\nLoading test data...")
    X_test, y_test, _ = load_and_process_features(TEST_PATH)
    
    # Feature scaling (optional for Naive Bayes but can help)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Naive Bayes classifier
    print("\nTraining Naive Bayes classifier...")
    nb = GaussianNB()
    nb.fit(X_train_scaled, y_train)
    
    # Calculate genre counts
    genre_counts = pd.Series(y_train).value_counts().sort_index()
    genre_counts.index = mappings
    
    print("\nDataset Summary:")
    print(f"Total number of training samples: {len(y_train)}")
    print(f"Total number of test samples: {len(y_test)}")
    print("\nSamples per genre:")
    for genre, count in genre_counts.items():
        print(f"{genre}: {count}")
    
    # Calculate cross-validation scores
    print("\nCalculating cross-validation scores...")
    cv_scores = cross_val_score(nb, X_train_scaled, y_train, cv=5, scoring='accuracy')
    
    # Predict on test set
    y_pred = nb.predict(X_test_scaled)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Classification report
    print("\nTest set performance:")
    class_report = classification_report(y_test, y_pred, target_names=mappings)
    print(class_report)
    
    # Save results to a file
    print("\nSaving results to 'naive_bayes_results.txt'...")
    with open("LAB5/naive_bayes_results.txt", "w") as f:
        f.write("Music Genre Classification - Naive Bayes Results\n")
        f.write("=============================================\n\n")
        f.write("Dataset Summary:\n")
        f.write(f"Total number of training samples: {len(y_train)}\n")
        f.write(f"Total number of test samples: {len(y_test)}\n\n")
        f.write("Samples per genre:\n")
        for genre, count in genre_counts.items():
            f.write(f"{genre}: {count}\n")
        f.write(f"\nTest Set Accuracy: {accuracy:.4f}\n\n")
        f.write(f"Cross-validation scores: {cv_scores}\n")
        f.write(f"Average CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})\n\n")
        f.write("Classification Report:\n")
        f.write(class_report)
    
    # Confusion matrix
    print("\nGenerating confusion matrix...")
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Save confusion matrix
    conf_df = pd.DataFrame(conf_matrix, index=mappings, columns=mappings)
    conf_df.to_csv("LAB5/confusion_matrix.csv")
    print("Confusion matrix saved to 'confusion_matrix.csv'")
    
    # Plot confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=mappings)
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.xticks(rotation=45)
    plt.title("Naive Bayes Confusion Matrix for Music Genre Classification")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
