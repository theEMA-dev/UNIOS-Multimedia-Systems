import os
import numpy as np
import librosa
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

def extract_features(file_path):
    """Extract audio features from a WAV file."""
    try:
        # Load audio file
        y, sr = librosa.load(file_path, duration=30)
        
        # Extract features
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        zero_crossing = librosa.feature.zero_crossing_rate(y=y)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        
        # Calculate more comprehensive statistics for each feature
        features = []
        for feature in [mfcc, spectral_centroid, zero_crossing, spectral_rolloff, chroma]:
            features.extend([
                np.mean(feature),
                np.std(feature),
                np.max(feature),
                np.min(feature),
                np.median(feature),
                np.percentile(feature, 25),  # First quartile
                np.percentile(feature, 75),  # Third quartile
                np.var(feature)  # Variance
            ])
        
        return features
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None

def load_dataset(data_path):
    """Load and prepare the dataset."""
    features = []
    labels = []
    
    for genre in os.listdir(data_path):
        genre_path = os.path.join(data_path, genre)
        if not os.path.isdir(genre_path) or genre == "doesent belong anywhere":
            continue
            
        print(f"Processing {genre} files...")
        for file_name in os.listdir(genre_path):
            if not file_name.endswith('.wav'):
                continue
                
            file_path = os.path.join(genre_path, file_name)
            extracted_features = extract_features(file_path)
            
            if extracted_features:
                features.append(extracted_features)
                labels.append(genre)
    
    return np.array(features), np.array(labels)

def main():
    # Load and prepare data
    data_path = "LAB2/data/genres"
    print("Loading dataset...")
    X, y = load_dataset(data_path)
    
    # Save number of samples per genre
    genre_counts = pd.Series(y).value_counts()
    
    print("\nDataset Summary:")
    print(f"Total number of samples: {len(y)}")
    print("\nSamples per genre:")
    for genre, count in genre_counts.items():
        print(f"{genre}: {count}")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Train Naive Bayes classifier
    print("\nTraining Naive Bayes classifier...")
    clf = GaussianNB()
    clf.fit(X_train, y_train)
    
    # Make predictions
    y_pred = clf.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Calculate cross-validation scores with stratification
    cv_scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
    
    # Print results
    print("\nResults:")
    print(f"Test Set Accuracy: {accuracy:.4f}")
    print(f"\nCross-validation scores: {cv_scores}")
    print(f"Average CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    print("\nDetailed Classification Report:")
    class_report = classification_report(y_test, y_pred)
    print(class_report)
    
    # Save results to a file
    print("\nSaving results to 'naive_bayes_results.txt'...")
    with open("LAB2/naive_bayes_results.txt", "w") as f:
        f.write("Music Genre Classification - Naive Bayes Results\n")
        f.write("=============================================\n\n")
        f.write("Dataset Summary:\n")
        f.write(f"Total number of samples: {len(y)}\n\n")
        f.write("Samples per genre:\n")
        for genre, count in genre_counts.items():
            f.write(f"{genre}: {count}\n")
        f.write("\nTest Set Accuracy: {:.4f}\n".format(accuracy))
        f.write("\nCross-validation scores: {}\n".format(cv_scores))
        f.write("Average CV score: {:.4f} (+/- {:.4f})\n".format(cv_scores.mean(), cv_scores.std() * 2))
        f.write("\nClassification Report:\n")
        f.write(class_report)
        
    print("\nConfusion Matrix:")
    genres = sorted(list(set(y)))
    conf_df = pd.DataFrame(conf_matrix, index=genres, columns=genres)
    print(conf_df)
    
    # Save confusion matrix to CSV
    conf_df.to_csv("LAB2/confusion_matrix.csv")
    print("\nConfusion matrix saved to 'confusion_matrix.csv'")

if __name__ == "__main__":
    main()
