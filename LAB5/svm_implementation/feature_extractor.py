import os
import numpy as np
import librosa
import librosa.display
import json
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def extract_features(file_path):
    """
    Extract audio features from a WAV file for SVM-based music genre classification.
    Returns a feature vector containing various audio characteristics.
    """
    try:
        # Load audio file
        y, sr = librosa.load(file_path, duration=30)
        
        # Initialize feature list
        features = []
        
        # 1. MFCC (Mel-frequency cepstral coefficients)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        
        # 2. Spectral Centroid
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        
        # 3. Zero Crossing Rate
        zero_crossing = librosa.feature.zero_crossing_rate(y=y)
        
        # 4. Spectral Rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        
        # 5. Chroma Features
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        
        # 6. Tempo (new)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        features.append(tempo)
        
        # 7. Spectral Bandwidth (new)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        
        # 8. Spectral Contrast (new)
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        
        # 9. RMS Energy (new)
        rms = librosa.feature.rms(y=y)
        
        # 10. Spectral Flatness (new)
        spectral_flatness = librosa.feature.spectral_flatness(y=y)
        
        # Calculate statistics for each feature
        feature_list = [
            mfcc, 
            spectral_centroid,
            zero_crossing,
            spectral_rolloff,
            chroma,
            spectral_bandwidth,
            spectral_contrast,
            rms,
            spectral_flatness
        ]
        
        for feature in feature_list:
            features.extend([
                np.mean(feature),
                np.std(feature),
                np.max(feature),
                np.min(feature),
                np.median(feature),
                np.percentile(feature, 25),  # First quartile
                np.percentile(feature, 75),  # Third quartile
                np.var(feature),  # Variance
                np.percentile(feature, 10),  # 10th percentile
                np.percentile(feature, 90),  # 90th percentile
                stats.kurtosis(feature.ravel()),  # Kurtosis
                stats.skew(feature.ravel())  # Skewness
            ])
        
        return features
    
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None

def extract_dataset_features(data_path):
    """
    Extract features from all WAV files in the dataset and organize them by genre.
    Returns a dictionary with genres as keys and lists of feature vectors as values.
    """
    dataset = {}
    
    for genre in os.listdir(data_path):
        genre_path = os.path.join(data_path, genre)
        if not os.path.isdir(genre_path):
            continue
            
        print(f"Processing {genre} files...")
        dataset[genre] = []
        
        for file_name in os.listdir(genre_path):
            if not file_name.endswith('.wav'):
                continue
                
            file_path = os.path.join(genre_path, file_name)
            extracted_features = extract_features(file_path)
            
            if extracted_features:
                # Convert numpy array to list for JSON serialization
                dataset[genre].append({
                    'file': file_name,
                    'features': [float(f) for f in extracted_features]
                })
    
    return dataset

def save_features_to_json(dataset, output_file):
    """Save the extracted features to a JSON file."""
    with open(output_file, 'w') as f:
        json.dump(dataset, f, indent=2)
    print(f"Features saved to {output_file}")

def main():
    """Extract features from the testing dataset and save them to JSON."""
    data_path = "LAB5/svm_implementation/genres_testing_data"
    output_file = "LAB5/svm_implementation/training.json"
    
    print("Extracting features from testing dataset...")
    dataset = extract_dataset_features(data_path)
    
    print("\nDataset Summary:")
    for genre, data in dataset.items():
        print(f"{genre}: {len(data)} files processed")
    
    save_features_to_json(dataset, output_file)

if __name__ == "__main__":
    main()
