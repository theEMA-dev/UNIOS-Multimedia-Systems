# LAB2: Basic Music Genre Classification

## Overview

This lab implements a basic music genre classification system using raw audio files and Naive Bayes classification.

## Components

- `naive_bayes_classifier.py` - Main implementation file
- `naive_bayes_results.txt` - Classification results and metrics
- `confusion_matrix.csv` - Detailed classification performance matrix
- `data/genres/` - Directory containing WAV audio files organized by genre

## Implementation Details

### Feature Extraction
Direct feature extraction from WAV files using librosa:
- MFCC (Mel-frequency cepstral coefficients)
- Spectral centroid
- Zero crossing rate
- Spectral rolloff
- Chroma features

### Classification
- Uses Gaussian Naive Bayes classifier
- 80/20 train/test split
- Cross-validation with 5 folds

## Results

- Test Set Accuracy: 43.75%
- Cross-validation average score: 53.75%
- Best performance on:
  - Classical (80% F1-score)
  - Hip-hop (67% F1-score)
  - Pop (50% F1-score)

## Dataset

Small dataset with 80 samples:
- 10 samples per genre
- 8 genres: classical, disco, hiphop, jazz, metal, pop, reggae, rock
- Short WAV audio clips (~30 seconds each)

## Usage

```bash
python naive_bayes_classifier.py
```

The script will:
1. Process all WAV files in the data/genres directory
2. Extract audio features
3. Train the classifier
4. Generate performance metrics
5. Save results to files
