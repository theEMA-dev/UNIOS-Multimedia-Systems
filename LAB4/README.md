# LAB4: Advanced Music Genre Classification

## Overview

This lab implements an advanced music genre classification system using pre-extracted features with Naive Bayes classification. It builds upon the LAB2 implementation with significant improvements in both methodology and performance.

## Components

- `naive_bayes_classifier.py` - Naive Bayes implementation
- `naive_bayes_results.txt` - Classification results and metrics
- `confusion_matrix.csv` - Detailed classification performance matrix
- `audio_features.json` - Pre-extracted audio features (not included in repository)

## Features

### Pre-extracted Features
Instead of processing raw audio files, this implementation uses pre-extracted features:
- MFCC features (4800, 13, 216)
- Chroma features (4800, 12, 216)

### Feature Processing
Advanced statistical feature extraction:
- Mean across time frames
- Standard deviation
- Maximum values
- Minimum values
- Median values

### Classification
Improved Naive Bayes implementation with:
- StandardScaler for feature normalization
- 5-fold cross-validation
- Detailed performance metrics

## Results

Naive Bayes Implementation Results:
- Overall accuracy: 54.9%
- Cross-validation average score: 56.41%
- Best performance on:
  - Classical (86% F1-score)
  - Metal (71% F1-score)
  - Jazz (66% F1-score)

## Dataset

Large-scale dataset with 4800 samples:
- 600 samples per genre
- 8 genres: classical, disco, hiphop, jazz, metal, pop, reggae, rock

## Important Note

The `audio_features.json` file is not included in the repository due to its large size. This file contains the pre-extracted MFCC and Chroma features for all 4800 audio samples. To use this implementation:

1. Obtain the audio_features.json file separately
2. Place it in the LAB4 directory
3. File structure should be:
   ```
   features:
   - mfcc: (4800, 13, 216)
   - chroma: (4800, 12, 216)
   - labels: [0-7]
   - mappings: [genre names]
   ```

## Usage

```bash
python NB_final.py
```

The script will:
1. Load pre-extracted features
2. Process and combine features
3. Train the Naive Bayes classifier
4. Generate performance metrics
5. Save results to files
