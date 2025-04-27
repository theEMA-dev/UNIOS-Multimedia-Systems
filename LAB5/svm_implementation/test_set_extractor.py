import json
import random

def extract_test_set(training_json_path, test_json_path, samples_per_genre=60):
    """Extract a test set from the training data with specified samples per genre."""
    
    # Load training data
    with open(training_json_path, 'r') as f:
        training_data = json.load(f)
    
    # Initialize test set
    test_set = {}
    training_set = {}
    
    # For each genre, randomly select samples for test set
    for genre, songs in training_data.items():
        print(f"Processing {genre}...")
        # Randomly select samples_per_genre items
        test_samples = random.sample(songs, samples_per_genre)
        
        # Add to test set
        test_set[genre] = test_samples
        
        # Remove selected samples from training data
        training_samples = [s for s in songs if s not in test_samples]
        training_set[genre] = training_samples
        
        print(f"Selected {len(test_samples)} test samples and kept {len(training_samples)} training samples")
    
    # Save test set
    with open(test_json_path, 'w') as f:
        json.dump(test_set, f, indent=2)
    print(f"\nTest set saved to {test_json_path}")
    
    # Update training set
    with open(training_json_path, 'w') as f:
        json.dump(training_set, f, indent=2)
    print(f"Updated training set saved to {training_json_path}")

def main():
    training_path = "LAB5/svm_implementation/training.json"
    test_path = "LAB5/svm_implementation/test.json"
    
    print("Extracting test set (60 samples per genre)...")
    extract_test_set(training_path, test_path)
    
    # Print summary
    with open(test_path, 'r') as f:
        test_data = json.load(f)
    print("\nTest Set Summary:")
    for genre, samples in test_data.items():
        print(f"{genre}: {len(samples)} samples")

if __name__ == "__main__":
    main()
