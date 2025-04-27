import json
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
import matplotlib.pyplot as plt

def load_data(json_file):
    """Load and prepare data from JSON file."""
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    X = []  # Features
    y = []  # Labels
    
    for genre, songs in data.items():
        for song in songs:
            X.append(song['features'])
            y.append(genre)
    
    return np.array(X), np.array(y)

def train_svm(X_train, y_train):
    """Train SVM classifier with RBF kernel."""
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Initialize and train SVM
    svm = SVC(kernel='rbf', C=10.0, gamma='scale', random_state=42)
    svm.fit(X_train_scaled, y_train)
    
    return svm, scaler

def evaluate_model(model, scaler, X_test, y_test, genres):
    """Evaluate the model and print performance metrics."""
    # Scale test data
    X_test_scaled = scaler.transform(X_test)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy:.4f}")
    
    # Print detailed classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Create and save confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    conf_df = pd.DataFrame(conf_matrix, index=genres, columns=genres)
    print("\nConfusion Matrix:")
    print(conf_df)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=genres)
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.xticks(rotation=45)
    plt.title("SVM Confusion Matrix for Music Genre Classification")
    plt.tight_layout()
    plt.show()
    
    # Save results
    with open("LAB5/svm_implementation/svm_results.txt", "w") as f:
        f.write("Music Genre Classification - SVM Results\n")
        f.write("=====================================\n\n")
        f.write(f"Accuracy: {accuracy:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(y_test, y_pred))
        
    # Save confusion matrix
    conf_df.to_csv("LAB5/svm_implementation/svm_confusion_matrix.csv")
    
    return accuracy, y_pred

def main():
    # Load training data
    print("Loading training data...")
    X_train, y_train = load_data("LAB5/svm_implementation/training.json")
    
    # Train model
    print("\nTraining SVM classifier...")
    model, scaler = train_svm(X_train, y_train)
    
    # Load and evaluate on test data
    print("\nLoading and evaluating on test data...")
    X_test, y_test = load_data("LAB5/svm_implementation/test.json")
    
    # Get unique genres
    genres = sorted(list(set(y_train)))
    
    # Evaluate model
    evaluate_model(model, scaler, X_test, y_test, genres)
    
if __name__ == "__main__":
    main()
