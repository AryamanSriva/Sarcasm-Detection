import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import pickle
import os

def load_and_preprocess_data(json_file_path="Sarcasm.json"):
    """
    Load and preprocess the sarcasm detection dataset.
    
    Args:
        json_file_path (str): Path to the JSON file containing the dataset
    
    Returns:
        tuple: Preprocessed training and testing data
    """
    # Load data
    print("Loading data...")
    data = pd.read_json(json_file_path, lines=True)
    print(f"Data shape: {data.shape}")
    print("First few rows:")
    print(data.head())
    
    # Map binary labels to descriptive labels
    data["is_sarcastic"] = data["is_sarcastic"].map({0: "Not Sarcasm", 1: "Sarcasm"})
    print("\nAfter label mapping:")
    print(data.head())
    
    # Extract features and labels
    data = data[["headline", "is_sarcastic"]]
    x = np.array(data["headline"])
    y = np.array(data["is_sarcastic"])
    
    # Vectorize the text data
    print("\nVectorizing text data...")
    cv = CountVectorizer()
    X = cv.fit_transform(x)
    
    # Split the data
    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Testing set size: {X_test.shape[0]}")
    
    # Save the vectorizer for later use
    if not os.path.exists('models'):
        os.makedirs('models')
    
    with open('models/vectorizer.pkl', 'wb') as f:
        pickle.dump(cv, f)
    print("Vectorizer saved to models/vectorizer.pkl")
    
    return X_train, X_test, y_train, y_test, cv

if __name__ == "__main__":
    # Check if data file exists
    if not os.path.exists("Sarcasm.json"):
        print("Error: Sarcasm.json file not found!")
        print("Please make sure the dataset file is in the same directory.")
        exit(1)
    
    # Preprocess data
    X_train, X_test, y_train, y_test, vectorizer = load_and_preprocess_data()
    print("\nData preprocessing completed successfully!")
