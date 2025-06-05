import pandas as pd
import numpy as np
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import os
from data_preprocessing import load_and_preprocess_data

def train_model(X_train, y_train):
    """
    Train a Bernoulli Naive Bayes model for sarcasm detection.
    
    Args:
        X_train: Training features
        y_train: Training labels
    
    Returns:
        Trained model
    """
    print("Training Bernoulli Naive Bayes model...")
    model = BernoulliNB()
    model.fit(X_train, y_train)
    print("Model training completed!")
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model on test data.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
    
    Returns:
        Model accuracy
    """
    print("\nEvaluating model...")
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.4f}")
    
    # Detailed classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion Matrix
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    return accuracy

def save_model(model, model_path='models/sarcasm_model.pkl'):
    """
    Save the trained model to disk.
    
    Args:
        model: Trained model
        model_path (str): Path to save the model
    """
    if not os.path.exists('models'):
        os.makedirs('models')
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {model_path}")

def main():
    """
    Main function to train and evaluate the sarcasm detection model.
    """
    # Check if data file exists
    if not os.path.exists("Sarcasm.json"):
        print("Error: Sarcasm.json file not found!")
        print("Please make sure the dataset file is in the same directory.")
        return
    
    # Load and preprocess data
    X_train, X_test, y_train, y_test, vectorizer = load_and_preprocess_data()
    
    # Train the model
    model = train_model(X_train, y_train)
    
    # Evaluate the model
    accuracy = evaluate_model(model, X_test, y_test)
    
    # Save the model
    save_model(model)
    
    print(f"\nTraining completed! Final accuracy: {accuracy:.4f}")
    print("Model and vectorizer saved in the 'models' directory.")

if __name__ == "__main__":
    main()
