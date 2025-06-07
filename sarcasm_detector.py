import pickle
import numpy as np
import os

class SarcasmDetector:
    """
    A class for detecting sarcasm in text using a pre-trained model.
    """
    
    def __init__(self, model_path='models/sarcasm_model.pkl', vectorizer_path='models/vectorizer.pkl'):
        """
        Initialize the SarcasmDetector with pre-trained model and vectorizer.
        
        Args:
            model_path (str): Path to the trained model
            vectorizer_path (str): Path to the vectorizer
        """
        self.model_path = model_path
        self.vectorizer_path = vectorizer_path
        self.model = None
        self.vectorizer = None
        self.load_model_and_vectorizer()
    
    def load_model_and_vectorizer(self):
        """
        Load the pre-trained model and vectorizer from disk.
        """
        try:
            # Load the trained model
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            print("Model loaded successfully!")
            
            # Load the vectorizer
            with open(self.vectorizer_path, 'rb') as f:
                self.vectorizer = pickle.load(f)
            print("Vectorizer loaded successfully!")
            
        except FileNotFoundError as e:
            print(f"Error: {e}")
            print("Please make sure you have trained the model first by running model_training.py")
            return False
        
        return True
    
    def predict(self, text):
        """
        Predict whether the given text is sarcastic or not.
        
        Args:
            text (str): Input text to analyze
        
        Returns:
            str: Prediction result ('Sarcasm' or 'Not Sarcasm')
        """
        if self.model is None or self.vectorizer is None:
            return "Error: Model or vectorizer not loaded properly."
        
        # Transform the input text using the fitted vectorizer
        text_vectorized = self.vectorizer.transform([text]).toarray()
        
        # Make prediction
        prediction = self.model.predict(text_vectorized)
        
        return prediction[0]
    
    def predict_with_probability(self, text):
        """
        Predict with probability scores for both classes.
        
        Args:
            text (str): Input text to analyze
        
        Returns:
            tuple: (prediction, probability_dict)
        """
        if self.model is None or self.vectorizer is None:
            return "Error: Model or vectorizer not loaded properly.", {}
        
        # Transform the input text
        text_vectorized = self.vectorizer.transform([text]).toarray()
        
        # Make prediction
        prediction = self.model.predict(text_vectorized)[0]
        
        # Get probability scores
        probabilities = self.model.predict_proba(text_vectorized)[0]
        classes = self.model.classes_
        
        prob_dict = {classes[i]: probabilities[i] for i in range(len(classes))}
        
        return prediction, prob_dict

def interactive_mode():
    """
    Run the sarcasm detector in interactive mode.
    """
    print("=== Sarcasm Detection Tool ===")
    print("Enter text to check for sarcasm, or 'quit' to exit.\n")
    
    # Initialize detector
    detector = SarcasmDetector()
    
    if detector.model is None or detector.vectorizer is None:
        print("Failed to load model. Please train the model first.")
        return
    
    while True:
        try:
            user_input = input("Enter text: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not user_input:
                print("Please enter some text.")
                continue
            
            # Get prediction with probabilities
            prediction, probabilities = detector.predict_with_probability(user_input)
            
            print(f"\nPrediction: {prediction}")
            print("Confidence scores:")
            for class_name, prob in probabilities.items():
                print(f"  {class_name}: {prob:.4f}")
            print("-" * 50)
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"An error occurred: {e}")

def main():
    """
    Main function with example usage.
    """
    # Check if model files exist
    if not os.path.exists('models/sarcasm_model.pkl') or not os.path.exists('models/vectorizer.pkl'):
        print("Error: Model files not found!")
        print("Please run model_training.py first to train the model.")
        return
    
    # Example usage
    detector = SarcasmDetector()
    
    if detector.model is None or detector.vectorizer is None:
        return
    
    # Test examples
    test_sentences = [
        "I am busy right now, can I ignore you some other time?",
        "The weather is really nice today!",
        "Oh great, another meeting that could have been an email",
        "I love spending my weekends doing laundry",
        "The new policy will definitely make everyone happy"
    ]
    
    print("=== Testing Sarcasm Detection ===\n")
    
    for sentence in test_sentences:
        prediction, probabilities = detector.predict_with_probability(sentence)
        print(f"Text: '{sentence}'")
        print(f"Prediction: {prediction}")
        print(f"Confidence: {max(probabilities.values()):.4f}")
        print("-" * 80)
    
    print("\nWould you like to try interactive mode? (y/n)")
    choice = input().strip().lower()
    
    if choice in ['y', 'yes']:
        interactive_mode()

if __name__ == "__main__":
    main()
