# Sarcasm Detection

This project implements a machine learning model to detect sarcasm in headlines using the Bernoulli Naive Bayes algorithm.

## Overview

The Sarcasm Detection project uses a dataset of headlines labeled as sarcastic or not sarcastic to train a model that can predict whether new headlines contain sarcasm. It utilizes the following technologies:

- Python
- pandas
- numpy
- scikit-learn
- Bernoulli Naive Bayes algorithm

## Features

- Data loading and preprocessing
- Text vectorization using CountVectorizer
- Model training using Bernoulli Naive Bayes
- Prediction of sarcasm in new text inputs

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/AryamanSriva/sarcasm-detection.git
   ```

2. Install the required dependencies:
   ```
   pip install pandas numpy scikit-learn
   ```

3. Download the `Sarcasm.json` dataset and place it in the project directory.

## Usage

1. Run the Jupyter notebook `Sarcasm Detection.ipynb`.
2. The notebook will guide you through the process of:
   - Loading and preprocessing the data
   - Training the model
   - Testing the model's accuracy
   - Using the model to predict sarcasm in new text inputs

## Model Performance

The current model achieves an accuracy of approximately 84.48% on the test set.

## Example

After training the model, you can input your own text to test for sarcasm:

```python
user = input("Enter a text: ")
data = cv.transform([user]).toarray()
output = model.predict(data)
print(output)
```

## Contributing

Contributions to improve the model or extend the project are welcome. Please feel free to submit a pull request or open an issue.


## Acknowledgments

- The dataset used in this project is sourced from Kaggle.
