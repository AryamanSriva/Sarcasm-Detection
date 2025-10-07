# Sarcasm Detection

This project detects sarcasm in headlines using the Bernoulli Naive Bayes algorithm.

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
