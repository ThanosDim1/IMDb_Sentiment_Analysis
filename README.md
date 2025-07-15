# IMDb Sentiment Analysis

This project explores sentiment analysis on the IMDb movie reviews dataset using three different machine learning approaches:

- **Logistic Regression** (custom implementation)
- **Naive Bayes** (custom Bernoulli Naive Bayes and scikit-learn comparison)
- **Recurrent Neural Networks (RNNs)** (including RNN, GRU, and LSTM with global max pooling)

## Dataset

All models use the [IMDb dataset](https://ai.stanford.edu/~amaas/data/sentiment/) available via TensorFlow/Keras. The dataset consists of 50,000 movie reviews labeled as positive or negative. Preprocessing includes:
- Limiting vocabulary size (top N most frequent words, skip top M, etc.)
- Mapping numeric sequences to human-readable text
- Splitting into training, validation, and test sets

## Project Structure

```
IMDb_Sentiment_Analysis/
├── Logistic Regression/
│   └── logistic_regression.ipynb
├── Naive Bayes/
│   └── Naive_Bayes.ipynb
├── RNN/
│   ├── RNN.py
│   ├── RNN.ipynb
│   ├── RNN_output.txt
│   └── RNN_Curves.png
```

### 1. Logistic Regression
- **File:** `Logistic Regression/logistic_regression.ipynb`
- **Approach:** Implements logistic regression from scratch using stochastic gradient descent (SGD) with L2 regularization.
- **Features:**
  - Binary bag-of-words representation using `CountVectorizer`.
  - Hyperparameter tuning for regularization (lambda) and classification threshold.
  - Evaluation using accuracy, precision, recall, F1-score, and learning curves.
  - Comparison with scikit-learn's logistic regression and Bernoulli Naive Bayes.

### 2. Naive Bayes
- **File:** `Naive Bayes/Naive_Bayes.ipynb`
- **Approach:** Custom implementation of Bernoulli Naive Bayes for binary sentiment classification.
- **Features:**
  - Binary bag-of-words features.
  - Manual calculation of log-probabilities with Laplace smoothing.
  - Evaluation using precision, recall, F1-score, and confusion matrix.
  - Comparison with scikit-learn's BernoulliNB, logistic regression, and AdaBoost.
  - Visualization of learning curves and metric tables.

### 3. Recurrent Neural Networks (RNN)
- **Files:** `RNN/RNN.py`, `RNN/RNN.ipynb`, `RNN_output.txt`, `RNN_Curves.png`
- **Approach:** Deep learning models for sequence data, including:
  - Vanilla RNN
  - GRU (Gated Recurrent Unit)
  - LSTM (Long Short-Term Memory)
  - All with global max pooling and bidirectional layers
- **Features:**
  - Custom PyTorch dataset for tokenized and padded sequences
  - Training and validation with loss curves
  - Evaluation using accuracy, precision, recall, F1-score
  - Comparison with Naive Bayes on the same data
  - Output and plots saved for further analysis

## How to Run

1. **Install dependencies:**
   - Python 3.8+
   - numpy, pandas, matplotlib, scikit-learn, torch, tensorflow
2. **Open the notebooks** in Jupyter or your preferred environment and run the cells in order.
3. **For RNN experiments:** You can also run `RNN.py` directly for scripted training and evaluation.

## Results Summary

- **Logistic Regression:**
  - Achieves ~87% accuracy on the test set after hyperparameter tuning.
- **Naive Bayes:**
  - Custom Bernoulli Naive Bayes achieves ~84% accuracy.
- **RNN/GRU/LSTM:**
  - All deep models achieve ~83-84% accuracy, with GRU and LSTM slightly outperforming vanilla RNN.



## Creators

This project was created by Athanasios Zois Dimitrakopoulos and Andreas Lampos, students of Athens University of Economics and Business (AUEB), as part of the requirements for the course "Artificial Intelligence."
