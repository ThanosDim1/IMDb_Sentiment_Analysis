import numpy as np
import tensorflow as tf
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import mutual_info_classif
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

(x_train_imdb, y_train_imdb), (x_test_imdb, y_test_imdb) = tf.keras.datasets.imdb.load_data(path="imdb.npz",seed=123, num_words=10000, skip_top=100)

word_index = tf.keras.datasets.imdb.get_word_index(path="imdb_word_index.json")
index2word = {i + 3: word for word, i in word_index.items()}
index2word[0] = '[pad]'
index2word[1] = '[bos]'
index2word[2] = '[oov]'

# Decode sequences
x_train_imdb = [' '.join(index2word.get(idx, '[oov]') for idx in text) for text in x_train_imdb]
x_test_imdb = [' '.join(index2word.get(idx, '[oov]') for idx in text) for text in x_test_imdb]

# Initialize CountVectorizer with binary=True for presence/absence
vectorizer = CountVectorizer(binary=True)

# Fit the vectorizer on training data and transform both training and test data
x_train_binary = vectorizer.fit_transform(x_train_imdb)
x_test_binary = vectorizer.transform(x_test_imdb)

information_gain = mutual_info_classif(x_train_binary,y_train_imdb)

m = 1000
top_m = np.argsort(information_gain)[-m:]

x_train_final = x_train_binary[:, top_m]
x_test_final = x_test_binary[:, top_m]

x_train_final = x_train_final.toarray()
x_test_final = x_test_final.toarray()

# Εφαρμογή Bernoulli Naive Bayes
class BernoulliNaiveBayes:
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        self.class_priors = np.zeros(len(self.classes))
        self.feature_probs = np.zeros((len(self.classes), n_features))

        for idx, c in enumerate(self.classes):
            X_c = X[y == c]
            self.class_priors[idx] = X_c.shape[0] / n_samples
            self.feature_probs[idx] = (X_c.sum(axis=0) + 1) / (X_c.shape[0] + 2)

    def predict(self, X):
        log_probs = []
        for x in X:
            log_class_probs = np.log(self.class_priors)
            log_feature_probs = np.sum(x * np.log(self.feature_probs) + (1 - x) * np.log(1 - self.feature_probs),axis=1)
            log_probs.append(log_class_probs + log_feature_probs)
        return np.argmax(log_probs, axis=1)
    
model = BernoulliNaiveBayes()
model.fit(x_train_final, y_train_imdb)

y_pred = model.predict(x_test_final)

# Evaluate the model
accuracy = accuracy_score(y_test_imdb, y_pred)
print(f"Accuracy: {accuracy}")

