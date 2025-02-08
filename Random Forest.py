import numpy as np
from collections import Counter
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import mutual_info_classif
import tensorflow as tf

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

class RandomForest:
    def __init__(self, n_trees=150, max_depth=15, sample_size=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.sample_size = sample_size
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        n_samples = X.shape[0]
        self.sample_size = self.sample_size if self.sample_size else n_samples

        for _ in range(self.n_trees):
            tree = DecisionTreeClassifier(criterion='entropy', max_depth=self.max_depth)
            indices = np.random.choice(n_samples, self.sample_size, replace=True)
            X_sample, y_sample = X[indices], y[indices]
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(tree_preds, 0, 1)
        return np.array([Counter(preds).most_common(1)[0][0] for preds in tree_preds])

random_forest = RandomForest()
random_forest.fit(x_train_final, y_train_imdb)

predictions = random_forest.predict(x_test_final)

accuracy = accuracy_score(y_test_imdb, predictions)
print(f"Accuracy: {accuracy}")