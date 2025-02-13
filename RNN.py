import numpy as np
import torch
# from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.feature_extraction.text import CountVectorizer
# from torchtext.data.utils import get_tokenizer
from tensorflow.keras.datasets import imdb
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_20newsgroups
# import gensim.downloader as api
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import string


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)


# Parameters
m = 1000  # Number of words in vocabulary
n = 20   # N most frequent words to skipdevice = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)
k = 0     # K least frequent words to skip

# Load IMDB dataset
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=m-k, skip_top=n)
word_index = imdb.get_word_index()

# Create index-to-word mapping
index2word = {i + 3: word for word, i in word_index.items()}
index2word[0] = '[pad]'
index2word[1] = '[bos]'
index2word[2] = '[oov]'

# Convert tokenized sequences back to text
x_train = [' '.join([index2word.get(idx, '[oov]') for idx in text]) for text in x_train]
x_test = [' '.join([index2word.get(idx, '[oov]') for idx in text]) for text in x_test]

# Split train set further into train/validation sets
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

# Create custom vocabulary using CountVectorizer
vectorizer = CountVectorizer(max_features=m, binary=True)
vectorizer.fit(X_train)
custom_vocab = vectorizer.vocabulary_

# Ensure special tokens are in the vocabulary
custom_vocab['PAD'] = len(custom_vocab)
custom_vocab['UNK'] = len(custom_vocab)

# Compute average sequence length
avg_length = int(np.mean([len(re.sub(r'[^a-zA-Z]', ' ', text.lower()).split()) for text in X_train]))

# Convert text data into binary bag-of-words representation
X_train_binary = torch.tensor(vectorizer.transform(X_train).toarray(), dtype=torch.float32)
X_val_binary = torch.tensor(vectorizer.transform(X_val).toarray(), dtype=torch.float32)
X_test_binary = torch.tensor(vectorizer.transform(x_test).toarray(), dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_val = torch.tensor(y_val, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

# Create DataLoader for PyTorch
train_dataset = TensorDataset(X_train_binary, y_train)
val_dataset = TensorDataset(X_val_binary, y_val)
test_dataset = TensorDataset(X_test_binary, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Print dataset info
print(f'Training samples: {len(train_dataset)}')
print(f'Validation samples: {len(val_dataset)}')
print(f'Test samples: {len(test_dataset)}')
print(f'Vocabulary size: {len(custom_vocab)}')



# Dataset class
class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_length):
        self.texts = [self.tokenize(text, vocab, max_length) for text in texts]
        self.labels = labels

    def tokenize(self, text, vocab, max_length):
        text = re.sub(r'[^a-zA-Z]', ' ', text.lower()).split()
        tokens = [vocab.get(word, vocab['UNK']) for word in text]
        if len(tokens) < max_length:
            tokens += [vocab['PAD']] * (max_length - len(tokens))
        else:
            tokens = tokens[:max_length]
        return tokens

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(self.texts[idx]), torch.tensor(self.labels[idx])

train_dataset = TextDataset(X_train, y_train, custom_vocab, avg_length)
val_dataset = TextDataset(X_val, y_val, custom_vocab, avg_length)
test_dataset = TextDataset(x_test, y_test, custom_vocab, avg_length)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)


# Define RNN, GRU, and LSTM models with optional Global Max Pooling
class RNNModel(nn.Module):
    def __init__(self, vocab_size,
                 embed_dim, hidden_dim, output_dim,
                 model_type='RNN',
                 pretrained=True, freeze=False, use_pooling=False):
        super(RNNModel, self).__init__()
        self.use_pooling = use_pooling
        if pretrained:
            self.embedding = nn.Embedding(vocab_size, embed_dim)  # Initialize embeddings randomly
        else:
            self.embedding = nn.Embedding(vocab_size, embed_dim)
        rnn_class = {'RNN': nn.RNN, 'GRU': nn.GRU, 'LSTM': nn.LSTM}[model_type]
        self.rnn = rnn_class(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.rnn(embedded)
        if self.use_pooling:
            pooled = torch.max(output, dim=1)[0]
            return torch.sigmoid(self.fc(pooled))
        else:
            return torch.sigmoid(self.fc(output[:, -1, :]))
        

        # Training and Evaluation Functions remain unchanged

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=10):
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for texts, labels in train_loader:
            optimizer.zero_grad()
            texts = texts.to(device)
            labels = labels.float().to(device)
            outputs = model(texts).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_losses.append(train_loss / len(train_loader))

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for texts, labels in val_loader:
                texts = texts.to(device)
                labels = labels.float().to(device)
                outputs = model(texts).squeeze()
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        val_losses.append(val_loss / len(val_loader))

        if epoch % 2 == 0:
            print(f'Epoch: {epoch:4.0f} / {epochs} | Training Loss: {train_loss:.5f}, Validation Loss: {val_loss:.5f}')

    return train_losses, val_losses

# Evaluation function
def evaluate_model(model, test_loader):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for texts, labels in test_loader:
            texts = texts.to(device)
            preds = model(texts).squeeze() > 0.5
            y_true.extend(labels.tolist())
            y_pred.extend(preds.tolist())
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return accuracy, precision, recall, f1


# Instantiate models and train
models = {
    'RNN': RNNModel(len(custom_vocab), 300, 32, 1, 'RNN'),
    'GRU': RNNModel(len(custom_vocab), 300, 32, 1, 'GRU', use_pooling=True),
    'LSTM': RNNModel(len(custom_vocab), 300, 32, 1, 'LSTM', use_pooling=True)
    }
    

results = {}
epochs = 10

for name, model in models.items():
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    print(f"Training {name}...")
    train_losses, val_losses = train_model(model.float().to(device), train_loader, val_loader,
                                           criterion, optimizer, epochs=epochs)
    results[name] = {'train_loss': train_losses, 'val_loss': val_losses}
    acc, prec, rec, f1 = evaluate_model(model, test_loader)
    results[name].update({'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1})

# Plot losses
for name in models:
    plt.plot(results[name]['train_loss'], linestyle='--', label=f'{name} Train')
    plt.plot(results[name]['val_loss'], label=f'{name} Val')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Train/Validation Loss Comparison')
plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
plt.show()

# Print evaluation metrics
for name, metrics in results.items():
    print(f"{name}: Accuracy={metrics['accuracy']:.4f}, Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}, F1={metrics['f1']:.4f}")