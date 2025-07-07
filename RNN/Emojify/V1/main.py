import numpy as np
import emoji
from embeddings import read_glove, softmax
from load_data import read_csv, one_hot
from get_emoji import label_to_emoji, predict

X_train, y_train = read_csv('/home/devcontainers/Deep_Learning/datasets/train_emoji.csv')
X_test, y_test = read_csv('/home/devcontainers/Deep_Learning/datasets/test_emoji.csv')
maxLen = len(max(X_train, key=len).split())

# Checking some training examples with labels as emojis
for idx in range(10):
    print(X_train[idx], label_to_emoji(y_train[idx]))
    
y_oh_train = one_hot(y_train, 5)
y_oh_test = one_hot(y_test, 5)
word_to_idx, idx_to_words, word_to_vec = read_glove('/home/devcontainers/Deep_Learning/RNN/Emojify/V1/glove.6B.100d.txt')

def get_avg_vec(sentence, word_to_vec):
    any_word = list(word_to_vec.keys())[0]
    words = sentence.lower().split()
    cnt = 0
    avg = np.zeros(word_to_vec[any_word].shape)
    for word in words:
        if word in word_to_vec:
            avg += word_to_vec[word]
            cnt += 1
    if cnt > 0:
        avg /= cnt
        
    return avg

def model(X, y, word_to_vec, lr = 0.01, iters = 100):
    any_word = list(word_to_vec.keys())[0]
    m = y.shape[0]
    n_h = word_to_vec[any_word].shape[0]
    n_classes = len(np.unique(y))
    W = np.random.randn(n_classes, n_h) / np.sqrt(n_h)
    b = np.zeros((n_classes,))
    y_oh = one_hot(y, n_classes)
    for i in range(iters):
        loss = 0
        for j in range(m):
            avg = get_avg_vec(X[j], word_to_vec)
            z = np.dot(W, avg) + b
            a = softmax(z)
            loss += -np.sum(np.dot(y_oh[j], np.log(a)))
            dz = a - y_oh[j]
            dW = np.dot(dz.reshape(n_classes,1), avg.reshape(1, n_h))
            db = dz
            W -= lr*dW
            b -= lr*db
        loss /= m
        if i%5==0:
            print("Epoch: " + str(i) + " --- cost = " + str(loss))
            pred = predict(X, y, W, b, word_to_vec)

    return pred, W, b

predictions, W, b = model(X_train, y_train, word_to_vec)
for i in range(10):
    pred = predictions[i].item()
    print(f"{X_train[i]} -> {label_to_emoji(int(pred))}")
print("Training set:")
pred_train = predict(X_train, y_train, W, b, word_to_vec)
print('Test set:')
pred_test = predict(X_test, y_test, W, b, word_to_vec)