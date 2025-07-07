import emoji
import numpy as np
from embeddings import softmax

emoji_dict = {"0": "\u2764\ufe0f", "1": ":baseball:", "2": ":smile:", "3": ":disappointed:", "4": ":fork_and_knife:"}

def label_to_emoji(label):
    return emoji.emojize(emoji_dict[str(label)], language='alias')

def predict(X, Y, W, b, word_to_vec):
    m = X.shape[0]
    Y_pred = np.zeros((m, 1))
    word = list(word_to_vec.keys())[0]
    n_h = word_to_vec[word].shape[0]
    for i in range(m):
        words = X[i].lower().split()
        avg = np.zeros((n_h,))
        cnt = 0
        for w in words:
            if w in word_to_vec:
                avg += word_to_vec[w]
                cnt += 1
                
        if cnt > 0:
            avg /= cnt
        Z = np.dot(W, avg) + b
        A = softmax(Z)
        Y_pred[i] = np.argmax(A)
        
    print("Accuracy: "  + str(np.mean(Y_pred == Y.reshape(Y.shape[0], 1))))
    return Y_pred
        