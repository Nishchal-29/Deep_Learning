import numpy as np

def read_glove(file_path):
    with open(file_path, 'r') as f:
        words = set()
        word_to_vec = {}
        for line in f:
            line = line.strip().split()
            word = line[0]
            vec = np.array(line[1:], dtype=np.float64)
            words.add(word)
            word_to_vec[word] = vec
            
        i = 1
        words_to_idx = {}
        idx_to_words = {}
        for w in sorted(words):
            words_to_idx[w] = i
            idx_to_words[i] = w
            i += 1
    return words_to_idx, idx_to_words, word_to_vec


def softmax(Z):
    e_Z = np.exp(Z - np.max(Z))
    return e_Z / e_Z.sum(axis=0)