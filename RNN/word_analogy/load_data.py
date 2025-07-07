import numpy as np

def load_glove_vecs(file_path):
    with open(file_path, 'r') as f:
        words = set()
        word_to_vec = {}
        for line in f:
            line = line.strip().split()
            word = line[0]
            vec = np.array(line[1:], dtype=np.float64)
            word_to_vec[word] = vec
            words.add(word)
    return words, word_to_vec