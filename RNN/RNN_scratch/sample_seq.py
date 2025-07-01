import numpy as np
from utils import softmax

def sample(parameters, char_to_idx, seed=42, max_len=50):
    Waa = parameters['Waa']
    Wax = parameters['Wax']
    Wya = parameters['Wya']
    ba = parameters['ba']
    by = parameters['by']
    vocab_size = Wya.shape[0]
    hidden_size = Waa.shape[1]  

    x = np.zeros((vocab_size, 1))
    a_prev = np.zeros((hidden_size, 1))
    indices = []

    idx = char_to_idx['\n'] 
    counter = 0

    np.random.seed(seed)

    while (idx != char_to_idx['\n'] or counter == 0) and counter < max_len:
        x = np.zeros((vocab_size, 1))
        if counter == 0:
            idx = char_to_idx['\n'] 
        else:
            x[idx] = 1

        a_next = np.tanh(np.dot(Wax, x) + np.dot(Waa, a_prev) + ba)
        y = np.dot(Wya, a_next) + by
        p = softmax(y)

        idx = np.random.choice(range(vocab_size), p=p.ravel())
        indices.append(idx)

        a_prev = a_next
        counter += 1

    return indices
