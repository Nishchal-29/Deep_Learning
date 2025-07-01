import numpy as np
from utils import softmax

def forward_cell(X, parameters, a_prev):
    Waa = parameters['Waa']
    Wax = parameters['Wax']
    Wya = parameters['Wya']
    ba = parameters['ba']
    by = parameters['by']
    
    a_next = np.tanh(np.dot(Waa, a_prev) + np.dot(Wax, X) + ba)
    p_t = softmax(np.dot(Wya, a_next) + by)
    
    return a_next, p_t

def forward(X, Y, a0, parameters, vocab_size = 27):
    x, a, p = {}, {}, {}
    a[-1] = np.copy(a0)
    loss = 0
    for t in range(len(X)):
        x[t] = np.zeros((vocab_size, 1))
        if X[t] != None:
            x[t][X[t]] = 1
        a[t], p[t] = forward_cell(x[t], parameters, a[t-1])
        loss -= np.log(p[t][Y[t], 0])
    cache = (x, a, p)
    return loss, cache