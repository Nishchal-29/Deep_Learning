import numpy as np

def backward_cell(X, a, a_prev, dy, parameters, gradients):
    gradients['dWya'] += np.dot(dy, a.T)
    gradients['dby'] += dy
    
    da = np.dot(parameters['Wya'].T, dy) + gradients['da_next']
    da_raw = (1 - a**2) * da
    
    gradients['dWax'] += np.dot(da_raw, X.T)
    gradients['dba'] += da_raw
    gradients['dWaa'] += np.dot(da_raw, a_prev.T)
    gradients['da_next'] = np.dot(parameters['Waa'].T, da_raw)
    
    return gradients

def backward(X, Y, cache, parameters):
    gradients = {}
    (x, a, p) = cache
    
    Waa = parameters['Waa']
    Wax = parameters['Wax']
    Wya = parameters['Wya']
    ba = parameters['ba']
    by = parameters['by']
    
    gradients['dWaa'] = np.zeros_like(Waa)
    gradients['dWax'] = np.zeros_like(Wax)
    gradients['dWya'] = np.zeros_like(Wya)
    gradients['dba'] = np.zeros_like(ba)
    gradients['dby'] = np.zeros_like(by)
    gradients['da_next'] = np.zeros_like(a[0])
    
    for t in reversed(range(len(X))):
        dy = np.copy(p[t])
        dy[Y[t]] -= 1
        gradients = backward_cell(x[t], a[t], a[t-1], dy, parameters, gradients)
        
    return gradients, a