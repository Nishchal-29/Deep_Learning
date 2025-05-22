# We will be implementing the pooling layer(both max pool and average pool) of a Convolutional Newural Network (CNN) from scratch including both forward and backward propagation.

import numpy as np
import matplotlib.pyplot as plt
import h5py
np.random.seed(42)

def poolForward(A_prev, hparameters, mode):
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    f = hparameters["f"]
    s = hparameters["stride"]
    n_H = int((n_H_prev - f) / s) + 1
    n_W = int((n_W_prev - f) / s) + 1
    n_C = n_C_prev
    
    A = np.zeros((m, n_H, n_W, n_C))
    
    for i in range(m):
        a_prev_slice = A_prev[i]
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    vert_start = h*s
                    vert_end = vert_start + f
                    horiz_start = w*s
                    horiz_end = horiz_start + f
                    
                    a_slice_prev = a_prev_slice[vert_start:vert_end,horiz_start:horiz_end,c]
                     
                    if mode == "max":
                        A[i, h, w, c] = np.max(a_slice_prev)
                    elif mode == "average":
                        A[i, h, w, c] = np.mean(a_slice_prev)
                        
    cache = (A_prev, hparameters)
    return A, cache

def createMask(x):
    mask = (x == np.max(x))
    return mask

def distribute(dZ, shape):
    (n_H, n_W) = shape
    average = np.prod(shape)
    a = np.ones(shape) * (dZ / average)
    return a

def poolBackward(dA, cache, mode):
    (A_prev, hparameters) = cache
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    (m, n_H, n_W, n_C) = dA.shape
    f = hparameters["f"]
    s = hparameters["stride"]
    
    dA_prev = np.zeros(A_prev.shape)
    
    for i in range(m):
        a_prev = A_prev[i]
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    vert_start = h*s
                    vert_end = vert_start + f
                    horiz_start = w*s
                    horiz_end = horiz_start + f
                    
                    if mode == "max":
                        a_prev_slice = a_prev[vert_start:vert_end,horiz_start:horiz_end,c]
                        mask = createMask(a_prev_slice)
                        dA_prev[i, vert_start:vert_end, horiz_start:horiz_end, c] += mask * dA[i, h, w, c]
                        
                    elif mode == "average":
                        da = dA[i, h, w, c]
                        shape = (f, f)
                        dA_prev[i, vert_start:vert_end, horiz_start:horiz_end, c] += distribute(da, shape)
                        
    return dA_prev

def test():
    np.random.seed(42)
    A_prev = np.random.randn(2, 4, 4, 3)
    hparameters = {"f": 2, "stride": 2}
    
    A, cache = poolForward(A_prev, hparameters, mode="max")
    print("A =", A)
    
    dA = np.random.randn(2, 2, 2, 3)
    dA_prev = poolBackward(dA, cache, mode="max")
    print("dA_prev =", dA_prev)
    print("A_prev =", A_prev)
    print("dA =", dA)
    print("dA_prev.shape =", dA_prev.shape)
    print("A.shape =", A.shape)
    print("dA.shape =", dA.shape)
    print("A_prev.shape =", A_prev.shape)
    print("hparameters =", hparameters)
    
if __name__ == "__main__":
    test()              