# We will be implementing the Convolutional Layer of a Convolutional Newural Network (CNN) from scratch including both forward and backward propagation.

import numpy as np
import matplotlib.pyplot as plt
import h5py
np.random.seed(42)

def zeroPad(X, pad):
    X_pad = np.pad(X, ((0, 0), (pad, pad), (pad, pad), (0, 0)), 'constant', constant_values=(0, 0))
    return X_pad

def convSingle(a_slice_prev, W, b):
    s = np.multiply(a_slice_prev, W)
    Z = np.sum(s)
    b = np.squeeze(b)
    Z += b
    return Z

def convForward(A_prev, W, b, hparameters):
    (m, n_C_prev, n_H_prev, n_W_prev) = A_prev.shape
    (f, f, n_C_prev, n_C) = W.shape
    stride = hparameters['stride']
    pad = hparameters['pad']
    
    n_H = int((n_H_prev + 2*pad - f)/stride) + 1
    n_W = int((n_W_prev + 2*pad - f)/stride) + 1
    
    Z = np.zeros((m, n_H, n_W, n_C))
    A_prev_pad = zeroPad(A_prev, pad)
    
    for i in range(m):
        a_prev_pad = A_prev_pad[i]
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    vert_start = h * stride
                    vert_end = vert_start + f
                    horiz_start = w * stride
                    horiz_end = horiz_start + f
                    
                    a_slice_prev = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]
                    Z[i, h, w, c] = convSingle(a_slice_prev, W[:, :, :, c], b[:, :, :, c])
                    
    cache = (A_prev, W, b, hparameters)
    return Z, cache

def convBackward(dZ, cache):
    (A_prev, W, b, haparameters) = cache
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    (f, f, n_C_prev, n_C) = W.shape
    (m, n_H, n_W, n_C) = dZ.shape
    stride = haparameters['stride']
    pad = haparameters['pad']
    
    dA_prev = np.zeros(A_prev.shape)
    dW = np.zeros(W.shape)
    db = np.zeros(b.shape)
    
    A_prev_pad = zeroPad(A_prev, pad)
    dA_prev_pad = zeroPad(dA_prev, pad)
    
    for i in range(m):
        a_prev_pad = A_prev_pad[i]
        da_prev_pad = dA_prev_pad[i]
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    vert_start = h * stride
                    vert_end = vert_start + f
                    horiz_start = w * stride
                    horiz_end = horiz_start + f
                    
                    a_slice = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]
                    
                    da_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :] += W[:, :, :, c] * dZ[i, h, w, c]
                    dW[:, :, :, c] += a_slice * dZ[i, h, w, c]
                    db[:, :, :, c] += dZ[i, h, w, c]
                    
        if pad != 0:
            dA_prev[i, :, :, :] = da_prev_pad[pad:-pad, pad:-pad, :]
        else:
            dA_prev[i, :, :, :] = da_prev_pad

        
    return dA_prev, dW, db

def test():
    np.random.seed(42)
    A_prev = np.random.randn(2, 4, 4, 3)
    W = np.random.randn(2, 2, 3, 2)
    b = np.random.randn(1, 1, 1, 2)
    hparameters = {"stride": 1, "pad": 0}
    
    Z, cache = convForward(A_prev, W, b, hparameters)
    print("Z =", Z)
    
    dZ = np.random.randn(2, 3, 3, 2)
    dA_prev, dW, db = convBackward(dZ, cache)
    print("dA_prev =", dA_prev)
    print("dW =", dW)
    print("db =", db)
    
if __name__ == "__main__":
    test()
