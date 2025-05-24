# This script implements a simple convolutional neural network (CNN) on random (4X4X3) input images based on this architecture:
# Convolution → ReLU → Flatten → Fully Connected → Softmax

import numpy as np
import matplotlib.pyplot as plt
import h5py
from convLayer import convForward, convBackward
np.random.seed(42)

def relu(Z):
    return np.maximum(0, Z)

def reluBackward(dA, Z):
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ

def softmax(Z):
    expZ = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    return expZ / np.sum(expZ, axis=1, keepdims=True)

def crossEntropyLoss(Y, Y_hat):
    m = Y.shape[0]
    log_likelihood = -np.log(Y_hat[range(m), Y])
    loss = np.sum(log_likelihood) / m
    return loss

def randomDataGenerator(m):
    X = np.random.randn(m, 4, 4, 3)
    sum = np.sum(X, axis=(1, 2, 3), keepdims=True)
    y = (sum > 0).astype(int).flatten()
    return X, y

def cnnClassifier(alpha = 0.01, epochs = 500, num_samples = 1000):
    X, y = randomDataGenerator(num_samples)
    W = np.random.randn(2, 2, 3, 1)*0.01
    b = np.zeros((1, 1, 1, 1))
    hparameters = {"stride": 1, "pad": 0}
    W_fc = np.random.randn(9, 2)*0.01
    b_fc = np.zeros((1, 2))
    
    losses = []
    for i in range(epochs):
        #Forward pass
        Z, cache_conv = convForward(X, W, b, hparameters)
        A = relu(Z)
        A_flat = A.reshape(A.shape[0], -1)
        logits = np.dot(A_flat, W_fc) + b_fc
        Y_hat = softmax(logits)
        
        #Compute loss
        loss = crossEntropyLoss(y, Y_hat)
        losses.append(loss)
        
        #Backward pass
        d_logits = Y_hat.copy()
        d_logits[range(num_samples), y] -= 1
        d_logits /= num_samples
        dW_fc = np.dot(A_flat.T, d_logits)
        db_fc = np.sum(d_logits, axis=0, keepdims=True)
        dA_flat = np.dot(d_logits, W_fc.T)
        dA = dA_flat.reshape(A.shape)
        dZ = reluBackward(dA, Z)
        dA_prev, dW, db = convBackward(dZ, cache_conv)
        
        #update parameters
        W -= alpha * dW
        b -= alpha * db
        W_fc -= alpha * dW_fc
        b_fc -= alpha * db_fc
        
        if i % 50 == 0:
            print(f"Epoch {i}, Loss: {loss:.4f}")
            
    Z, cache_conv = convForward(X, W, b, hparameters)
    A = relu(Z)
    A_flat = A.reshape(A.shape[0], -1)
    preds = np.argmax(softmax(np.dot(A_flat, W_fc) + b_fc), axis=1)
    
    return preds, y, losses

y_pred, y_true, losses = cnnClassifier()

#plotting the loss
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(losses, label="Loss")
plt.xlabel("Epoch")
plt.ylabel("Cross-Entropy Loss")
plt.title("Training Loss Over Epochs")
plt.legend()

plt.subplot(1, 2, 2)
acc = np.mean(y_pred == y_true)
plt.bar(["Accuracy"], [acc], color="green")
plt.title("Final Classification Accuracy")
plt.ylim(0, 1)

plt.tight_layout()
plt.show()