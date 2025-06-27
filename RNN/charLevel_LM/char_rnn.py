# Minimal character level RNN which tries to learn the pattern of names from names.txt

import numpy as np

data = open('names.txt','r').read()
vocab = list(set(data))
dataSize, vocabSize = len(data), len(vocab)
print(f"We have {dataSize} characters and {vocabSize} unique characters.")

char_to_idx = {ch:i for i, ch in enumerate(vocab)}
idx_to_char = {i:ch for i, ch in enumerate(vocab)}

hidden_size = 100
seq_length = 30
lr = 0.1

Wax = np.random.randn(hidden_size, vocabSize) * 0.01
Waa = np.random.randn(hidden_size, hidden_size) * 0.01
Wya = np.random.randn(vocabSize, hidden_size) * 0.01
ba = np.zeros((hidden_size, 1))
by = np.zeros((vocabSize, 1))

def compute_loss(inputs, targets, aprev):
    x, a, y, p = {}, {}, {}, {}
    a[-1] = np.copy(aprev)
    loss = 0
    
    for t in range(len(inputs)):
        x[t] = np.zeros((vocabSize, 1))
        x[t][inputs[t]] = 1
        a[t] = np.tanh(np.dot(Wax, x[t]) + np.dot(Waa, a[t-1]) + ba)
        y[t] = np.dot(Wya, a[t]) + by
        p[t] = np.exp(y[t]) / np.sum(np.exp(y[t]))
        loss += -np.log(p[t][targets[t], 0])
        
    dWax, dWaa, dWya = np.zeros_like(Wax), np.zeros_like(Waa), np.zeros_like(Wya)
    dba, dby = np.zeros_like(ba), np.zeros_like(by)
    danext = np.zeros_like(a[0])
    for t in reversed(range(len(inputs))):
        dy = np.copy(p[t])
        dy[targets[t]] -= 1
        dWya += np.dot(dy, a[t].T)
        dby += dy
        da = danext + np.dot(Wya.T, dy)
        da_raw = (1 - a[t]**2) * da
        dba += da_raw
        dWax += np.dot(da_raw, x[t].T)
        dWaa += np.dot(da_raw, a[t-1].T)
        danext = np.dot(Waa.T, da_raw)
    for dparam in [dWax, dWaa, dWya, dba, dby]:
        np.clip(dparam, -5, 5, out=dparam)
    return loss, dWax, dWaa, dWya, dba, dby, a[len(inputs)-1]

def sample(a, seed_idx, n, temperature=1.0):
    x = np.zeros((vocabSize, 1))
    x[seed_idx] = 1
    indices = []
    for t in range(n):
        a = np.tanh(np.dot(Wax, x) + np.dot(Waa, a) + ba)
        y = np.dot(Wya, a) + by
        p = np.exp(y / temperature) / np.sum(np.exp(y / temperature))
        idx = np.random.choice(range(vocabSize), p = p.ravel())
        x = np.zeros((vocabSize, 1))
        x[idx] = 1
        indices.append(idx)
    return indices

n, p = 0, 0
mWax, mWaa, mWya = np.zeros_like(Wax), np.zeros_like(Waa), np.zeros_like(Wya)
mba, mby = np.zeros_like(ba), np.zeros_like(by)
smooth_loss = -np.log(1.0/vocabSize) * seq_length

while n<5000:
    if p+seq_length+1 >= dataSize or n == 0:
        aprev = np.zeros((hidden_size, 1))
        p = 0
    inputs = [char_to_idx[ch] for ch in data[p:p+seq_length]]
    targets = [char_to_idx[ch] for ch in data[p+1:p+seq_length+1]]
    
    if n%100 == 0:
        sample_idx = sample(aprev, inputs[0], 30)
        txt = ''.join(idx_to_char[idx] for idx in sample_idx)
        print(f"----\n{txt}\n----")
        
    loss, dWax, dWaa, dWya, dba, dby, aprev = compute_loss(inputs, targets, aprev)
    smooth_loss = 0.999 * smooth_loss + 0.001 * loss
    
    if n%100 == 0:
        print(f"iter {n}, loss: {smooth_loss:.4f}")
        
    for param, dparam, mem in zip([Wax, Waa, Wya, ba, by], [dWax, dWaa, dWya, dba, dby], [mWax, mWaa, mWya, mba, mby]):
        mem += dparam * dparam
        param -= lr * dparam / np.sqrt(mem + 1e-8)
        
    p += seq_length
    n += 1
    
print("Training done!")
a = np.zeros((hidden_size, 1))
seed_idx = np.random.choice(range(vocabSize))
sample_idx = sample(a, seed_idx, 30, temperature=0.8)
txt = ''.join(idx_to_char[idx] for idx in sample_idx)
print(f"Sampled text:\n{txt}")