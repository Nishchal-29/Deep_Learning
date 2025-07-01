import numpy as np

def initialize_parameters(vocab_size, hidden_size):
    np.random.seed(42)  # For reproducibility
    
    Wax = np.random.randn(hidden_size, vocab_size) * 0.01
    Waa = np.random.randn(hidden_size, hidden_size) * 0.01
    Wya = np.random.randn(vocab_size, hidden_size) * 0.01
    ba = np.zeros((hidden_size, 1))
    by = np.zeros((vocab_size, 1))
    
    parameters = {'Wax': Wax, 'Waa': Waa, 'Wya': Wya, 'ba': ba, 'by': by}
    
    return parameters

def update_parameters(parameters, gradients, lr):
    parameters['Wax'] -= lr * gradients['dWax']
    parameters['Waa'] -= lr * gradients['dWaa']
    parameters['Wya'] -= lr * gradients['dWya']
    parameters['ba'] -= lr * gradients['dba']
    parameters['by'] -= lr * gradients['dby']
    
    return parameters

def clip_gradients(gradients, maxValue = 5):
    for key in gradients:
        np.clip(gradients[key], -maxValue, maxValue, out=gradients[key])
    return gradients