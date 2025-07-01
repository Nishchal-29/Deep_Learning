import numpy as np
from utils import softmax
from forward_pass import forward
from backward_pass import backward
from sample_seq import sample
from params import initialize_parameters, update_parameters, clip_gradients

data = open('dinos.txt', 'r').read()
data = data.lower()
vocab = list(set(data))
data_size, vocab_size = len(data), len(vocab)
print(f"We have {data_size} characters and {vocab_size} unique characters.")

char_to_idx = {ch: i for i, ch in enumerate(vocab)}
idx_to_char = {i: ch for i, ch in enumerate(vocab)}

def optimize(X, Y, a_prev, parameters, lr = 0.01):
    loss, cache = forward(X, Y, a_prev, parameters)
    gradients, a = backward(X, Y, cache, parameters)
    gradients = clip_gradients(gradients)
    parameters = update_parameters(parameters, gradients, lr)
    
    return loss, gradients, a[len(X)-1]

def train(data_x, idx_to_char, char_to_idx, iters = 40000, n_a = 50, dino_names = 7, vocab_size = 27):
    parameters = initialize_parameters(vocab_size, n_a)
    loss = -np.log(1.0 / vocab_size) * dino_names
    a_prev = np.zeros((n_a, 1))
    examples = [x.strip() for x in data_x]
    np.random.shuffle(examples)
    
    for i in range(iters):
        idx = i % len(examples)
        example_chars = examples[idx]
        examples_idx = [char_to_idx[c] for c in example_chars]
        X = [char_to_idx['\n']] + examples_idx
        Y =X[1:]
        Y = Y + [char_to_idx['\n']]
        curr_loss, gradients, a_prev = optimize(X, Y, a_prev, parameters)
        loss = loss * 0.999 + curr_loss * 0.001
        
        # if i % 1000 == 0:
        #     print(f"Iteration {i}, Loss: {loss}")
        #     for j in range(dino_names):
        #         sampled_indices = sample(parameters, char_to_idx)
        #         sampled_chars = [idx_to_char[idx] for idx in sampled_indices]
        #         print(f"Sample {j+1}: {''.join(sampled_chars)}")
        #     print("-" * 20)
        
        if i % 1000 == 0:
            print(f"Iteration {i}, Loss: {loss:.4f}")
            sampled_indices = sample(parameters, char_to_idx)
            sampled_chars = [idx_to_char[idx] for idx in sampled_indices]
            print(f"Sampled sequence: {''.join(sampled_chars)}")
            print("-" * 20)
        
    return parameters

if __name__ == "__main__":
    parameters = train(data.split("\n"), idx_to_char, char_to_idx)
    print("Training complete.")
    
    sampled_indices = sample(parameters, char_to_idx)
    sampled_chars = [idx_to_char[idx] for idx in sampled_indices]
    print(f"Sampled sequence: {''.join(sampled_chars)}")