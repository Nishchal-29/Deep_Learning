import numpy as np
import csv

def read_csv(filename):
    phrase = []
    emoji = []
    with open(filename) as f:
        reader = csv.reader(f)
        for row in reader:
            phrase.append(row[0])
            emoji.append(row[1])
            
    X = np.asarray(phrase)
    Y = np.asarray(emoji, dtype=int)
    
    return X, Y

def one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)]
    return Y