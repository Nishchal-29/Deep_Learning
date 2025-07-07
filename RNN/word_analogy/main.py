from load_data import load_glove_vecs
import numpy as np

words, word_to_vec = load_glove_vecs('glove.6B.200d.txt')

def cosine_similarity(vec_a, vec_b):
    prod = np.dot(vec_a, vec_b)
    norm_a = np.sqrt(np.sum(vec_a ** 2))
    norm_b = np.sqrt(np.sum(vec_b ** 2))
    return prod / (norm_a * norm_b)

boy = word_to_vec["boy"]
girl = word_to_vec["girl"]
table = word_to_vec["table"]
plant = word_to_vec["plant"]
man = word_to_vec["man"]
mars = word_to_vec["mars"]
earth = word_to_vec["earth"]

print("cosine_similarity(boy, girl) = ", cosine_similarity(boy, girl))
print("cosine_similarity(table, plant) = ",cosine_similarity(table, plant))
print("cosine_similarity(man, girl) = ",cosine_similarity(man, girl))
print("cosine_similarity(mars, earth) = ",cosine_similarity(mars, earth))


def analogy(word_a, word_b, word_c, word_to_vec):
    word_a = word_a.lower()
    word_b = word_b.lower()
    word_c = word_c.lower()
    
    e_a = word_to_vec[word_a]
    e_b = word_to_vec[word_b]
    e_c = word_to_vec[word_c]
    
    words = word_to_vec.keys()
    max_sim = -1
    best_word = None
    for word in words:
        if word not in [word_a, word_b, word_c]:
            e_word = word_to_vec[word]
            cosine_sim = cosine_similarity(e_b - e_a, e_word - e_c)
            if cosine_sim > max_sim:
                max_sim = cosine_sim
                best_word = word
                
    return best_word

tests = [('India', 'Indian', 'Spain'), ('India', 'Delhi', 'Nepal'), ('Earth', 'Blue', 'Mars'), ('boy', 'man', 'girl')]
for test in tests:
    print ('{} -> {} :: {} -> {}'.format( *test, analogy(*test, word_to_vec)))
    