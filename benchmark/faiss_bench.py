import faiss
import numpy as np
import numba as nb
from vecs_io import loader
from sorter import Sorter

@nb.jit
def true_positives(topK, Q, G, T):
    result = np.empty(shape=(len(Q)))
    for i in nb.prange(len(Q)):
        result[i] = len(np.intersect1d(G[i], topK[i][:T]))
    return result

def sum_recall(topK, Q, G, T):
    # Compute #TP for each q \in Q
    # G: the KNN computed by PQ algorithm
    true_positive = true_positives(topK, Q, G, T)
    return np.sum(true_positive) / len(G[0])  # TP / K

if __name__ == '__main__':
    dataset = 'sift'
    topk = 20
    codebook = 4
    Ks = 256
    metric = 'euclidean'
    num_table = 2
    d = 128
    nBits = 2 * 128

    lsh = faiss.IndexLSH(d, nBits)   # build the index

    # Base, training, query, ground truth
    X, T, Q, G = loader(dataset, topk, metric, folder='data/')
    number_of_queries = len(Q)

    lsh.train(X)                  # add vectors to the index
    lsh.add(X)
    D, I = lsh.search(Q, topk)     # actual search

    Ts = [2 ** i for i in range(2 + int(math.log2(len(X))))]

    recalls = np.zeros(shape=(len(Ts))) 
    recalls[:] = recalls[:] + [sum_recall(topk, Q, G, t) for t in Ts]