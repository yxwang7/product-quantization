from pq import *
from multiprocessing import cpu_count
import numba as nb
import math
import tqdm
from scipy.spatial.distance import hamming

@nb.jit
def arg_sort(distances): # Return indices of top-k neighbors
    top_k = min(131072, len(distances)-1)
    # Return Top-K indices
    indices = np.argpartition(distances, top_k)[:top_k]
    return indices[np.argsort(distances[indices])]


@nb.jit
def product_arg_sort(q, compressed):
    distances = np.dot(compressed, -q)
    return arg_sort(distances)

@nb.jit
def angular_arg_sort(q, compressed, norms_sqr):
    norm_q = np.linalg.norm(q)
    distances = np.dot(compressed, q) / (norm_q * norms_sqr)
    return arg_sort(distances)


@nb.jit
def euclidean_arg_sort(q, compressed):
    distances = np.linalg.norm(q - compressed, axis=1)
    return arg_sort(distances)


@nb.jit
def sign_arg_sort(q, compressed):
    distances = np.empty(len(compressed), dtype=np.int32)
    for i in range(len(compressed)):
        distances[i] = np.count_nonzero((q > 0) != (compressed[i] > 0))
    return arg_sort(distances)


@nb.jit
def euclidean_norm_arg_sort(q, compressed, norms_sqr):
    distances = norms_sqr - 2.0 * np.dot(compressed, q)
    return arg_sort(distances)

# @nb.jit
# def hamming_arg_sort(q_encoded, vecs_encoded):
#     distances = np.empty(len(vecs_encoded), dtype=np.int32)
#     for i in range(len(vecs_encoded)):
#         distances[i] = hamming(q_encoded, vecs_encoded[i])
#     return arg_sort(distances)

# @nb.jit
# def binary_arg_sort(q_encoded, vecs_encoded):
#     distances = np.empty(len(vecs_encoded), dtype=np.int32)
#     for i in range(len(vecs_encoded)):
#         distances[i] = 0 if (int(hamming(q_encoded, vecs_encoded[i])) == 0) else 1
#     return arg_sort(distances)

@nb.jit
def multiple_table_sort(q_encoded, vecs_encoded):
    distances = np.zeros(vecs_encoded.shape[1], dtype=np.int32)
    mid = np.product((q_encoded != np.transpose(vecs_encoded, (1,0,2)).astype(int)), axis=2)
        # for i in nb.prange(vecs_encoded.shape[1]):
        #     distances[h, i] = 0 if np.array_equal(q_encoded[h, :], vecs_encoded[h, i, :]) else 1
    distances = np.sum(mid, axis=1) / len(q_encoded)
    return arg_sort(distances)
    

@nb.jit
def parallel_sort(metric, compressed, Q, X, norms_sqr=None):
    """
    for each q in 'Q', sort the compressed items in 'compressed' by their distance,
    where distance is determined by 'metric'
    :param metric: euclid product
    :param compressed: compressed items, same dimension as origin data, shape(N * D)
    :param Q: queries, shape(len(Q) * D)
    :return:
    """
    rank = None
    if(metric == 'multitable'):
        rank = np.empty((Q.shape[0], min(131072, compressed.shape[1]-1)), dtype=np.int32)
    else:
        rank = np.empty((Q.shape[0], min(131072, compressed.shape[0]-1)), dtype=np.int32)

    p_range = nb.prange(Q.shape[0])

    if metric == 'product':
        for i in p_range:
            rank[i, :] = product_arg_sort(Q[i], compressed)
    elif metric == 'angular':
        if norms_sqr is None:
            norms_sqr = np.linalg.norm(compressed, axis=1) ** 2
        for i in p_range:
            rank[i, :] = angular_arg_sort(Q[i], compressed, norms_sqr)
    elif metric == 'euclid_norm':
        if norms_sqr is None:
            norms_sqr = np.linalg.norm(compressed, axis=1) ** 2
        for i in p_range:
            rank[i, :] = euclidean_norm_arg_sort(Q[i], compressed, norms_sqr)
    elif metric == 'multitable':
        for i in p_range:
            rank[i, :] = multiple_table_sort(Q[i], compressed)
    else:
        for i in p_range:
            rank[i, :] = euclidean_arg_sort(Q[i], compressed)

    return rank # rank: sizeof(Q) \times top-k matrix


@nb.jit
def true_positives(topK, Q, G, T):
    result = np.empty(shape=(len(Q)))
    for i in nb.prange(len(Q)):
        result[i] = len(np.intersect1d(G[i], topK[i][:T]))
    return result


class Sorter(object):
    def __init__(self, compressed, Q, X, metric, norms_sqr=None):
        self.Q = Q
        self.X = X
        self.topK = parallel_sort(metric, compressed, Q, X, norms_sqr=norms_sqr)

    def recall(self, G, T):
        t = min(T, len(self.topK[0]))
        # Compute the average recall on Q
        return t, self.sum_recall(G, T) / len(self.Q)

    def sum_recall(self, G, T):
        assert len(self.Q) == len(self.topK), "number of query not equals"
        assert len(self.topK) <= len(G), "number of queries should not exceed the number of queries in ground truth"
        # Compute #TP for each q \in Q
        # G: the KNN computed by PQ algorithm
        true_positive = true_positives(self.topK, self.Q, G, T)
        return np.sum(true_positive) / len(G[0])  # TP / K


class BatchSorter(object):
    def __init__(self, compressed, Q, X, G, Ts, metric, batch_size, norms_sqr=None):
        self.Q = Q
        self.X = X
        self.recalls = np.zeros(shape=(len(Ts)))
        for i in range(math.ceil(len(Q) / float(batch_size))):
            q = None
            if metric == 'multitable':
                q = Q[i * batch_size: (i + 1) * batch_size, :, :]
            else:
                q = Q[i * batch_size: (i + 1) * batch_size, :]
            g = G[i * batch_size: (i + 1) * batch_size, :]
            # compressed: compressed database; q: part of query; X: original database
            sorter = Sorter(compressed, q, X, metric=metric, norms_sqr=norms_sqr)
            self.recalls[:] = self.recalls[:] + [sorter.sum_recall(g, t) for t in Ts]
        self.recalls = self.recalls / len(self.Q)

    def recall(self):
        return self.recalls
