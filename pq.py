from __future__ import division
from __future__ import print_function
import numpy as np
from scipy.cluster.vq import vq, kmeans2
import tqdm
import numba as nb



class PQ(object):
    def __init__(self, M, Ks, verbose=True):
        assert 0 < Ks <= 2 ** 32
        self.M, self.Ks, self.verbose = M, Ks, verbose
        self.code_dtype = np.uint8 if Ks <= 2 ** 8 else (np.uint16 if Ks <= 2 ** 16 else np.uint32)
        self.codewords = None
        self.Ds = None
        self.Dim = -1

    # M: number of sub-quantizers; Ks: number of centroids in each sub-quantizer.
    def class_message(self):
        return "Subspace PQ, M: {}, Ks : {}, code_dtype: {}".format(self.M, self.Ks, self.code_dtype)

    # fit: K-means training
    def fit(self, vecs, iter):
        assert vecs.dtype == np.float32
        assert vecs.ndim == 2
        N, D = vecs.shape
        assert self.Ks < N, "the number of training vector should be more than Ks"
        self.Dim = D

        reminder = D % self.M
        quotient = int(D / self.M)
        # [q + 1, ..., q + 1, q, ..., q]
        dims_width = [quotient + 1 if i < reminder else quotient for i in range(self.M)]
        self.Ds = np.cumsum(dims_width)     # prefix sum
        self.Ds = np.insert(self.Ds, 0, 0)  # insert zero at beginning

        # [m][ks][ds]: m-th subspace, ks-the codeword, ds-th dim
        self.codewords = np.zeros((self.M, self.Ks, np.max(self.Ds)), dtype=np.float32)
        for m in range(self.M):
            if self.verbose:
                print("#    Training the subspace: {} / {}, {} -> {}".format(m, self.M, self.Ds[m], self.Ds[m+1]))
            # All columns and specific roes
            vecs_sub = vecs[:, self.Ds[m]:self.Ds[m+1]]
            # dims_width[m] = self.Ds[m+1] - self.Ds[m]
            # Gaussian distribution initialization
            self.codewords[m, :, :self.Ds[m+1] - self.Ds[m]], _ = kmeans2(
                vecs_sub, self.Ks, iter=iter, minit='random')

        return self

    def encode(self, vecs):
        assert vecs.dtype == np.float32
        assert vecs.ndim == 2
        N, D = vecs.shape

        # codes[n][m] : code of n-th vec, m-th subspace
        codes = np.empty((N, self.M), dtype=self.code_dtype)
        for m in range(self.M):
            vecs_sub = vecs[:, self.Ds[m]: self.Ds[m+1]]
            # Last parameter: fix the length (because the #dimensions of each subspace is not equivalent).
            # vq from scipy: distance measure?
            codes[:, m], _ = vq(vecs_sub,
                                self.codewords[m, :, :self.Ds[m+1] - self.Ds[m]])

        return codes

    def decode(self, codes):
        assert codes.ndim == 2
        N, M = codes.shape
        assert M == self.M
        assert codes.dtype == self.code_dtype

        vecs = np.empty((N, self.Dim), dtype=np.float32)
        for m in range(self.M):
            vecs[:, self.Ds[m]: self.Ds[m+1]] = self.codewords[m, codes[:, m], :self.Ds[m+1] - self.Ds[m]]

        return vecs

    def compress(self, vecs):
        return self.decode(self.encode(vecs))

class MPQ(object):
    @nb.jit
    def __init__(self, numTable, M, Ks, verbose=True):
        self.numTable, self.M, self.Ks, self.verbose = numTable, M, Ks, verbose
        self.tables = []
        self.code_dtype = np.uint8 if Ks <= 2 ** 8 else (
            np.uint16 if Ks <= 2 ** 16 else np.uint32)
        for _ in nb.prange(self.numTable):
            self.tables.append(PQ(M=self.M, Ks=self.Ks, verbose=False))

    def class_message(self):
        return "#PQ Table: {}, ".format(self.numTable) + self.tables[0].class_message()

    def fit(self, vecs, iter):
        print('\n# Start training...')

        # generate (self.numTable) ways of index permutations and store it
        self.permutations = [np.random.rand(
            vecs.shape[1]).argsort() for i in nb.prange(self.numTable)]
        if(self.verbose):
            for i in tqdm.tqdm(nb.prange(self.numTable)):
                self.tables[i].fit(vecs[:, self.permutations[i]], iter)
        else:
            for i in nb.prange(self.numTable):
                self.tables[i].fit(vecs[:, self.permutations[i]], iter)
        print('# Training finish!\n')

    def encode(self, vecs):
        assert vecs.dtype == np.float32
        assert vecs.ndim == 2
        N, D = vecs.shape
        codes = np.empty(
            (self.numTable, N, self.tables[0].M), dtype=self.code_dtype)
        for i in nb.prange(self.numTable):
            codes[i, :, :] = self.tables[i].encode(
                vecs[:, self.permutations[i]])
        return codes

    def decode(self, codes):
        assert codes.ndim == 3
        nTable, N, M = codes.shape
        assert nTable == self.numTable
        vecs = np.empty((self.numTable, N, self.tables[0].Dim), dtype=np.float32)
        for i in nb.prange(self.numTable):
            vecs[i, :, :] = self.tables[i].decode(codes[i, :, :])
            # Need verification
            vecs = vecs[:, self.permutations[i].argsort()]
        return np.array(vecs)

    def compress(self, vecs):
        return [t.compress(vecs) for t in self.tables]