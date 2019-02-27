from sorter import *
from transformer import *
from vecs_io import loader
from prettytable import PrettyTable


def chunk_compress(pq, vecs):
    chunk_size = 1000000
    compressed_vecs = np.empty(shape=vecs.shape, dtype=np.float32)
    for i in tqdm.tqdm(range(math.ceil(len(vecs) / chunk_size))):
        compressed_vecs[i * chunk_size: (i + 1) * chunk_size, :] \
            = pq.compress(vecs[i * chunk_size: (i + 1) * chunk_size, :].astype(dtype=np.float32))
    return compressed_vecs


def execute(pq, X, T, Q, G, metric, train_size=100000):
    Q=Q[0:100, :]
    np.random.seed(123)
    f = open('PQ_{a}_{b}_{c}_result.txt'.format(a=metric, b=X.shape[0], c=Q.shape[0]), 'w')
    print("# Ranking metric {}".format(metric))
    print("# " + pq.class_message())
    if T is None:
        pq.fit(X[:train_size].astype(dtype=np.float32), iter=20)
    else:
        pq.fit(T.astype(dtype=np.float32), iter=20)

    print('# Compressing items...')
    compressed = chunk_compress(pq, X)
    print("# Sorting items...")
    Ts = [2 ** i for i in range(2 + int(math.log2(len(X))))]
    recalls = BatchSorter(compressed, Q, X, G, Ts, metric=metric, batch_size=200).recall()
    print("# Searching...\n")

    table = PrettyTable()
    table.field_names = ["Expected Items", "Overall time", "AVG Recall", "AVG precision", "AVG error", "AVG items"]
    for i, (t, recall) in enumerate(zip(Ts, recalls)):
        table.add_row([2 ** i, 0, recall, recall * len(G[0]) / t, 0, t])
    
    print(table)
    f.write(table.get_string())
    f.close()
    # print("expected items, overall time, avg recall, avg precision, avg error, avg items")
    # for i, (t, recall) in enumerate(zip(Ts, recalls)):
    #     print("{}, {}, {}, {}, {}, {}".format(
    #         2**i, 0, recall, recall * len(G[0]) / t, 0, t))


def parse_args():
    # override default parameters with command line parameters
    import argparse
    parser = argparse.ArgumentParser(description='Process input method and parameters.')
    parser.add_argument('--dataset', type=str, help='choose data set name')
    parser.add_argument('--topk', type=int, help='required topk of ground truth')
    parser.add_argument('--metric', type=str, help='metric of ground truth')
    parser.add_argument('--num_codebook', type=int, help='number of codebooks')
    parser.add_argument('--Ks', type=int, help='number of centroids in each quantizer')
    args = parser.parse_args()
    return args.dataset, args.topk, args.num_codebook, args.Ks, args.metric


if __name__ == '__main__':
    dataset = 'netflix'
    topk = 20
    codebook = 4
    Ks = 256
    metric = 'product'
    # override default parameters with command line parameters
    import sys
    if len(sys.argv) > 3:
        dataset, topk, codebook, Ks, metric = parse_args()
    else:
        import warnings
        warnings.warn("Using default Parameters ")
    print("# Parameters: dataset = {}, topK = {}, codebook = {}, Ks = {}, metric = {}"
          .format(dataset, topk, codebook, Ks, metric))

    # Base, training, query, ground truth
    X, T, Q, G = loader(dataset, topk, metric, folder='data/')
    # pq, rq, or component of norm-pq
    quantizer = PQ(M=codebook, Ks=Ks)
    execute(quantizer, X, T, Q, G, metric)
