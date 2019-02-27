from sorter import *
from transformer import *
from vecs_io import loader
from prettytable import PrettyTable
import time


def chunk_encode(mpq, vecs):
    chunk_size = 1000000
    encoded_vecs = np.empty(
        shape=(mpq.numTable, len(vecs), mpq.M), dtype=np.float32)

    for h in range(mpq.numTable):
        # for i in tqdm.tqdm(range(math.ceil(len(vecs) / chunk_size))):
        shuffled_vecs = np.take(vecs, mpq.permutations[h], axis=1)
        for i in range(math.ceil(len(shuffled_vecs) / chunk_size)):
            encoded_vecs[h, i * chunk_size: (i + 1) * chunk_size, :] \
                = mpq.tables[h].encode(shuffled_vecs[i * chunk_size: (i + 1) * chunk_size, :].astype(dtype=np.float32))

    return encoded_vecs


def execute(mpq, X, T, Q, G, metric, train_size=100000):
    f = open('./data/result/MPQ_{a}_{b}_{c}_result.txt'.format(a=metric, b=X.shape[0], c=mpq.numTable), 'w')
    f.write('################################################################')
    f.write('################################################################\n')
    f.write('################################################################')
    f.write('################################################################\n')
    if T is None:
        f.write('# Training data size: {t}; query size: {q}.\n'.format(
            t=X.shape, q=Q.shape))
    else:
        f.write('# Training data size: {t}; query size: {q}.\n'.format(
            t=T.shape, q=Q.shape))
    f.write('# Metric: {m}, #PQ Tables:{t}.\n'.format(
        m=metric, t=mpq.numTable))
    start_time = time.time()
    np.random.seed(123)
    print("# Ranking metric {}".format(metric))
    print("# " + mpq.class_message())
    if T is None:
        mpq.fit(X[:train_size].astype(dtype=np.float32), iter=20)
    else:
        mpq.fit(T.astype(dtype=np.float32), iter=20)
    train_time = time.time()

    print('# Encoding dataset and queries...')
    vecs_encoded = chunk_encode(mpq, X)
    query_encoded = chunk_encode(mpq, Q)

    # print(vecs_encoded.shape)
    # print(query_encoded.shape)
    # print(Q.shape)

    encode_time = time.time()

    query_encoded = np.transpose(query_encoded, (1, 0, 2))
    # print(query_encoded.shape)
    # print('{}'.format(query_encoded.shape[0]==Q.shape[0]))

    print("# Sorting items...")
    Ts = [2 ** i for i in range(2 + int(math.log2(len(X))))]
    recalls, collides = BatchSorter(vecs_encoded, query_encoded, X,
                          G, Ts, metric='multihamming', batch_size=200).result()
    print("# Finish searching!\n")
    

    finish_time = time.time()
    f.write('# Training time:{t}.\n'.format(t=train_time - start_time))
    f.write('# Encoding time:{t}.\n'.format(t=encode_time - train_time))
    f.write('# Query time:{t}.\n'.format(t=finish_time - encode_time))
    f.write('################################################################')
    f.write('################################################################\n')
    f.write('################################################################')
    f.write('################################################################\n')

    table = PrettyTable()
    table.field_names = ["Expected Items", "Overall time",
                         "AVG Recall", "AVG precision", "AVG error", "AVG items"]
    for i, (t, recall) in enumerate(zip(Ts, recalls)):
        table.add_row([2 ** i, 0, recall, recall * len(G[0]) / t, 0, t])
    print(table)
    f.write(table.get_string())
    f.close()
    np.savetxt('./data/result/MPQ_{a}_{b}_{c}_collide.txt'.format(a=metric, b=X.shape[0], c=mpq.numTable), collides)
    # print("expected items, overall time, avg recall, avg precision, avg error, avg items")
    # for i, (t, recall) in enumerate(zip(Ts, recalls)):
    #     print("{}, {}, {}, {}, {}, {}".format(
    #         2**i, 0, recall, recall * len(G[0]) / t, 0, t))


def parse_args():
    # override default parameters with command line parameters
    import argparse
    parser = argparse.ArgumentParser(
        description='Process input method and parameters.')
    parser.add_argument('--num_table', type=int,
                        help='choose the number of PQ tables')
    parser.add_argument('--dataset', type=str, help='choose data set name')
    parser.add_argument('--topk', type=int,
                        help='required topk of ground truth')
    parser.add_argument('--metric', type=str, help='metric of ground truth')
    parser.add_argument('--num_codebook', type=int, help='number of codebooks')
    parser.add_argument(
        '--Ks', type=int, help='number of centroids in each quantizer')
    args = parser.parse_args()
    return args.dataset, args.topk, args.num_codebook, args.Ks, args.metric, args.num_table


if __name__ == '__main__':
    dataset = 'netflix'
    topk = 20
    codebook = 4
    Ks = 256
    metric = 'product'
    num_table = 2

    # override default parameters with command line parameters
    import sys
    if len(sys.argv) > 3:
        dataset, topk, codebook, Ks, metric, num_table = parse_args()
    else:
        import warnings
        warnings.warn("Using default Parameters ")
    print("# Parameters: #tables = {}, dataset = {}, topK = {}, codebook = {}, Ks = {}, metric = {}"
          .format(num_table, dataset, topk, codebook, Ks, metric))

    # Base, training, query, ground truth
    X, T, Q, G = loader(dataset, topk, metric, folder='data/')
    # pq, rq, or component of norm-pq
    quantizer = MPQ(numTable=num_table, M=codebook, Ks=Ks)
    execute(quantizer, X, T, Q, G, metric)