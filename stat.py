from sorter import *
from transformer import *
from vecs_io import loader
from prettytable import PrettyTable
import time


def parse_args():
    # override default parameters with command line parameters
    import argparse
    parser = argparse.ArgumentParser(
        description='Process input method and parameters.')
    parser.add_argument('--num_table', type=int,
                        help='choose the number of PQ tables')
    parser.add_argument('--alg', type=str,
                        help='choose the type of algorithm')
    parser.add_argument('--dataset', type=str, help='choose data set name')
    parser.add_argument('--metric', type=str, help='metric of ground truth')
    parser.add_argument('--ndata', type=int, help='number of data points')
    args = parser.parse_args()
    return args.alg, args.dataset, args.ndata, args.metric, args.num_table


if __name__ == '__main__':
    # override default parameters with command line parameters
    import sys
    if len(sys.argv) > 3:
        alg, dataset, ndata, metric, num_table = parse_args()
    else:
        import warnings
        warnings.warn("Using default Parameters ")

    filename = '{a}_{b}_{c}_{d}_stat.txt'.format(a=alg, b=metric, c=ndata, d=num_table)
    x = np.fromfile(filename) 

    print(x)