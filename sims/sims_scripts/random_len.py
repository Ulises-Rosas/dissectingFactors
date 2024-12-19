#!/usr/bin/env python3

import numpy as np
import argparse

# iqtree2="/Users/ulisesrosas/Desktop/iqtree-2.3.6-macOS/bin/iqtree2"
# twohypos="/Users/ulisesrosas/Desktop/dissectingFactors/sims/prota_sims/prota_myboth_hypos_v2.txt"

def generate_seqs_lenghts(n_alns, ave, var, seed, output):
    np.random.seed(seed)
    a = ( np.random.normal(ave, var, n_alns) //3 )*3
    if not output:
        for i in a:
            print(int(i))

    else:
        np.savetxt(output, a, fmt='%i')
    # return a

def args_prota():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--seed', type=int, help='seed for the random number generator')
    parser.add_argument('--n', type=int, help='number of alignments to simulate')
    parser.add_argument('--mean', type=int, help='average length of the alignments')
    parser.add_argument('--var', type=int, help='variance length of the alignments')
    parser.add_argument('--output', type=str, help='output file', default=None)
    args = parser.parse_args()
    return args

def main():
    args = args_prota()
    generate_seqs_lenghts(args.n, args.mean, args.var, args.seed, args.output)

if __name__ == "__main__":
    main()
