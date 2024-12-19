import os
import random
from collections import deque
import argparse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


import numpy as np 


def split_data(X, y, num_test, seed = 123):
    random.seed(seed)
    n,_ = X.shape

    test_idx = random.sample(range(n), k = num_test)
    train_idx = list( set(range(n)) - set(test_idx) )

    X_train, X_test = X[train_idx,:], X[test_idx,:]
    y_train, y_test = y[train_idx]  , y[test_idx]

    return X_train, X_test, y_train, y_test



def read_x_file(x_file):

    features_names = None
    x_num = deque()
    x_names = deque()
    with open(x_file, 'r') as f:
        for n,i in enumerate(f.readlines()):
            line = i.strip().split()
            if n == 0:
                features_names = line[1:]
                continue

            x_names.append(line[0])
            x_num.append(list(map(float, line[1:])))

    return features_names, x_names, np.array(x_num)

def read_y_file(y_file, y_replace = '.iqtree'):
    y_num = deque()
    y_names = deque()
    with open(y_file, 'r') as f:
        for i in f.readlines():
            line = i.strip().split()
            y_names.append(line[0].replace(y_replace, '.fa'))
            y_num.append(list(map(float, line[1:])))

    return y_names, np.array(y_num)


def get_Xy_sims(x_file, y_file, y_replace = '.iqtree'):

    y_names, y_num = read_y_file(y_file, y_replace)
    features_names, x_names, x_num = read_x_file(x_file)

    choosen = set(y_names).intersection(x_names)

    y_num = y_num[[y_names.index(i) for i in choosen]]
    x_num = x_num[[x_names.index(i) for i in choosen]]

    return x_num, y_num, features_names


def LM_model(x_file, y_file, y_replace = '.iqtree'):

    X, y, feature_names = get_Xy_sims(x_file, y_file, y_replace = y_replace)
    feature_names = np.array(feature_names)
    
    # scale data
    X = X - np.mean(X, axis = 0)
    print('assuming variance = 0 for the first column (seq_len)')
    std = np.std(X[:,1:], axis = 0)
    X[:,1:] = X[:,1:] / std

    X_1 = np.hstack((np.ones((X.shape[0],1)), X))
    # print(X_1)
    coeffs =  np.abs((np.linalg.pinv(X_1) @  y))[1:,:]

    return feature_names, coeffs

def write_LM_coeffs(out_file, feature_names, coeffs):
    np.savetxt(out_file, np.vstack((feature_names, coeffs[:,1], coeffs[:,0])).T, fmt='%s')

def myargs():
    parser = argparse.ArgumentParser(description='Process iqtree files.')
    parser.add_argument('x_file', type=str, help='features file')
    parser.add_argument('y_file', type=str, help='target file')
    parser.add_argument('-o','--output', type=str, help='output file', default=None, required=True)
    parser.add_argument('-r','--replace', type=str, help='replace string in y file', default='_au.iqtree')
    args = parser.parse_args()
    return args

def main():
    args = myargs()
    feature_names, coeffs = LM_model(args.x_file, args.y_file, args.replace)
    write_LM_coeffs(args.output, feature_names, coeffs)

if __name__ == "__main__":
    main()






# x_file = '/Users/ulisesrosas/Desktop/dissectingFactors/sims/sim_filtered_oscer_progress/features_stats.tsv'
# y_file = '/Users/ulisesrosas/Desktop/dissectingFactors/sims/sim_filtered_oscer_progress/au_support_700.txt'
# feature_names, coeffs= LM_model(x_file, y_file, y_replace = '_au.iqtree')

# import matplotlib.pyplot as plt

# p = len(feature_names)
# coeffs = coeffs[:,1]
# indx = np.argsort(np.abs(coeffs))
# plt.figure(figsize=(10, 10))
# plt.barh(range(p), coeffs[indx])
# plt.yticks(range(p), feature_names[indx])
# plt.xlabel('|Coefficients|')
# plt.xscale('log')
