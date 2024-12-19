import os
import time
import random
from collections import deque
import json
import argparse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


import numpy as np 
import tensorflow as tf
from tensorflow import keras
import keras_tuner




def split_data(X, y, num_test, seed = 123):
    random.seed(seed)
    n,_ = X.shape

    test_idx = random.sample(range(n), k = num_test)
    train_idx = list( set(range(n)) - set(test_idx) )

    X_train, X_test = X[train_idx,:], X[test_idx,:]
    y_train, y_test = y[train_idx]  , y[test_idx]

    return X_train, X_test, y_train, y_test

def permutation_importance(X, y, model, num_test = 100, seed = 12038, iterations = 1000):
    """
    Permutation importance

    X: data
    y: target
    model: model
    num_test: number of test samples
    seed: random seed
    iterations: number of iterations

    returns: Feature importance for all features. 
        Each column represents the change in error
    """

    np.random.seed(seed)

    n,p = X.shape

    X_train, X_test, y_train,y_test = split_data(X, y, num_test, seed = seed)
    model.fit(X_train, y_train)
    error_orig = model.score(X_test, y_test)


    out = np.zeros((iterations, p))

    for i in range(iterations):

        for j in range(X_test.shape[1]):
            # Create a copy of X_test
            X_test_copy = X_test.copy()

            # Scramble the values of the given predictor
            X_test_copy[:,j] = np.random.permutation(X_test_copy[:,j])
                        
            # Calculate the new RMSE
            error_perm = model.score(X_test_copy, y_test)

            out[i,j] = error_perm - error_orig

    return out

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

def hyperparameter_tunning(X_train, y_train, X_test, y_test,
                            max_trials = 10, 
                            n_epochs = 2000, 
                            ncpus = 2, out_folder = ".",
                            prefix = 'sims'):

   
    # X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    print(f"X_train shape: {X_train.shape}")

    def hp_caller(hp):
        model = keras.Sequential()
        model.add( keras.layers.InputLayer( shape = (X_train.shape[1],) ) )

        # Tune the number of layers.
        for i in range(hp.Int("num_layers", 4, 10)):
            drop_out = hp.Float(f"drop_{i}", min_value=1e-4, max_value=0.9, sampling="log")
            
            model.add( keras.layers.BatchNormalization() )
            model.add( keras.layers.Dropout(drop_out) )
            model.add(
                keras.layers.Dense(
                    # Tune number of units separately.
                    units=hp.Int(f"units_{i}", min_value=5, max_value=100, step=3),
                    kernel_initializer = 'lecun_normal',
                    use_bias = False
                )
            )
            model.add( keras.layers.Activation('selu') )

        model.add( keras.layers.Dense( units = 2, activation = 'selu' ) )

        learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-1, sampling="log")
        decay_rate = hp.Float("decay", min_value=1e-7, max_value=9e-1, sampling="log")
        
        optimizer = keras.optimizers.SGD(
            learning_rate = learning_rate,
            momentum = 0.90,
            nesterov = True,
            decay = decay_rate,
            )

        model.compile(
            optimizer=optimizer,
            loss = 'mse',
            metrics=[ tf.keras.metrics.CosineSimilarity(axis=1) ]
        )

        return model


    # max_trials = 10
    # n_epochs = 2000
    tuner = keras_tuner.BayesianOptimization(
        hypermodel=hp_caller,
        objective="val_loss",
        max_trials=max_trials,
        overwrite=True,
        seed=12038,
        project_name=f"{prefix}", # folder name
    )

    early_stopping_cb = keras.callbacks.EarlyStopping('val_loss', patience =100, restore_best_weights=True, mode = 'min')

    tuner.search(
        x = X_train,
        y = y_train,
        epochs=n_epochs,
        validation_data=(
            X_test, 
            y_test, 
        ),
        callbacks=[
            early_stopping_cb,
            # onecycle
        ],
        # workers=ncpus,
        # use_multiprocessing=True,
        batch_size = 32,

    )

    # import time
    sele_model = tuner.get_best_models()[0]
    loss,cos_simi = sele_model.evaluate( X_test, y_test)

    o_name_base = f"{prefix}_tuner_E{round(loss,6)}_S{round(cos_simi,6)}_ID{int(time.time())}_hyperparams.json"
    o_name = os.path.join( out_folder, o_name_base )

    myparams = tuner.get_best_hyperparameters()[0].values

    with open( o_name, 'w') as f:
        json.dump( myparams, f)

    print()
    print(f"""
        Hyperparameter Test dataset
        loss    : {loss}
        cos sim : {cos_simi}
        """)
    loss2, cos_simi2 = sele_model.evaluate(X_train, y_train)

    print(f"""
        Hyperparameter Train dataset
        loss  : {loss2}
        cos sim : {cos_simi2}
        """)
    print()

    return myparams

def set_dnn_model(params, shape):

    # keras.utils.set_random_seed(seed = seed)

    model = keras.Sequential()
    model.add( keras.layers.InputLayer(shape = shape) )
    # model.add( keras.layers.InputLayer( input_shape = (X_train.shape[1],) ) )

    for l in range( params['num_layers'] ):

        units = params[f'units_{l}']
        drop_rate = params[f'drop_{l}']

        model.add( keras.layers.BatchNormalization() )
        model.add( keras.layers.Dropout(drop_rate) )
        model.add(
            keras.layers.Dense(
                units = units,
                kernel_initializer = 'lecun_normal',
                # activity_regularizer = keras.regularizers.L1(act_reg),
                use_bias = False
            )
        )
        model.add( keras.layers.Activation('selu') )

    model.add( keras.layers.Dense( units = 2, activation = 'selu' ) )

    learning_rate = params['lr']
    decay_rate    = params['decay']
    
    optimizer = keras.optimizers.SGD(
        learning_rate = learning_rate,
        momentum = 0.90,
        nesterov = True,
        decay = decay_rate,
    )

    model.compile(
        optimizer=optimizer,
        loss = 'mse',
        metrics=[ tf.keras.metrics.CosineSimilarity(axis=1) ]
    )
    return model

def DNN_model(X_train, y_train, X_test, y_test, myparams, n_epochs, ncpus = 2):

    early_stopping_cb = keras.callbacks.EarlyStopping('val_loss',
                                                       patience = 100, 
                                                       restore_best_weights=True, 
                                                       mode = 'min')

    model = set_dnn_model(myparams, shape = (X_train.shape[1],))

    model.fit(
        X_train,
        y_train,
        epochs=n_epochs,
        validation_data=( X_test, y_test ),
        callbacks =[
                early_stopping_cb,
        ],
        # workers=ncpus,
        # use_multiprocessing=True,
        # verbose=0
    )

    error_orig, cos_orig = model.evaluate(X_test, y_test)

    print(f"""
        Test dataset after training
        loss    : {error_orig}
        cos sim : {cos_orig}
        """)
    
    return model, error_orig, cos_orig


def permutation_errors(iterations, X_test, y_test, model, error_orig, seed = 12038):
    np.random.seed(seed)

    out = np.zeros( (iterations, X_test.shape[1]) )
    for i in range(iterations):
        print(f'Iteration: {i}')
        for j in range(X_test.shape[1]):
            # Create a copy of X_test
            X_test_copy = X_test.copy()

            # Scramble the values of the given predictor
            X_test_copy[:,j] = np.random.permutation(X_test_copy[:,j])
                        
            # Calculate the new RMSE
            error_perm,_ = model.evaluate(X_test_copy, y_test, verbose=0)

            out[i,j] = error_perm - error_orig

    return out


def DNN_pipeline(out_folder, prefix, x_file, y_file, max_trials, n_epochs, ncpus, test_size, seed, iterations, n_epochs_tuner = 2000):

    X, y, feature_names = get_Xy_sims(x_file, y_file, y_replace = '_au.iqtree')
    feature_names = np.array(feature_names)

    # scale data
    X = X - np.mean(X, axis = 0)
    
    std = np.std(X, axis = 0)
    std[std == 0] = 1

    X = X / std


    n,p = X.shape
    X_train, X_test, y_train, y_test = split_data(X, y, num_test = int(test_size*n), seed = seed)

    # X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    # X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    # Hyperparameter tunning
    print('\nHyperparameter tunning started')
    myparams = hyperparameter_tunning(X_train, y_train, X_test, y_test, 
                                    max_trials = max_trials, n_epochs = n_epochs_tuner,
                                    ncpus = ncpus, out_folder = out_folder, prefix = prefix)

    # DNN model
    print('\nDNN model started')
    model, error_orig, _ = DNN_model(X_train, y_train, X_test, y_test, myparams, n_epochs = n_epochs, ncpus = ncpus)

    # save model
    model.save( os.path.join(out_folder, f'{prefix}_model.keras') )

    # Permutation importance
    print('\nPermutation importance started')
    FI = permutation_errors(iterations, X_test, y_test, model, error_orig, seed = seed)

    # save FI
    np.savetxt( os.path.join(out_folder, f'{prefix}_FI.txt'), FI)



def myargs():
    parser = argparse.ArgumentParser(description='DNN pipeline', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('x_file', type=str, help='features file')
    parser.add_argument('y_file', type=str, help='target file')
    parser.add_argument('-o','--output', type=str, help='output folder', default='.')
    parser.add_argument('-p','--prefix', type=str, help='prefix for output files', default='test')
    parser.add_argument('-t','--max_trials', type=int, help='max trials for hyperparameter tunning', default=10)
    parser.add_argument('-e','--n_epochs', type=int, help='number of epochs', default=2000)
    parser.add_argument('--tu_epochs', type=int, help='number of epochs for hyperparameter tunning', default=2000)
    parser.add_argument('-c','--ncpus', type=int, help='number of cpus', default=2)
    parser.add_argument('-s','--test_size', type=float, help='test size', default=0.35)
    parser.add_argument('-i','--iterations', type=int, help='number of iterations for permutation importance', default=10)
    parser.add_argument('--seed', type=int, help='random seed', default=12038)
    args = parser.parse_args()
    return args

def main():
    args = myargs()
    
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    # keras_tuner.random.set_seed(args.seed)

    DNN_pipeline(args.output, args.prefix, args.x_file, args.y_file, 
                 args.max_trials,args.n_epochs, args.ncpus, args.test_size, 
                 args.seed, args.iterations, args.tu_epochs)
    
if __name__ == "__main__":
    main()



# import matplotlib.pyplot as plt
# aves = np.mean(FI, axis=0)
# indx = np.argsort(aves)
# plt.figure(figsize=(10, 10))
# plt.barh(range(p), aves[indx])
# plt.yticks(range(p), feature_names[indx])
# plt.xlabel('RMSE increase')
# # plt.xscale('log')


