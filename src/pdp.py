
import os
import sys

import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
from recurrent_utils import *
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold


import tensorflow as tf
from tensorflow import keras
n_workers = 6

def split_data(X,y,num_test, seed = 123):

    random.seed(seed)
    n,_ = X.shape

    test_idx  = random.sample(range(n), k = num_test)
    train_idx = list( set(range(n)) - set(test_idx) )

    X_train, X_test = X[train_idx,:], X[test_idx,:]
    y_train, y_test = y[train_idx]  , y[test_idx]

    return X_train,y_train, X_test, y_test

def permutation_importance(X, y, model, num_test = 100, seed = 12038, iterations = 1000):

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

def sample_feature(X, feature, quantiles, sample, integer = False):
    """
    sample unweighted column
    """
    # X = X_train_new
    # feature =1
    Xpdp = X[:,feature]

    if integer:
        return np.sort(np.unique(Xpdp))
    
    q = np.linspace(
        start = quantiles[0],
        stop  = quantiles[1],
        num   = sample
    )

    # sample from the original feature without interpolation
    Xq = np.quantile(Xpdp, q = q, interpolation='nearest')
    # Xq[5] in Xpdp
    # Xq[7] in Xpdp
    # plt.hist(Xpdp)
    # for i in Xq:
    #     # add a vertical line
    #     plt.axvline(i, color = 'red')

    return Xq    

def pdp_1d(X, feature, model, quantiles = [0,1], sample = 70, integer = False):
    """
    1D partial dependence plot
    """

    Xq = sample_feature(X, feature, quantiles, sample, integer)
    pdp_values = []
    for n in Xq:
        # copy original values
        X_tmp = X.copy()
        # make original values with 
        # modified feature column
        X_tmp[:,feature] = n

        pdp_values.append(
            np.mean(
                model.predict(X_tmp)
            )
        )

    out = np.hstack((
        Xq.reshape(-1,1), 
        np.array(pdp_values).reshape(-1,1)
        ))
    
    return out

def pdp_2d(X, f1_idx, f2_idx, model, quantiles, sample, integer):
    """
    2D partial dependence plot
    """

    Xq1 = sample_feature(X, f1_idx, quantiles, sample, integer[0])
    Xq2 = sample_feature(X, f2_idx, quantiles, sample, integer[1])

    n_xq1 = len(Xq1)
    n_xq2 = len(Xq2)

    X1, X2 = np.meshgrid(Xq1, Xq2)

    Y1 = np.zeros((n_xq1, n_xq2))
    Y2 = np.zeros((n_xq1, n_xq2))
    for i in range(n_xq1):
        for j in range(n_xq2):
            # j,i = 0,0
            X_tmp = X.copy()
            X_tmp[:,f1_idx] = X1[i,j]
            X_tmp[:,f2_idx] = X2[i,j]

            ave_h1, ave_h2 = np.mean( model.predict(X_tmp, workers=n_workers, verbose=0), 0 )

            Y1[i,j] = ave_h1
            Y2[i,j] = ave_h2

    return X1, X2, Y1, Y2


def stack_pdp(X1, X2, Y1):
    stacked = []
    for i in range(X1.shape[0]):
        for j in range(X1.shape[1]):
            stacked.append([X1[i,j], X2[i,j], Y1[i,j]])

    return stacked


# data --------------------------------------
suffix = 'PROTA'
out_folder = "./../data"
base_path = './../data'

if suffix == 'FLAT':
    # K = keras.backend
    f1_name = "gc_mean_pos3"
    f2_name = "inter_len_mean"


    # Features: qcutils results
    features_file    = os.path.join(base_path, 'flat_features.tsv')
    # Labels: GGI results
    all_ggi_results  = os.path.join(base_path, 'join_ggi_flat_version2.tsv' )

    hyper_file = os.path.join(base_path, 'hyperparamters/DNN_FLAT.txt')
    
else:
    f1_name = "gap_var_pos3"
    f2_name = "LB_std"
    
    # Features: qcutils results
    features_file    = os.path.join(base_path, 'prota_features.tsv')
    # Labels: GGI results
    all_ggi_results  = os.path.join(base_path, 'joined_1017_two_hypos_prota.txt' )

    hyper_file = os.path.join(base_path, 'hyperparamters/DNN_PROTA.txt')
# data --------------------------------------




self = Post_ggi(
    feature_file = features_file,
    all_ggi_results = all_ggi_results,
)

new_df = self.features
ggi_pd = pd.DataFrame( self.ggi_df[1:], columns=self.ggi_df[0]   )

all_labels, new_df = make_au_labels( ggi_pd, new_df )
new_df_num = new_df.drop(["aln_base"], axis = 1)
all_labels_dis = np.argmax( all_labels, axis=1 ) == 0

# load hyperparameters
with open(hyper_file, 'r') as f:
    myparams = eval(f.readline().strip())



# pdp parameters
sample = 5
f1_idx = new_df_num.columns.get_loc(f1_name)
f2_idx = new_df_num.columns.get_loc(f2_name)
quantiles = [0,1]
integer = [False, False]



########## iteration parameters ###################
# max_trials = 50
# n_epochs = 1500
# boots = 5
######## iteration parameters ###################
# tests
max_trials = 2
n_epochs = 5
boots = 2


##########   CROSS-VALIDATION ###########
print('starting cross validation')

# region
def build_model(params, input_shape):

    model = keras.Sequential()
    model.add( keras.layers.InputLayer(input_shape = input_shape) )

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
    
    optimizer = keras.optimizers.legacy.SGD(
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


early_stopping_cb = keras.callbacks.EarlyStopping('val_loss', patience = 100, restore_best_weights=True, mode = 'min')

# prota
# lowest_loss = float('+Inf')
# n_epochs = 5
X1 = np.zeros((sample, sample))
X2 = np.zeros((sample, sample))
Y1 = np.zeros((sample, sample))
Y2 = np.zeros((sample, sample))


Y1s = []
Y2s = []
cv = 0
for b in range(boots):
    # b = 1
    sys.stdout.write(f'\n\nboot: {b}\n')
    kfold = StratifiedKFold(n_splits=4, shuffle=True, random_state=None)

    for train, test in kfold.split(new_df, all_labels_dis):
        train,test
        # len(test)/len(train)
        
        X_train = new_df_num.iloc[train,:]
        y_train = all_labels[train]

        X_test = new_df_num.iloc[test,:]
        y_test = all_labels[test]

        X_train_new = transform_data(X_train, X_train)
        X_test_new  = transform_data(X_train, X_test)

        resampled_features, resampled_labels = do_resampling_dis(X_train_new, y_train)

        model = build_model(myparams, input_shape = resampled_features.shape[-1])

        model.fit(
            resampled_features,
            resampled_labels,
            epochs=n_epochs,
            validation_data=( X_test_new, y_test ),
            callbacks =[
                 early_stopping_cb,
            ],
            workers=n_workers,
            use_multiprocessing=True,
            verbose=0
        )
        # evaluate the model
        loss, cos_simi = model.evaluate(X_test_new, y_test, verbose=0)

        sys.stdout.write("\n\033[92m%s: %.2f\033[0m\n" % (model.metrics_names[0], loss))
        sys.stdout.write("%s: %.2f\n"                  % (model.metrics_names[1], cos_simi))
        sys.stdout.write("Calculating PDPs\n")
        sys.stdout.flush()

        tmp_X1, tmp_X2, tmp_Y1, tmp_Y2 = pdp_2d(X_test_new, f1_idx, f2_idx, model, quantiles, sample, integer)

        for i in range(X1.shape[0]):
            for j in range(X1.shape[1]):
                X1[i,j] += tmp_X1[i,j]
                X2[i,j] += tmp_X2[i,j]
                Y1[i,j] += tmp_Y1[i,j]
                Y2[i,j] += tmp_Y2[i,j]
        cv += 1


X1 /= cv
X2 /= cv
Y1 /= cv
Y2 /= cv



# store these four matrices with '_prota' as suffix
np.save(f'X1_{suffix}', X1)
np.save(f'X2_{suffix}', X2)
np.save(f'Y1_{suffix}', Y1)
np.save(f'Y2_{suffix}', Y2)




