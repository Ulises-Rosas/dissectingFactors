import os
import sys
import time
import pickle

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# import csv
import numpy as np 
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from recurrent_utils import *

import tensorflow as tf
from tensorflow import keras

import eli5
from eli5.sklearn import PermutationImportance

# data --------------------------------------
suffix = 'PROTA'
out_folder = "./../data"
base_path = './../data'


# Features: qcutils results
features_file    = os.path.join(base_path, 'prota_features.tsv')
# Labels: GGI results
all_ggi_results  = os.path.join(base_path, 'joined_1017_two_hypos_prota.txt' )

# hyperparameters file
hyper_file = os.path.join(base_path, 'hyperparamters/DNN_Encoder_PROTA.txt')
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


########## iteration parameters ###################
max_trials = 50
n_epochs = 1500
boots = 20
######## iteration parameters ###################
# tests
# max_trials = 2
# n_epochs = 5
# boots = 2


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

cvscores = []
for b in range(boots):
    sys.stdout.write(f'\n\nboot: {b}\n')

    kfold = StratifiedKFold(n_splits=4, shuffle=True, random_state=None)

    cv = 1
    for train, test in kfold.split(new_df, all_labels_dis):
        # train,test
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
            workers=10,
            use_multiprocessing=True,
            verbose=0
        )
        # evaluate the model
        loss, cos_simi = model.evaluate(X_test_new, y_test, verbose=0)

        sys.stdout.write("\n\033[92m%s: %.2f\033[0m\n" % (model.metrics_names[0], loss))
        sys.stdout.write("%s: %.2f\n"                  % (model.metrics_names[1], cos_simi))
        sys.stdout.write("Calculating Feature Importances\n")
        sys.stdout.flush()

        perm = PermutationImportance(
            model, 
            scoring = 'neg_mean_squared_error', 
            n_iter=100
            ).fit(X_test_new, y_test)
        ext_ = '_boot%s_cv%s_FI.txt' % (b, cv)

        (
        eli5.explain_weights_df(
            perm,
            feature_names = new_df_num.columns.tolist() )
            .sort_values( 'weight', ascending=False )
            .to_csv( os.path.join(out_folder, suffix + ext_) )
        )
        cv += 1

# endregion