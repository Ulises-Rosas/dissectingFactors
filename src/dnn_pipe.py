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


# data --------------------------------------
suffix = 'PROTA'
out_folder = "./../data"
base_path = './../data'

# Features: qcutils results
features_file    = os.path.join(base_path, 'prota_features.tsv')
# Labels: GGI results
all_ggi_results  = os.path.join(base_path, 'joined_1017_two_hypos_prota.txt' )
# data --------------------------------------


self = Post_ggi(
    feature_file = features_file,
    all_ggi_results = all_ggi_results,
)

new_df = self.features
ggi_pd = pd.DataFrame( self.ggi_df[1:], columns=self.ggi_df[0]   )

all_labels, new_df = make_au_labels( ggi_pd, new_df )
new_df_num = new_df.drop(["aln_base"], axis = 1)


########## iteration parameters ###################
max_trials = 200 # bayesian optimization
n_epochs = 1500 
boots = 5 
######## iteration parameters ###################

# testing params
# max_trials = 2
# n_epochs = 150
# boots = 1


################### hyperparameter tuning ###################
# region
all_labels_dis = np.argmax( all_labels, axis=1 ) == 0

split = StratifiedShuffleSplit(n_splits = 1, test_size = 0.35, random_state = 42)
for train_index, test_index in split.split(new_df_num, all_labels_dis):
    # train_index, test_index

    X_train = new_df_num.iloc[train_index,:]
    X_test  = new_df_num.iloc[test_index,:]
    
    y_train = all_labels[train_index]
    y_test  = all_labels[test_index]

X_train_new = transform_data(X_train, X_train)
X_test_new  = transform_data(X_train, X_test)

resampled_features, resampled_labels = do_resampling_dis(X_train_new, y_train)

# K = keras.backend
encoder_weights = []


import keras_tuner

def build_model(hp):
    model = keras.Sequential()
    model.add( keras.layers.InputLayer( input_shape = resampled_features.shape[1] ) )

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

tuner = keras_tuner.BayesianOptimization(
    hypermodel=build_model,
    objective="val_loss",
    max_trials=max_trials,
    # max_trials=10,
    overwrite=True,
    project_name="GAAA", # random folder name
)

early_stopping_cb = keras.callbacks.EarlyStopping('val_loss', patience =100, restore_best_weights=True, mode = 'min')

tuner.search(
    x = resampled_features,
    y = resampled_labels,
    epochs=n_epochs,
    validation_data=(
        X_test_new, 
        y_test, 
    ),
    callbacks=[
        early_stopping_cb,
        # onecycle
    ],
)

# import time
sele_model = tuner.get_best_models()[0]
loss,cos_simi = sele_model.evaluate( X_test_new, y_test)

o_name_base = f"tuner_E{round(loss,6)}_S{round(cos_simi,6)}_ID{int(time.time())}_encoder_{suffix}"
o_name = os.path.join( out_folder, o_name_base )

with open( o_name, 'wb') as f:
    pickle.dump(tuner, f)

print()
print(
f"""
Hyperparameter Test dataset
loss    : {loss}
cos sim : {cos_simi}
"""
)
print()

loss2, cos_simi2 = sele_model.evaluate(resampled_features, resampled_labels)
print(
f"""
Hyperparameter Train dataset
loss  : {loss2}
cos sim : {cos_simi2}
"""
)
print()

myparams = tuner.get_best_hyperparameters()[0].values
print( myparams )


with open(o_name + "_params.txt", 'w') as f:
    f.write( str(myparams) + "\n" )

# endregion
# myparams = {'num_layers': 8, 'drop_0': 0.00015288259146435229, 'units_0': 26, 'drop_1': 0.02365370227603947, 'units_1': 95, 'drop_2': 0.0019439356344310324, 'units_2': 29, 'drop_3': 0.002818964756443786, 'units_3': 59, 'lr': 0.009761241552629777, 'decay': 0.003623615836258005, 'drop_4': 0.0006875387406483257, 'units_4': 68, 'drop_5': 0.07071271931600467, 'units_5': 47, 'drop_6': 0.0001, 'units_6': 5, 'drop_7': 0.0001, 'units_7': 5}

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
        # metrics=[ tf.keras.metrics.CosineSimilarity(axis=1) ]s
    )
    return model

def cosine_similarity(a,b):
    # l2-normalization
    a = ( a.T/np.linalg.norm(a, 2, axis = 1) ).T
    b = ( b.T/np.linalg.norm(b, 2, axis = 1) ).T

    return np.mean( np.sum(a*b, axis = 1) )



early_stopping_cb = keras.callbacks.EarlyStopping('val_loss', patience = 100, restore_best_weights=True, mode = 'min')

# prota
lowest_loss = float('+Inf')
# n_epochs = 5
cvscores = []
for b in range(boots):
    sys.stdout.write(f'\n\nboot: {b}\n')

    kfold = StratifiedKFold(n_splits=4, shuffle=True, random_state=None)
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
            workers=6,
            use_multiprocessing=True,
            verbose=1
        )
        # evaluate the model

        y_pred_rf = model.predict(X_test_new)
        mse       = np.mean( (y_pred_rf - y_test)**2 )

        cos_sim   = cosine_similarity(y_test, y_pred_rf)


        sys.stdout.write("\n\033[92m%s: %.2f\033[0m\n" % ('MSE', mse))
        sys.stdout.write("%s: %.2f\n"                  % ('simi', cos_sim))
        sys.stdout.flush()

        if lowest_loss > mse:

            o_base = f"model_E{round(mse,6)}_S{round(cos_sim,6)}_ID{int(time.time())}_{suffix}"
            o_name = os.path.join(out_folder, o_base)
            model.save(o_name)
            lowest_loss = mse

        cvscores.append([ mse, cos_sim ])

cvscores = np.array(cvscores)
print( "\033[92m%.3f (+/- %.3f)\033[0m" % (np.mean(cvscores[:,0]), np.std(cvscores[:,0])) )

np.savetxt(
    os.path.join( out_folder, '%s_scores_DNN.csv' % suffix ),
    cvscores,
    delimiter=','
)
# endregion

