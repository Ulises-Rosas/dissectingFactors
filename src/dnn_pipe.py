import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# import csv
import sys
import numpy as np 
from sklearn.model_selection import StratifiedKFold

import pandas as pd
import pickle
from recurrent_utils import *

import tensorflow as tf
from tensorflow import keras
import time

# debugging --------------------------------------
import os
# base_path = '/condo/fishevolab/urosas/McL/data/prota'
suffix     = 'FLAT'
# out_folder = "/condo/fishevolab/urosas/McL/DNN/flat"
out_folder = '/Users/ulises/Desktop/GOL/software/GGpy/comparing_models/flat'

max_trials = 200
n_epochs = 1500
boots = 5

# n_epochs_encoder = 3
# max_trials = 2
# n_epochs = 5
# boots = 2

# encoder_file = '/condo/fishevolab/urosas/McL/DNN_Autoencoder/prota/prota_all_vars_L0.356052_E0.182115_4Layers.pkl'
# out_folder = "/condo/fishevolab/urosas/McL/DNN_Autoencoder/prota"

if suffix == 'PROTA':
    base_path = '/Users/ulises/Desktop/GOL/software/GGpy/proofs_ggi/postRaxmlBug/prota/xgboostclass'
    # base_path = '/condo/fishevolab/urosas/McL/data/prota'
    file_comparisons = os.path.join(base_path, 'comparisonkey.txt')
    features_file    = os.path.join(base_path, 'features_prota_features_last.tsv')
    all_ggi_results  = os.path.join(base_path, 'joined_1017_two_hypos_prota.txt' )

else:
    base_path = '/Users/ulises/Desktop/GOL/software/GGpy/proofs_ggi/postRaxmlBug/flatfishes/two_hypo_ASTRAL-ML/version2'
    # base_path = '/condo/fishevolab/urosas/McL/data/flat'

    file_comparisons = os.path.join(base_path, 'comparisonfile_3.txt')
    features_file    = os.path.join(base_path, 'features_991exons_fishlife_2_aln_name.tsv')
    all_ggi_results  = os.path.join(base_path, 'join_ggi_flat_version2.tsv')


# base_path = '/Users/ricardobetancur/Desktop/Ulises/ggpy_tests'
# seq_path = '/Users/ricardobetancur/Desktop/Ulises/ggpy_tests/alns'
self = Post_ggi(
    feature_file = features_file,
    all_ggi_results = all_ggi_results,
    file_comparisons = file_comparisons,
    threads = 6,
)

if suffix == 'PROTA':

    tax_prop = pd.read_csv(
                os.path.join(base_path, 'features_exons1024'),
                sep = "\t"
            )[['aln_base', 'tax_prop']]

    new_df = (
        pd.merge(self.features, 
                tax_prop, 
                on = 'aln_base', 
                how='left').query('tax_prop == 1.').drop('tax_prop', axis=1) 
    )
else:
    new_df = self.features

# np.abs(np.array([-1,1,-2]))
for c in self.drop_columns:
    try:
        new_df = new_df.drop( labels = c, axis = 1 )

    except KeyError:
        pass

# new_df = new_df.drop(tree_vars, axis=1)
added_features = os.path.join(base_path, 'new_features_%s.csv' % suffix) # for flat
# added_features  = os.path.join(base_path, 'new_features_PROTA.csv') # for prota

new_df = correct_flat_features(
            added_features ,
            new_df,
            filtering_features=None
        )
ggi_pd = pd.DataFrame( self.ggi_df[1:], columns=self.ggi_df[0]   )

all_labels, new_df = make_au_labels( ggi_pd, new_df )
new_df_num = new_df.drop(["aln_base"], axis = 1)


################# data viz
def dim_red(X, n_com):
    # extracting averages
    X_centered = (X - np.mean(X, axis=0))

    # U,E,Vt (SVD)
    _,E,Vt = np.linalg.svd(X_centered)
    # print(E)
    # PCA components
    W = Vt.T[:, :n_com]
    # projection into n_com dimensions
    return X_centered.dot(W)

def gls_errors(X, y, pseudo_inv = False):

    if pseudo_inv:
        B_gls = np.linalg.pinv(X.T.dot(X)).dot(X.T.dot(y))

    else:
        B_gls = np.linalg.inv(X.T.dot(X)).dot(X.T.dot(y))

    return X.dot(B_gls) - y

def test_normality_linear(X, y, pseudo_inv = False):
    from scipy import stats

    errors = gls_errors(X, y, pseudo_inv=pseudo_inv).flatten()
    res = stats.normaltest(errors)
    return res.pvalue,errors

def standard_scale(u):
    return (u - np.mean(u, axis=0))/np.std( u, axis= 0)

def plot_errors(X, y, pseudo_inv =  False, bins = 30):

    pval,errors = test_normality_linear(X, y, pseudo_inv = pseudo_inv)

    fig, ax = plt.subplots()
    # the histogram of the data
    ax.hist(errors, bins, density=True)
    ax.set_title('p-value = %s' % pval)
    plt.show()

X = standard_scale(new_df_num.to_numpy())
plot_errors(X,all_labels, bins = 20, pseudo_inv= True)


# errors = gls_errors(X, all_labels, pseudo_inv=True).flatten()

# test_normality_linear(X, all_labels, pseudo_inv = True)

# num_bins = 30
# fig, ax = plt.subplots()

# # the histogram of the data
# ax.hist(errors, num_bins, density=True)
# ax.set_title('P-val: %s' % 1)



# from sklearn.decomposition import PCA
# pca = PCA(n_components=1, svd_solver='full')
# pca.fit(X.T)
# pca.explained_variance_ratio_


# X1D = dim_red(X, 1)

# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# ax.scatter(X1D, all_labels[:,0], all_labels[:,1],)
# ax.view_init(30, 120 )

# # X1D = dim_red(X, 1)
# import matplotlib.pyplot as plt
# plt.scatter(X1D[:,0], X1D[:,1])

#################



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

tuner = keras_tuner.BayesianOptimization(
    hypermodel=build_model,
    objective="val_loss",
    # max_trials=max_trials,
    max_trials=10,
    overwrite=True,
    project_name="GAAA",
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

myparams = {'num_layers': 8, 'drop_0': 0.00015288259146435229, 'units_0': 26, 'drop_1': 0.02365370227603947, 'units_1': 95, 'drop_2': 0.0019439356344310324, 'units_2': 29, 'drop_3': 0.002818964756443786, 'units_3': 59, 'lr': 0.009761241552629777, 'decay': 0.003623615836258005, 'drop_4': 0.0006875387406483257, 'units_4': 68, 'drop_5': 0.07071271931600467, 'units_5': 47, 'drop_6': 0.0001, 'units_6': 5, 'drop_7': 0.0001, 'units_7': 5}

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
    
    optimizer = keras.optimizers.SGD(
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

