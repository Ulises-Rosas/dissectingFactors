import os
import sys
import time
import csv
import random

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


import keras_tuner
import numpy as np 
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import  StratifiedShuffleSplit
import tensorflow as tf
from tensorflow import keras


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


def do_resampling_dis(X_train_new, y_train):
    # train_num2.shape
    # tree id 1
    train_labels = np.argmax(y_train, axis=1) == 0

    pos_features = X_train_new[ train_labels]
    neg_features = X_train_new[~train_labels]
    

    pos_labels = y_train[ train_labels]
    neg_labels = y_train[~train_labels]


    if len(pos_features) < len(neg_features):

        ids = np.arange(len(pos_features))

        # taking as much as neg features are
        # available
        choices = np.random.choice(ids, len(neg_features)) 

        pos_features  = pos_features[choices]
        pos_labels    = pos_labels[choices]

    if len(pos_features) > len(neg_features):

        ids = np.arange(len(neg_features))

        # taking as much as pos features are
        # available
        choices = np.random.choice(ids, len(pos_features)) 

        neg_features = neg_features[choices]
        neg_labels   = neg_labels[choices]

    # res_pos_features.shape
    resampled_features = np.concatenate([pos_features, neg_features], axis=0)
    resampled_labels   = np.concatenate([pos_labels, neg_labels], axis=0)

    order = np.arange(len(resampled_labels))
    
    np.random.shuffle(order)

    resampled_features = resampled_features[order]
    resampled_labels   = resampled_labels[order]

    return (resampled_features, 
            resampled_labels  ,)


def _scaler(ref, dat, include_clip = True):

    from sklearn.preprocessing import StandardScaler

    standarizer = StandardScaler().fit(ref)

    sted = standarizer.transform(dat)

    if include_clip:
        return np.clip(sted, -5, 5)
    else:
        return sted

def ordinal(dat):
    from sklearn.preprocessing import OrdinalEncoder
    oe = OrdinalEncoder()
    return oe.fit_transform(dat)

def transform_data(ref, dat, bucket_list = None):

    # ref, dat = train_num, train_num
    if bucket_list:
        # bucket_list = ['gc_mean_pos2',]

        # complete name
        cl = [i + "_bucket" for i in bucket_list]

        ref_num = ref.drop(cl, axis = 1)
        dat_num = dat.drop(cl, axis = 1)

        if not len(dat_num.columns):
            return ordinal(dat)

        dat_scaled = _scaler(ref_num, dat_num, True)
        
        return np.append(dat_scaled, ordinal(dat[cl]), axis = 1)

    else:
        return _scaler(ref, dat, True)


def make_au_labels(ggi_pd, set_pd):
    # ggi_pd,set_pd = ggi_pd, new_df
    ggi_pd['tree_id'] = ggi_pd['tree_id'].astype(int  )
    ggi_pd['au_test'] = ggi_pd['au_test'].astype(float)

    out_labels = []
    has_ggi = []
    
    for seq in set_pd['aln_base']:
        # seq
        tmp_df = ggi_pd[ ggi_pd['alignment'] == seq ]

        if len(tmp_df) < 2:
            has_ggi += [False]
            continue

        has_ggi += [True]

        out_labels.append(

            tmp_df[['tree_id', 'au_test']]
                .sort_values('tree_id')
                ['au_test']
                .tolist()
        )

    labs_2d = np.array(out_labels)
    return labs_2d,set_pd.iloc[has_ggi,:]



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


# read tsv features_file file
feature_names = []
joined_df = {}
n_trees = 2

with open(features_file, 'r') as f:
    rd = csv.reader(f, delimiter = '\t')
    for i,row in enumerate(rd):
        if i == 0:
            feature_names = row
        else:
            joined_df[row[0]] = row[1:]

au_data = {k:[0]*n_trees for k in joined_df.keys() }

with open(all_ggi_results, 'r') as f:
    rd = csv.reader(f, delimiter = '\t')
    for i,row in enumerate(rd):
        if i == 0:
            continue

        if row[0] not in au_data:
            continue

        tree_id = int(row[1])
        au_test = float(row[4])

        au_data[row[0]][tree_id - 1] = au_test

au_data = {k:au_data[k] for k in au_data if sum(au_data[k]) > 0}

# intersection between au_data and joined_df
intersection = set(au_data.keys()).intersection(joined_df.keys())

X = np.zeros((len(intersection), len(feature_names)-1))
y = np.zeros((len(intersection), n_trees))

for i,seq in enumerate(intersection):
    # print(i, seq)
    X[i,:] = np.array(joined_df[seq]).astype(float)
    y[i,:] = au_data[seq]









def read_features(features_file):
    return pd.read_csv(features_file, sep = '\t')

def read_ggi_df(all_ggi_results):

    if not all_ggi_results:
        return None

    ggi_rows = []
    with open(all_ggi_results, 'r') as f:
        reader = csv.reader(f, delimiter = '\t')
        for row in reader:
            ggi_rows.append(row)

    return ggi_rows




new_df = read_features(features_file)
ggi_df = read_ggi_df(all_ggi_results)

ggi_pd = pd.DataFrame( ggi_df[1:], columns=ggi_df[0]   )

all_labels, new_df = make_au_labels( ggi_pd, new_df )
new_df_num = new_df.drop(["aln_base"], axis = 1)
all_labels_dis = np.argmax( all_labels, axis=1 ) == 0



# # load hyperparameters
# with open(hyper_file, 'r') as f:
#     myparams = eval(f.readline().strip())


########## iteration parameters ###################
# max_trials = 50
# n_epochs = 1500
# boots = 20
######## iteration parameters ###################
# tests
max_trials = 5
n_epochs = 100
boots = 2



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
    project_name="keras_hypopt", # random folder name
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
    # workers=20,
    batch_size = 32,
)

# import time
sele_model = tuner.get_best_models()[0]
loss,cos_simi = sele_model.evaluate( X_test_new, y_test)

o_name_base = f"tuner_E{round(loss,6)}_S{round(cos_simi,6)}_ID{int(time.time())}_encoder_{suffix}"
o_name = os.path.join( out_folder, o_name_base )
import json
with open( o_name, 'w') as f:
    json.dump( tuner.get_best_hyperparameters()[0].values, f)

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

# read o_name
with open(o_name, 'r') as f:
    myparams = eval(f.readline().strip())




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