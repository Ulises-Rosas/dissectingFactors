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
n_epochs_encoder = 100000
max_trials = 200
n_epochs = 1500
boots = 5
######## iteration parameters ###################
# testing params
# n_epochs_encoder = 3
# max_trials = 2
# n_epochs = 5
# boots = 2


######## encoder ##########

# region
X_all = transform_data(new_df_num, new_df_num)

K = keras.backend
kl_divergence = keras.losses.kullback_leibler_divergence

class Sampling(keras.layers.Layer):
    def call(self, inputs):
        mean, log_var = inputs
        return K.random_normal(tf.shape(log_var)) * K.exp(log_var / 2) + mean 

tf.random.set_seed(42)
np.random.seed(42)
codings_size = 10


n_fea = X_all.shape[-1]

inputs = keras.layers.Input(shape= n_fea )
z = keras.layers.Dense(35, activation="selu", name = 'encoder_1')(inputs)
z = keras.layers.Dense(30, activation="selu", name = 'encoder_2')(z)
z = keras.layers.Dense(25, activation="selu", name = 'encoder_3')(z)
z = keras.layers.Dense(20, activation="selu", name = 'encoder_4')(z)
codings_mean = keras.layers.Dense(codings_size)(z)
codings_log_var = keras.layers.Dense(codings_size)(z)
codings = Sampling()([codings_mean, codings_log_var])
variational_encoder = keras.models.Model(
    inputs=[inputs], outputs=[codings_mean, codings_log_var, codings]
)
decoder_inputs = keras.layers.Input(shape=[codings_size])
x = keras.layers.Dense(20, activation="selu")(decoder_inputs)
x = keras.layers.Dense(25, activation="selu")(x)
x = keras.layers.Dense(30, activation="selu")(x)
x = keras.layers.Dense(35, activation="selu")(x)
outputs = keras.layers.Dense( n_fea, activation="selu" )(x)
variational_decoder = keras.models.Model(inputs=[decoder_inputs], outputs=[outputs])

_, _, codings = variational_encoder(inputs)
reconstructions = variational_decoder(codings)
variational_ae = keras.models.Model(inputs=[inputs], outputs=[reconstructions])

latent_loss = -0.5 * K.sum(
    1 + codings_log_var - K.exp(codings_log_var) - K.square(codings_mean),
    axis=-1)

variational_ae.add_loss(K.mean(latent_loss) / n_fea )

variational_ae.compile(
    # loss="binary_crossentropy",
     optimizer="rmsprop", 
    # metrics=[rounded_accuracy]
    loss="mse",
    # optimizer = optimizer,
    metrics=[tf.keras.metrics.MeanSquaredError()]
    )

history = variational_ae.fit(
    X_all, X_all, epochs=n_epochs_encoder,
    workers=10, use_multiprocessing=not False,
)

myloss,mymse = variational_ae.evaluate(X_all, X_all)

print()
print(
    myloss, mymse
)

# variational_encoder.layers
# variational_encoder.summary()
WB_TREE0 = variational_encoder.layers[1].get_weights()
WB_TREE1 = variational_encoder.layers[2].get_weights()
WB_TREE2 = variational_encoder.layers[3].get_weights()
WB_TREE3 = variational_encoder.layers[4].get_weights()


o_name_base = f"{suffix}_all_vars_L{round(myloss, 6)}_E{round(mymse, 6)}_4Layers.pkl" 
o_name = os.path.join(out_folder, o_name_base)
with open(o_name, 'wb') as f:
    pickle.dump((WB_TREE0, WB_TREE1, WB_TREE2, WB_TREE3), f)

# endregion


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

encoder_weights = [
    WB_TREE0, 
    WB_TREE1,
    WB_TREE2,
    WB_TREE3,
]


import keras_tuner

def build_model(hp):
    model = keras.Sequential()
    model.add( keras.layers.InputLayer( input_shape = resampled_features.shape[1] ) )

    model.add( keras.layers.Dense( units = 35 ) )
    model.layers[ 0 ].set_weights( encoder_weights[0] )
    model.layers[ 0 ].trainable = not False

    model.add( keras.layers.Dense( units = 30 ) )
    model.layers[ 1 ].set_weights( encoder_weights[1] )
    model.layers[ 1 ].trainable = not False

    model.add( keras.layers.Dense( units = 25 ) )
    model.layers[ 2 ].set_weights( encoder_weights[2] )
    model.layers[ 2 ].trainable = not False

    model.add( keras.layers.Dense( units = 20 ) )
    model.layers[ 3 ].set_weights( encoder_weights[3] )
    model.layers[ 3 ].trainable = not False

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



##########   CROSS-VALIDATION ###########

print('starting cross validation')

# region

def build_model(params, input_shape):

    model = keras.Sequential()
    model.add( keras.layers.InputLayer(input_shape = input_shape) )

    model.add( keras.layers.Dense( units = 35 ) )
    model.layers[ 0 ].set_weights( encoder_weights[0] )
    model.layers[ 0 ].trainable = not False

    model.add( keras.layers.Dense( units = 30 ) )
    model.layers[ 1 ].set_weights( encoder_weights[1] )
    model.layers[ 1 ].trainable = not False

    model.add( keras.layers.Dense( units = 25 ) )
    model.layers[ 2 ].set_weights( encoder_weights[2] )
    model.layers[ 2 ].trainable = not False

    model.add( keras.layers.Dense( units = 20 ) )
    model.layers[ 3 ].set_weights( encoder_weights[3] )
    model.layers[ 3 ].trainable = not False


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
    decay_rate = params['decay']
    
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
lowest_loss = float('+Inf')
# n_epochs = 5

cvscores = []

for b  in range(boots):
    
    sys.stdout.write(f'\n\nboot: {b}\n')

    kfold = StratifiedKFold(n_splits=4, shuffle=True, random_state=None)
    for train, test in kfold.split( new_df, all_labels_dis ):
        # train,test
        # len(test)/len(train)

        X_train = new_df_num.iloc[train,:]
        y_train = all_labels[train]

        X_test = new_df_num.iloc[test,:]
        y_test = all_labels[test]

        X_train_new = transform_data(X_train, X_train)
        X_test_new  = transform_data(X_train, X_test)

        resampled_features, resampled_labels = do_resampling_dis(X_train_new, y_train)

        model = build_model(myparams, 
                            input_shape = resampled_features.shape[-1])

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
        sys.stdout.flush()

        if lowest_loss > loss:

            o_base = f"model_E{round(loss,6)}_S{round(cos_simi,6)}_ID{int(time.time())}_{suffix}"
            o_name = os.path.join(out_folder, o_base)
            model.save(o_name)
            lowest_loss = loss

        cvscores.append([loss,cos_simi])

cvscores = np.array(cvscores)
print("\033[92m%.3f (+/- %.3f)\033[0m" % (np.mean(cvscores[:,0]), np.std(cvscores[:,0])))

np.savetxt(
    os.path.join(out_folder, '%s_scores_rf.csv' % suffix), 
    cvscores,
    delimiter=','
)
# endregion