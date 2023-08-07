import os
import sys

import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


import pandas as pd
from recurrent_utils import *

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


# discrete labels
all_labels_dis  = np.argmax( all_labels, axis=1) == 0

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

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV,StratifiedKFold

# from sklearn.metrics import mean_squared_error, pairwise
# import tensorflow as tf

param_grid = {
    'max_depth': np.linspace(5, 100, 5,  dtype=int),
    'min_samples_split': np.linspace(2, 50, 5,  dtype=int),
    'max_leaf_nodes': np.linspace(100, 250, 5,  dtype=int) 
}

n_jobs = 4
n_estimators = 500

base_estimator = RandomForestRegressor( n_estimators=n_estimators, n_jobs=n_jobs, verbose=1 )
rsearch = RandomizedSearchCV(base_estimator, param_grid, cv = 2).fit(resampled_features, resampled_labels)

best_rf_params = rsearch.best_params_

params_file = os.path.join(out_folder, 'rf_params_%s.txt' % suffix)

with open(params_file, 'w') as f:
    f.write(  str(best_rf_params) + "\n" )

# y_pred_rf = rsearch.best_estimator_.predict(X_test_new)

def cosine_similarity(a,b):
    # l2-normalization
    a = ( a.T/np.linalg.norm(a, 2, axis = 1) ).T
    b = ( b.T/np.linalg.norm(b, 2, axis = 1) ).T

    return np.mean( np.sum(a*b, axis = 1) )


lowest_loss = float('+Inf')
# n_epochs = 15
boots = 5
cvscores = []
for b in range(boots):
    # b = 1

    kfold = StratifiedKFold(n_splits=4, shuffle=True, random_state=None)
    sys.stdout.write(f'\n\nboot: {b}\n')

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

        # from sklearn.linear_model import LinearRegression
        # reg = LinearRegression().fit(resampled_features, resampled_labels)
        # reg.predict( X_test_new )

        base_estimator = RandomForestRegressor(
                            n_estimators=n_estimators, 
                            n_jobs=n_jobs, 
                            verbose=0, 
                            **best_rf_params 
                        )
        base_estimator.fit(resampled_features, resampled_labels)
        y_pred_rf = base_estimator.predict(X_test_new)

        mse     = np.mean( (y_pred_rf - y_test)**2 )
        cos_sim = cosine_similarity(y_test, y_pred_rf)

        # evaluate the model
        sys.stdout.write("\033[92m%s: %.2f\033[0m\n" % ("mse", mse))
        sys.stdout.write("%s: %.2f\n"                % ("cos_sim", cos_sim))
        sys.stdout.flush()
        # if lowest_loss > mse:
        #     o_base = f"model_E{round(mse,6)}_S{round(cos_sim,6)}_ID{int(time.time())}"
        #     o_name = os.path.join(the_path, o_base)
        #     model.save(o_name)

        #     lowest_loss = mse
        cvscores.append([mse,cos_sim])

cvscores = np.array(cvscores)
print("\033[92m%.3f (+/- %.3f)\033[0m" % (np.mean(cvscores[:,0]), np.std(cvscores[:,0])))

np.savetxt(
    os.path.join(out_folder, '%s_scores_rf.csv' % suffix), 
    cvscores,
    delimiter=','
)


