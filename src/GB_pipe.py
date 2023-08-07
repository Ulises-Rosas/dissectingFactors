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


from sklearn.model_selection import RandomizedSearchCV,StratifiedKFold
from scipy.stats import expon, reciprocal
import xgboost


param_grid = {
    # 'gamma': np.linspace(0.001, 10, 100),
    'max_depth'       : np.linspace(3, 15, 5,dtype=int),
    'subsample'       : np.linspace(0.1,  1 , 20),
    'gamma'           : reciprocal(1, 10),
    'min_child_weight': np.linspace(0.1,  10, 20),
    'learning_rate'   : expon(scale = 1.0),
}

n_jobs = 6

base_estimator = xgboost.XGBRegressor(
    tree_method="hist",
    n_estimators = 500 # number of weak learners
    )
rsearch = (
    RandomizedSearchCV(
        base_estimator,
          param_grid, cv = 2,
            n_iter=1000, # random rounds
            n_jobs=n_jobs)
    .fit( 
        resampled_features, 
        resampled_labels,
        early_stopping_rounds = 100,
        eval_set = [(X_test_new, y_test)] 
        )
)

best_rf_params = rsearch.best_params_


params_file = os.path.join(out_folder, 'xgboost_params_%s.txt' % suffix)

with open(params_file, 'w') as f:
    f.write(  str(best_rf_params) + "\n" )


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

        base_estimator =  xgboost.XGBRegressor(
                            **best_rf_params,
                            tree_method="hist",
                            n_estimators = 500,
                            ) 
        
        # i = 0
        base_estimator.fit(
            resampled_features, 
            # resampled_labels[:,i],
            resampled_labels,
            early_stopping_rounds = 100,
            eval_set = [(X_test_new, y_test)] 
            )

        y_pred_rf = base_estimator.predict(X_test_new)
        mse       = np.mean( (y_pred_rf - y_test)**2 )
        cos_sim   = cosine_similarity(y_test, y_pred_rf)

        # evaluate the model
        sys.stdout.write("\033[92m%s: %.2f\033[0m\n" % ("mse", mse))
        sys.stdout.write("%s: %.2f\n"                % ("cos_sim", cos_sim))
        sys.stdout.flush()
        cvscores.append([mse,cos_sim])

cvscores = np.array(cvscores)

print("\033[92m%.3f (+/- %.3f)\033[0m" % (np.mean(cvscores[:,0]), np.std(cvscores[:,0])))

np.savetxt(  
    os.path.join(out_folder, '%s_scores_xgboost.csv' % suffix), 
    cvscores,
    delimiter=','
)

