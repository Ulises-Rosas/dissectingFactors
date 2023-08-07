import numpy as np
from sklearn.metrics import  accuracy_score, plot_confusion_matrix
from sklearn.model_selection import  StratifiedShuffleSplit

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import csv
import sys

import random

import collections

import pickle
# import xgboost

import numpy as np # downloaded as shap dependency
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from sklearn.metrics import  accuracy_score, plot_confusion_matrix


from itertools import combinations

import keras






# joined_all_ranks = pd.merge(
#     new_df,
#     ggi_pd.rename(columns={'alignment':'aln_base'}),
#     on = 'aln_base',
#     how = 'left'
#     )

# joined_all_ranks['seq_len_quantile'] = pd.qcut(joined_all_ranks['seq_len'],q = 10)

# mpl.rcParams['figure.figsize'] = (10, 5)
# joined_all_ranks[['au_test', 'seq_len_quantile', 'tree_id', 'rank']].query('tree_id == 2').boxplot( column = 'au_test',by = ['seq_len_quantile','rank'])
# plt.tick_params(axis='x', labelrotation=90)
# plt.title("Prota H2")
# plt.ylabel("p-AU")
# plt.tight_layout()
# plt.axhline(y=0.05, color='r', linestyle=':')
# plt.axhline(y=0.5, color='r', linestyle=':')
# plt.axhline(y=0.95, color='r', linestyle=':')
# plt.savefig('/Users/ulises/Desktop/GOL/software/GGpy/ggpy/prota_h2_ranks.png', dpi = 330)
# plt.close()




# ggi_rk1 = ggi_pd.rename(columns={'alignment':'aln_base'}).query('rank == "1"')
# # new_df

# new_joined = pd.merge(new_df, ggi_rk1, on = 'aln_base', how = 'left' )
# new_joined['au_test'] = new_joined['au_test'].astype(float)
# bins = [i for i in range(0, 2500, 350)] + [2500]
# labels = [f"{bins[i]}-{bins[i + 1]}" for i in range(len(bins) - 1) ]
# new_joined['seq_len_cat'] = pd.cut(new_joined['seq_len'],
#                                 bins   = bins,
#                                 labels = labels
#                                 )
# new_joined.groupby('seq_len_cat').apply(len)

# new_joined[['au_test','seq_len_cat']].boxplot(by = 'seq_len_cat')
# plt.tick_params(axis='x', labelrotation=90)



# new_joined['seq_len_quantile'] = pd.qcut(new_joined['seq_len'],10)
# new_joined.groupby('seq_len_quantile').apply(len)

# bin_size = new_joined.groupby('seq_len_quantile').apply(len).mean()

# mpl.rcParams['figure.figsize'] = (7, 5)
# new_joined[['au_test','seq_len_quantile']].boxplot(by = 'seq_len_quantile')
# plt.tick_params(axis='x', labelrotation=90)
# plt.title("Prota rank 1, average bin size: %s" % bin_size)
# plt.ylabel("p-AU")
# plt.tight_layout()
# plt.axhline(y=0.5, color='r', linestyle='dashed')
# plt.axhline(y=0.95, color='r', linestyle='dashed')
# plt.savefig('/Users/ulises/Desktop/GOL/software/GGpy/ggpy/prota_rank1_seq.png', dpi = 330)
# plt.close()
# plt.gca().set_ylim(0, 1) # set the vertical range to [0-1]



METRICS = [
      keras.metrics.TruePositives(name='tp'),
      keras.metrics.FalsePositives(name='fp'),
      keras.metrics.TrueNegatives(name='tn'),
      keras.metrics.FalseNegatives(name='fn'), 
      keras.metrics.BinaryAccuracy(name='accuracy'),
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall'),
      keras.metrics.AUC(name='auc'),
      keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
]


def plot_metrics(history):
    import matplotlib as mpl

    mpl.rcParams['figure.figsize'] = (12, 10)
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    metrics = ['loss', 'prc', 'accuracy', 'recall',]
    for n, metric in enumerate(metrics):
        name = metric.replace("_"," ").capitalize()
        plt.subplot(2,2,n+1)
        plt.plot(history.epoch, history.history[metric], color=colors[0], label='Train')
        plt.plot(history.epoch, history.history['val_'+metric],
                color=colors[0], linestyle="--", label='Val')
        plt.xlabel('Epoch')
        plt.ylabel(name)
        if metric == 'loss':
            plt.ylim([0, plt.ylim()[1]])
        elif metric == 'auc':
            plt.ylim([0.8,1])
        else:
            plt.ylim([0,1])

        plt.legend()



def plot_cm(labels, predictions, p=0.5, onehotter = False):
    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    if onehotter:
        pred_int = np.where(np.round(predictions) == 1)[1]
        obs_int = np.where(labels == 1)[1]

        cm = confusion_matrix(obs_int, pred_int)

        pass
    else:
        cm = confusion_matrix(labels, predictions > p)

    plt.figure(figsize=(5,5))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title('Confusion matrix @{:.2f}'.format(p))
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')

    print('Legitimate Transactions Detected (True Negatives): ', cm[0][0])
    print('Legitimate Transactions Incorrectly Detected (False Positives): ', cm[0][1])
    print('Fraudulent Transactions Missed (False Negatives): ', cm[1][0])
    print('Fraudulent Transactions Detected (True Positives): ', cm[1][1])
    print('Total Fraudulent Transactions: ', np.sum(cm[1]))



seq_vars = [
 'nheaders',
 'pis',
 'vars',
 'seq_len',
 'seq_len_nogap',
 'gap_prop',
 'nogap_prop',
 'gc_mean',
 'gc_var',
 'gap_mean',
 'gap_var',
 'pi_mean',
 'pi_std',
 'rcv',
 'gc_mean_pos1',
 'gc_var_pos1',
 'gap_mean_pos1',
 'gap_var_pos1',
 'gc_mean_pos2',
 'gc_var_pos2',
 'gap_mean_pos2',
 'gap_var_pos2',
 'gc_mean_pos3',
 'gc_var_pos3',
 'gap_mean_pos3',
 'gap_var_pos3',
 'saturation',
 'treeness_o_rcv']

tree_vars = [
 'LB_std',
 'coeffVar_len',
 'inter_len_mean',
 'inter_len_var',
 'saturation',
 'supp_mean',
 'ter_len_mean',
 'ter_len_var',
 'total_tree_len',
 'treeness',
 'treeness_o_rcv']

class Post_ggi:

    def __init__(self, 
                 feature_file = None,
                 all_ggi_results = None,
                 file_comparisons = None,
                 model_prefix = "post_ggi",
                 max_display = 17,
                 cnfx_ncols = 3,
                 threads = 1,
                 seq_path = '',
                 ) -> None:

        self.feature_file = feature_file
        self.all_ggi_results = all_ggi_results
        self.file_comparisons = file_comparisons

        self.model_prefix = model_prefix
        self.cnf_ncols = cnfx_ncols
        self.threads = threads

        self.seq_path = seq_path

        self.drop_columns = [
            'Group', 
            # 'aln_base', 
            'SymPval', 
            'MarPval', 
            'IntPval'
        ]

        # self.metadata_columns = [
        #     'model_filename',
        #     'accuracy',
        #     'pos_hypo',
        #     'neg_hypo'
        # ]

        self.metadata_columns = [
            'pos_hypo',
            'neg_hypo',
            'accuracy',
        ]

        self.max_display = max_display

        # previously tuned hyperparameters
        self.gamma = 0.14210526315789473
        self.learning_rate = 0.03
        self.max_depth = 4
        self.reg_lambda = 0.018
        self.n_estimators = 4200

    @property    
    def ggi_df(self):

        if not self.all_ggi_results:
            return None

        ggi_rows = []
        with open(self.all_ggi_results, 'r') as f:
            reader = csv.reader(f, delimiter = '\t')
            for row in reader:
                ggi_rows.append(row)

        return ggi_rows

    @property
    def tree_id_comp(self):

        if not self.file_comparisons:
            return None

        tree_id_comp = []
        with open(self.file_comparisons, 'r') as f:
            reader = csv.reader(f, delimiter = ',')
            for row in reader:
                tree_id_comp.append(row)

        return tree_id_comp

    @property
    def features(self):
        return pd.read_csv(self.feature_file, sep = '\t')

    def subset_tree_id(self, tree_id_comp1):
        """
        subset tree_id_comp df
        by a tree_id
        """
        ha,hb = tree_id_comp1

        if not self.ggi_df:
            return None

        subset_df = []
        for row in self.ggi_df:

            rank = row[3]
            if rank != '1':
                continue

            aln_base = row[0]
            tree_id  = row[1]

            # contained = set(tree_id_comp1) & set([tree_id])

            if tree_id == ha or tree_id == hb:
                subset_df.append([aln_base, tree_id])

        if not subset_df:
            sys.stderr.write("Tree ids '%s' do not match with\n" % ", ".join(tree_id_comp1))
            sys.stderr.write("'%s' data frame\n" % self.all_ggi_results)
            sys.stderr.flush()
            return None

        return pd.DataFrame(subset_df, columns = ['alignment', 'hypothesis'])
        
    def make_specific_prefix(self, tree_id_comp1):
        return "%s_h%s_h%s" % (self.model_prefix, tree_id_comp1[0], tree_id_comp1[1])

    def _bar_data(self, shap_values, all_num):

        max_display = self.max_display
        shap_means  = np.abs(shap_values).mean(0)
        mysort      = np.argsort(shap_means)[::-1]
        
        y_axis = all_num.columns[mysort][:max_display][::-1]
        x_axis = shap_means[mysort][:max_display][::-1]

        return y_axis, x_axis

    def _my_dependency_plot(self, all_num, lead, shap_values):

        from shap.plots._scatter import dependence_legacy
        import numpy as np
        import matplotlib.colors as mcolors
        from matplotlib import cm


        tra = shap_values.values.T

        lead = 'supp_mean'
        # layer = 'supp_mean'
        for layer in all_num.columns:

            if layer == lead:
                continue

            ind = np.argwhere( all_num.columns == lead  ).flatten()[0]
            interaction_index = np.argwhere( all_num.columns == layer  ).flatten()[0]

            covar = all_num.iloc[:,interaction_index]
            pre_cmap = cm.get_cmap('viridis', len(covar))

            hexes = [ mcolors.rgb2hex( pre_cmap(i)  ) for i in range(len(covar)) ]
            cmap,norm = mcolors.from_levels_and_colors(sorted([0]+list(covar)), hexes)

            plt.scatter(
                all_num.iloc[:,ind],
                tra[ind,:],
                c = list(covar),
                cmap = cmap,
                norm = norm,
                s = 19, alpha = 0.9
            )
            plt.xlabel(lead)
            plt.ylabel('SHAP value')
            cbar = plt.colorbar( cm.ScalarMappable(norm = norm, cmap= cmap) )
            cbar.set_label(layer)
            
            plt.tight_layout(pad=0.05)
            plt.savefig("dependencies_plot/shap_%s_%s.png" % (lead,layer), dpi = 330)
            plt.close()

    def _update_meta_and_bardata(self, shap_values, accuracy, _groups_dict, all_num):
        """
        # update metadata & get bar data
        """
        # model_filename = '%s.sav' % new_prefix
        # self.add_metdata(
        #     self._join_shaps(shap_values, 
        #                      [ model_filename,
        #                        accuracy,
        #                        _groups_dict[True ],
        #                        _groups_dict[False]  ] ) )

        self.add_metdata(
            self._join_shaps(shap_values, 
                             [ _groups_dict[True ],
                               _groups_dict[False],
                               accuracy, ] ), 
            False
        )

        return self._bar_data( shap_values, all_num ) 

    def shap_things(self, xgb_clf_no_nor, all_num, new_prefix, _groups_dict, accuracy):
        
        # heavy imports
        from shap import Explainer
        from shap.plots._beeswarm import summary_legacy
        # heavy imports

        # xgb_clf_no_nor, all_num, new_prefix = xgb_clf_no_nor, all_num, new_prefix

        bee_20_filename = "best%s_beeswarm_%s.png" % (self.max_display,new_prefix)

        explainer     = Explainer(xgb_clf_no_nor, all_num)
        shap_values   = explainer(all_num)

        y_axis,x_axis = self._update_meta_and_bardata( shap_values.values, accuracy, 
                                                       _groups_dict, all_num )
        
        gs = gridspec.GridSpec(1, 2,  width_ratios=[1, 1.7]) 
        # plt.rcParams['font.size'] = 14.0

        plt.subplot( gs[0] )
        plt.barh(
            y_axis, x_axis, height=0.6,
            align  ='center',
            color  = 'gray',
            zorder = 3
        )
        plt.yticks([])
        plt.grid(True, which='major', axis='x')
        plt.gca().invert_xaxis()
        plt.xlabel('mean(|SHAP value|)', fontsize = 13)

        plt.subplot( gs[1] )
        summary_legacy(
            shap_values,
            max_display = self.max_display,
            plot_size   = (self.max_display*0.5, self.max_display*0.3),
            show = False
        )

        plt.title(_groups_dict[True], loc = 'right')
        plt.title(_groups_dict[False], loc = 'left')
        plt.tight_layout(pad=0.05)
        plt.savefig(bee_20_filename, dpi = 330)
        plt.close()

        # delete later
        # np.savetxt("/Users/ulises/Desktop/GOL/software/GGpy/proofs_ggi/postRaxmlBug/prota/prota_shapvalues.csv", shap_values.values, delimiter=",")
        # delete later

    def _join_shaps(self, shap_values, values):

        joined = []
        for row in shap_values:
            joined += [values + list(row)]

        return joined

    def _get_columns(self):
        cols = list(self.features.columns)

        for i in self.drop_columns:
            if cols.__contains__(i):
                cols.remove(i)

        return self.metadata_columns + cols

    def add_metdata(self, values, init):

        metadata_filename = '%s_metadata.tsv' % self.model_prefix

        if init:

            with open(metadata_filename, 'w') as f:
                writer = csv.writer(f, delimiter = "\t")
                writer.writerows( [self._get_columns()] )

        else:

            with open(metadata_filename, 'a') as f:
                writer = csv.writer(f, delimiter = "\t")
                writer.writerows(values)

        sys.stdout.write('\tWritting metadata at: "%s"\n\n' % metadata_filename)
        sys.stdout.flush()
        
    # def xgboost_classifier(self, tree_id_comp1):

    #     # tree_id_comp1 = self.tree_id_comp[0]

    #     if not tree_id_comp1:
    #         return None

    #     dataset = "H%s-H%s" % tuple(tree_id_comp1)
    #     new_prefix = self.make_specific_prefix(tree_id_comp1)
    #     # model_filename = '%s.sav' % new_prefix

    #     sys.stdout.write('Processing: "%s" dataset\n' % dataset)
    #     sys.stdout.flush()

    #     features = self.features
    #     target = self.subset_tree_id(tree_id_comp1)

    #     # merge 
    #     aln_base,hypothesis = target.columns
    #     aln_feature = features.columns[0]

    #     target = target.rename({aln_base: aln_feature}, axis = 1)
    #     merged_dataset = features.merge(target, on = aln_feature, how='left')
    #     new_df = merged_dataset[merged_dataset[hypothesis].notna()].reset_index(drop=True)

    #     for c in self.drop_columns:
    #         try:
    #             new_df = new_df.drop( labels = c, axis = 1 )
    #         except KeyError:
    #             pass

    #     # hypotheses definition
    #     _groups_dict = { 
    #         True  : 'H%s' % tree_id_comp1[0],
    #         False : 'H%s' % tree_id_comp1[1]
    #     }

    #     seqs   = list(new_df[aln_feature])
    #     labels = list(new_df[hypothesis])

    #     out_train, out_test = random_sampling(
    #         labels,
    #         seqs,
    #         seq_path=self.seq_path,
    #         test_size=0.30
    #     )
        
    #     strat_train_set = new_df[new_df[aln_feature].isin(out_train)]
    #     strat_test_set  = new_df[new_df[aln_feature].isin(out_test) ]


    #     strat_test_set.to_csv(
    #         new_prefix + "_features_hypos_test.csv",
    #         sep = ",", 
    #         index = False
    #     )



    #     train_num = strat_train_set.drop([aln_feature, hypothesis], axis = 1)
    #     train_labels = strat_train_set[hypothesis] == tree_id_comp1[0]

    #     test_num = strat_test_set.drop([aln_feature, hypothesis], axis = 1 )
    #     test_labels = strat_test_set[hypothesis] == tree_id_comp1[0]

    #     # labels = list(new_df[hypothesis])

    #     # split = StratifiedShuffleSplit(n_splits = 1, test_size = 0.25, random_state = 42)
    #     # for train_index, _ in split.split(new_df, new_df[hypothesis]):
    #     #     strat_train_set = new_df.loc[train_index]
    #         # strat_test_set  = new_df.loc[test_index]
    #         #         
    #     # train_num = strat_train_set.drop(hypothesis, axis=1)
    #     # train_labels = strat_train_set[hypothesis] == tree_id_comp1[0]
    
    #     all_num  = new_df.drop([aln_feature,hypothesis], axis=1)
    #     all_labels = new_df[hypothesis] == tree_id_comp1[0]

    #     # TODO: delete. Lines only used for testing purposes
    #     new_df.to_csv(
    #         new_prefix + "_features_hypos.csv",
    #          sep = ",", 
    #          index = False
    #     )
    #     # TODO: delete. Lines only used for testing purposes
        
    #     fr_tr = collections.Counter(all_labels)
    #     # fr_tr = collections.Counter(test_labels)

    #     # if len(fr_tr) <= 1:
    #     #     sys.stdout.write("Less than two labels found in the '%s' dataset\n" % dataset)
    #     #     sys.stdout.flush()
    #     #     return None
        
    #     # scale_pos_weight =  abs(1 - fr_tr[False]/fr_tr[True]) # work around
    #     scale_pos_weight =  fr_tr[False]/fr_tr[True]


    #     self.max_depth = 3
    #     self.learning_rate = 0.0003
    #     self.gamma = 0.4
    #     # self.reg_lambda = 0.05
    #     self.reg_lambda = 10

    #     xgb_clf_no_nor = xgboost.XGBClassifier(
    #                         objective = 'binary:logistic',
    #                         subsample = 0.5,
    #                         colsample_bylevel = 0.7,
    #                         colsample_bytree  = 1,
    #                         colsample_bynode  = 0.7,
    #                         min_child_weight  = 6.66,


    #                         use_label_encoder = not True,
    #                         scale_pos_weight = scale_pos_weight,
                            
    #                         gamma = self.gamma,
    #                         learning_rate = self.learning_rate,
    #                         max_depth = self.max_depth,

    #                         reg_lambda = self.reg_lambda,
    #                         # n_estimators = self.n_estimators,
    #                         n_estimators = 4000,
    #                         num_parallel_tree = 6,

    #                         n_jobs = self.threads,
    #                         validate_parameters = False
    #                     )

    #     sys.stdout.write("\tRunning the XGBoost classifier\r")
    #     sys.stdout.flush()

    #     # from sklearn.preprocessing import (
    #     #     StandardScaler,MinMaxScaler,QuantileTransformer,
    #     #     normalize,MaxAbsScaler
    #     #     )
    #     # transformer = StandardScaler()
    #     # # train_num2 = normalize(train_num, norm='l2')
    #     # # test_num2  = normalize(test_num , norm='l2')

    #     # train_num2 =transformer.fit_transform(train_num)
    #     # test_num2 =transformer.fit_transform(test_num)

    #     xgb_clf_no_nor.fit(train_num, train_labels,
    #                        eval_set = [ (test_num, test_labels) ],
    #                        early_stopping_rounds = 500,
    #                        verbose = not False,
    #                        eval_metric = 'auc'
    #                     )

    #     accuracy_score(test_labels, xgb_clf_no_nor.predict(test_num))
    #     accuracy_score(train_labels, xgb_clf_no_nor.predict(train_num))

    #     accuracy = accuracy_score(all_labels, xgb_clf_no_nor.predict(all_num))
    #     accuracy = accuracy_score(test_labels, xgb_clf_no_nor.predict(test_num))
    #     accuracy = accuracy_score(train_labels, xgb_clf_no_nor.predict(train_num))
    #     plot_confusion_matrix(
    #         xgb_clf_no_nor, all_num, all_labels,
    #         values_format  = 'd',
    #         display_labels = [_groups_dict[i] for i in xgb_clf_no_nor.classes_]
    #     )

    #     plot_confusion_matrix(
    #         xgb_clf_no_nor, test_num, test_labels,
    #         values_format  = 'd',
    #         display_labels = [_groups_dict[i] for i in xgb_clf_no_nor.classes_]
    #     )


    #     sys.stdout.write("\tRunning the XGBoost classifier, overall accuracy: %s\n" % round(accuracy, 6))
    #     sys.stdout.flush()

    #     sys.stdout.write("\tCalculating SHAP values\n")
    #     sys.stdout.flush()

    #     self.shap_things(
    #         xgb_clf_no_nor,
    #         all_num, 
    #         new_prefix, 
    #         _groups_dict, 
    #         accuracy
    #     )

    #     # TODO: DETELE. Lines only used for testing purposes
    #     import joblib
    #     joblib.dump(xgb_clf_no_nor, "../proofs_ggi/postRaxmlBug/flatfishes/two_hypo_ASTRAL-ML/version2/" + new_prefix + "_XGBoost_model_test.sav")
    #     # TODO: DETELE. Lines only used for testing purposes


    #     return (all_num, all_labels, 
    #             _groups_dict, xgb_clf_no_nor, 
    #             dataset)

    def confusion_plots(self, mytables):

        if not mytables:
            return None

        cnf_mx_filename = "cnf_mx_%s.png" % self.model_prefix

        if len(self.tree_id_comp)  <= 1:

            X, y,_groups_dict,xgb_clf_no_nor,title = mytables[0]
            plot_confusion_matrix(
                xgb_clf_no_nor, X, y,
                values_format  = 'd',
                display_labels = [_groups_dict[i] for i in xgb_clf_no_nor.classes_]
            )
            # axes[i].set_title(title, fontsize = 20)
            plt.savefig(cnf_mx_filename, bbox_inches = 'tight')
            plt.close()
            

        else: 
            if len(self.tree_id_comp) <= 3:
                nrows = 1
                self.cnf_ncols = len(self.tree_id_comp)
                
            else:
                res = len(mytables) % self.cnf_ncols
                nrows = (len(mytables) // self.cnf_ncols) + bool(res)

            f,axes = plt.subplots(nrows = nrows, ncols = self.cnf_ncols, figsize=(18, 5), dpi = 400)
            for i in range(len(mytables)):
                X, y,_groups_dict,xgb_clf_no_nor,title = mytables[i]

                plot_confusion_matrix(
                    xgb_clf_no_nor, X, y,
                    values_format  = 'd',
                    display_labels = [_groups_dict[i] for i in xgb_clf_no_nor.classes_],
                    ax = axes[i]
                )
                axes[i].set_title(title, fontsize = 20)

            plt.savefig(cnf_mx_filename, bbox_inches = 'tight')
            plt.close()

    def xgboost_iterator(self):

        self.add_metdata(None,  True)        

        mytables = []
        for tree_id_comp1 in self.tree_id_comp:

            tmp_model = self.xgboost_classifier(tree_id_comp1)
            if tmp_model:
                mytables.append(tmp_model)

        sys.stdout.write("Plotting confusion matrices\n")
        sys.stdout.flush()

        self.confusion_plots(mytables)


def spps_set_size(seq_path, seq_set):
    # seq_path, seq_set = seq_path,  test0_alns
    out = set()
    for s0 in seq_set:
        tmp_file = os.path.join(seq_path, s0)
        
        with open(tmp_file, 'r') as f:
            for line in f.readlines():
                if line.startswith(">"):
                    out.add( line.strip() )
    
    # level1_groups = set([i.split('_')[group] for i in out])
    # species, group
    return len(out)

def random_sampling(labels, seqs, seq_path, test_size = 0.25):
    
    counts = collections.Counter(labels)
    out_test = []
    out_train = []

    for k,v in counts.items():
        test_dim = int(v*test_size)
        # train_dim = v - test_dim

        sample_space = set()
        for n,i in enumerate(labels):
            if i == k:
                sample_space.add( seqs[n] )

        while True:

            test_seqs  = set( random.sample(sample_space, test_dim) )
            train_seqs = sample_space - test_seqs

            # checking species completeness
            test_spps_len  = spps_set_size(seq_path, test_seqs)
            train_spps_len = spps_set_size(seq_path, train_seqs)

            if test_spps_len == train_spps_len:
                break

        out_test += list(test_seqs)
        out_train += list(train_seqs)

        # out_test += list(zip(test_seqs, [k]*test_dim))
        # out_train += list(zip(train_seqs, [k]*train_dim))

    return out_train, out_test


def red_rank_wholeset(new_df, rank = 15):
    num_cols = list(set(new_df.columns.tolist()) - {'aln_base', 'hypothesis'})

    red = my_svd_red(np.array(new_df[num_cols]), rank)

    red_pd = pd.DataFrame(red, columns = num_cols)
    red_pd['aln_base'] = new_df['aln_base']
    red_pd['hypothesis'] = new_df['hypothesis']

    return red_pd

def bucketing(new_df, column = None, q = None ):
    # column = 'gc_mean_pos2'
    # q = 3
    # source idea
    # https://developers.google.com/machine-learning/data-prep/transform/bucketing

    precision = 3

    new_df[column + "_bucket"] = pd.qcut(
        new_df[column], 
        q=q,
        precision=precision,
        labels=range(1, q + 1)
    )

    return new_df

def iter_bucketing(new_df, columns = None, q = None, drop_nobuck = False):

    for i in columns:
        # i = 'gc_mean_pos2'
        new_df = bucketing(new_df, column=i, q=q)

        if drop_nobuck:
            new_df = new_df.drop( i, axis = 1 )
        
    return new_df

def correct_flat_features(mfile,new_df, filtering_features = None, reduce_rank = False, rank = None):
    # mfile = os.path.join(seq_path, 'new_features.csv')
    # invariants, singletons, patterns
    # entropy

    new_features = pd.read_csv(mfile)
    joined = pd.merge(new_df, new_features, on = 'aln_base', how='left')
    
    # drop gap_prop as it is equivalent as gap_mean
    joined = joined.drop( labels = 'gap_prop', axis = 1 )

    joined['pis'] = joined['pis']*100/joined['seq_len']
    joined['vars'] = joined['vars']*100/joined['seq_len']
    joined['invariants'] = joined['invariants']*100/joined['seq_len']
    joined['seq_len_nogap'] = joined['seq_len_nogap']*100/joined['seq_len']

    if filtering_features:
        joined = joined[['aln_base'] + filtering_features + ['hypothesis']]

    return joined

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




def dependent_columns(all_columns, mat, n):

    for j1,j2 in combinations(range(n), 2):

        v = mat[:,j1]
        w = mat[:,j2]
        # v = resampled_features[j1,:]
        # w = resampled_features[j2,:]
        v_norm = np.linalg.norm(v)
        w_norm = np.linalg.norm(w)

        if np.abs( v.T.dot(w) - v_norm * w_norm ) < 1e-6:
            print( all_columns[j1], all_columns[j2] )

def dependent_rows(all_rows, mat, m):

    for j1,j2 in combinations(range(m), 2):

        v = mat[j1,:]
        w = mat[j2,:]        
        # v = resampled_features[j1,:]
        # w = resampled_features[j2,:]
        v_norm = np.linalg.norm(v)
        w_norm = np.linalg.norm(w)

        if np.abs( v.T.dot(w) - v_norm * w_norm ) < 1e-6:
            print( all_rows[j1], all_rows[j2] )


bins   = [  0, 0.50, 0.60, 0.70, 0.80, 0.95, 1 ]
labels = [  0.1,   0.3,  0.5,  0.6,  0.7,   1  ]


def get_sample_weights(ggi_pd, seq_set, 
    bins = [0, 0.5, 0.6, 0.7, 0.8, 0.95, 1],
    labels= [0.1, 0.5, 0.6, 0.7, 0.8, 1]
    ):
    """
    ggi_pd: pandas data frame
    seq_set: set of sequences
    """
    # seq_set = X_train['aln_base']
    # bins = [0, 0.5, 0.6, 0.7, 0.8, 0.95, 1],
    # labels= [0.1, 0.5, 0.6, 0.7, 0.8, 1]

    tmp = (ggi_pd
              .loc[ggi_pd['alignment'].isin( seq_set ) & (ggi_pd['rank'] == '1'),]
              .set_index('alignment')
              .loc[seq_set] # sorter
              ['au_test']
              .astype(float)
            )

    return np.array(
                pd.cut(
                    tmp,
                    bins = bins,
                    labels= labels,
                    include_lowest=True
                )
            )


def do_resampling(train_num2, train_labels, train_weight, transform_labels = False):
    # train_num2.shape
    pos_features = train_num2[ train_labels]
    neg_features = train_num2[~train_labels]

    pos_labels = train_labels[ train_labels]
    neg_labels = train_labels[~train_labels]

    pos_train_weigth = train_weight[train_labels]
    neg_train_weigth = train_weight[~train_labels]

    if len(pos_features) < len(neg_features):

        ids = np.arange(len(pos_features))

        # taking as much as neg features are
        # available
        choices = np.random.choice(ids, len(neg_features)) 

        pos_features      = pos_features[choices]
        pos_labels        = pos_labels.iloc[choices]
        pos_train_weigth  = pos_train_weigth[choices]

    if len(pos_features) > len(neg_features):

        ids = np.arange(len(neg_features))

        # taking as much as pos features are
        # available
        choices = np.random.choice(ids, len(pos_features)) 

        neg_features      = neg_features[choices]
        neg_labels        = neg_labels.iloc[choices]
        neg_train_weigth  = neg_train_weigth[choices]

    # res_pos_features.shape
    resampled_features = np.concatenate([pos_features, neg_features], axis=0)
    resampled_labels   = np.concatenate([pos_labels, neg_labels], axis=0)
    resampled_weights  = np.concatenate([pos_train_weigth, neg_train_weigth], axis=0)

    order = np.arange(len(resampled_labels))
    
    np.random.shuffle(order)

    resampled_features = resampled_features[order]
    resampled_labels   = resampled_labels[order]
    resampled_weights  = resampled_weights[order]

    if transform_labels:
        resampled_labels = oneHotter(resampled_labels)
    

    return (resampled_features, 
            resampled_labels  , 
            resampled_weights )




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






def class_weight_ala_sklearn(labels):
    """
    source: https://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_class_weight.html
    """
    out = {}
    for k,v in collections.Counter(labels).items():
        out[k] = len(labels)/(2*v)

    return out

def drop_anomalies(dat, k, per = 4):
    # k = 3
    # per = 4
    # dat = train_num2
    from sklearn.mixture import GaussianMixture

    gm = (
        GaussianMixture(n_components=k,
                        n_init=10,
                        random_state=42)
            .fit(dat)
        )
    
    densities = gm.score_samples(dat)
    density_threshold = np.percentile(densities, per)

    to_filter = densities >= density_threshold

    return (dat[to_filter], to_filter)

def all_spps_in(X_train, X_test, seq_path = None, all_spps = None, hs = None, hypothesis = 'hypothesis'):
    
    test0_alns  = X_test[X_test[hypothesis] == hs[0]]['aln_base']
    train0_alns = X_train[X_train[hypothesis] == hs[0]]['aln_base']

    test1_alns  = X_test[X_test[hypothesis] == hs[1]]['aln_base']
    train1_alns = X_train[X_train[hypothesis] == hs[1]]['aln_base']

    # addin group level
    is_all_spps = (
        spps_set_size(seq_path,  test0_alns) ==\
        spps_set_size(seq_path, train0_alns) ==\
        spps_set_size(seq_path, test1_alns) ==\
        spps_set_size(seq_path, train1_alns) ==\
        all_spps
        )

    return is_all_spps


def _scaler(ref, dat, include_clip = True):

    # from sklearn.preprocessing import (
    #     StandardScaler,MinMaxScaler,QuantileTransformer,
    #     normalize,MaxAbsScaler,LabelEncoder, OneHotEncoder
    #     )
    # ref, dat = train_num, train_num
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

def oneHotter(dat):
    from sklearn.preprocessing import OneHotEncoder
    return (OneHotEncoder()
                .fit_transform( np.array(dat )
                .reshape(-1,1) )
                .toarray())

def generate_datasets(X_train, X_test, y_train, y_test, ggi_pd,
                      tree_id_comp1 = None,
                      aln_feature = 'aln_base', 
                      hypothesis = 'hypothesis',):

    from sklearn.preprocessing import StandardScaler,LabelEncoder

    train_num    = X_train.drop([aln_feature, hypothesis], axis = 1)
    train_labels = y_train == tree_id_comp1[0]

    test_num    = X_test.drop([aln_feature, hypothesis], axis = 1 )
    test_labels = y_test == tree_id_comp1[0]

    # standarizer = StandardScaler().fit(train_num)

    train_num2 = transform_data(ref = train_num, dat = train_num)
    test_num2  = transform_data(ref = train_num, dat = test_num)

    train_filter = np.array([True]*len(train_num))
    test_filter  = np.array([True]*len(test_num))

    # (train_num2, train_filter) = drop_anomalies( train_num2, k = 3, per = 15)
    # (test_num2 , test_filter ) = drop_anomalies( test_num2 , k = 3, per = 15)

    # train_label_Y = LabelEncoder().fit_transform( train_labels[train_filter])
    # test_label_Y  = LabelEncoder().fit_transform( test_labels[test_filter]  )

    train_label_Y = oneHotter( train_labels[train_filter])
    test_label_Y  = oneHotter( test_labels[test_filter]  )

    # bins   = [ 0, 0.50, 0.60, 0.70, 0.80, 0.95, 1]
    # labels = [0.1,  0.5,   0.6,  0.7,  0.8,   1  ]

    train_weight=get_sample_weights(ggi_pd,X_train['aln_base'][train_filter],bins,labels)
    test_weight =get_sample_weights(ggi_pd,X_test['aln_base'][test_filter]  ,bins,labels)

    return (train_num2, train_label_Y, train_weight,test_num2,test_label_Y,test_weight)

def log_reg(my_df):
    from sklearn.linear_model import LogisticRegression

    X = np.array(my_df.iloc[:, 0]).reshape(-1,1)

    if X.shape[0] < 30:
        return pd.NA

    y = my_df.iloc[:, 2]

    clf = LogisticRegression(random_state=0).fit(X, y)
    acc = clf.score(X, y)
    return acc


parts = 15
def assembly_data(parts,all_num,all_num2,all_labels):

    logreg_table = []
    m,n = all_num2.shape

    for i in range(n):
        # i
        fi = all_num.columns[i]
        # print(fi)
        tmp_df = pd.DataFrame(all_num2[:,i])
        
        mmin = float(tmp_df.min())
        mmax = float(tmp_df.max())
        window = (mmax - mmin)/parts
        
        bins = [mmin + window*i for i in range(parts)] + [float(mmax)]

        tmp_df['cat'] = pd.cut(
                    tmp_df.iloc[:,0],
                    bins=bins,
                    labels=range(parts)
        )
        tmp_df['label'] = all_labels
        
        log_out = tmp_df.groupby('cat').apply(log_reg).dropna()

        logreg_table.append([   
            fi,
            log_out.min(),
            log_out.max(),
            log_out.mean(),
            log_out.std(),
            log_out.__len__(),
            parts
        ])

    colnames = [
        'feature',
        'min',
        'max',
        'mean',
        'std',
        'len',
        'part'
        ]

    return pd.DataFrame(logreg_table, columns=colnames)

def my_svd_red(all_num2, r_new):
    U,S,Vt = np.linalg.svd( all_num2 )# full_matrices=False 
    S = np.diag(S)

    # low rank approximation
    cf_approx = ( U[:, 0:r_new]
                    .dot( S[0:r_new, 0:r_new] )
                    .dot( Vt.T[0:r_new,:] ) )    

    return cf_approx

def low_rank_data(r_new, parts,all_num, all_num2, all_labels):

    logreg_table = []
    m,n = all_num2.shape

    # low rank approximation
    cf_approx = my_svd_red(all_num2, r_new)

    for i in range(n):
        # i
        fi = all_num.columns[i]
        # print(fi)
        tmp_df = pd.DataFrame(cf_approx[:,i])
        
        mmin = float(tmp_df.min())
        mmax = float(tmp_df.max())
        window = (mmax - mmin)/parts
        
        bins = [mmin + window*i for i in range(parts)] + [float(mmax)]

        tmp_df['cat'] = pd.cut(
                    tmp_df.iloc[:,0],
                    bins=bins,
                    labels=range(parts)
        )
        tmp_df['label'] = all_labels
        log_out = tmp_df.groupby('cat').apply(log_reg).dropna()

        logreg_table.append([   
            fi,
            log_out.min(),
            log_out.max(),
            log_out.mean(),
            log_out.std(),
            log_out.__len__(),
            r_new
        ])

    colnames = [
        'feature',
        'min',
        'max',
        'mean',
        'std',
        'len',
        'part'
        ]

    return pd.DataFrame(logreg_table, columns=colnames)



# def _filter_whole_dataset(whole_set, k = 3, per = 10):
#     numerical_columns = set(whole_set.columns) - {'aln_base', 'hypothesis'}
#     std_df = StandardScaler().fit_transform(whole_set[numerical_columns])
#     _,df_fil = drop_anomalies( std_df, k = k, per = per)

#     # filtering the whole dataset
#     # by anomalies
#     return whole_set[df_fil].reset_index(drop = True)

