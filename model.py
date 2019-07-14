#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author：Genie time:2019/6/23

import lightgbm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from matplotlib import pyplot as plt
import pandas
import numpy

DATA_PATH_FINAL = r".\data\processed"

# =====================Data====================
train_df = pandas.read_csv(os.path.join(DATA_PATH_FINAL, "Train_pre.csv"), index=False)
test_df = pandas.read_csv(os.path.join(DATA_PATH_FINAL, "Test_pre.csv"), index=False)
X_train = train_df .iloc[:, :-1].values
Y_train = train_df .iloc[:, -1].values
X_test = test_df .iloc[:, :-1].values
Y_test = test_df.iloc[:, -1].values
seed = 10
# =====================Tuning====================
class Para:
    method = 'ligntgbm'
    k = 10  # 10 fold cross validation
    max_depth = 5
    min_data_in_leaf = 70

    feature_fraction = 0.5  # Default value is 1; specify the feature parts needed for each iteration
    bagging_fraction = 0.8  # The default value is 1; specify the data portion required for each iteration
    bagging_freq = 2
    min_gain_to_split = 0  # Default value is 1; Minimum information gain for splitting
    num_boost_round = 100  # iteration time
    learning_rate = 0.005
    num_leaves = 50  # Number of leaves per tree, default 31, type int
    max_bin = 255
    seed = 42
    boosting = "gbdt"
    reg_alpha = 0
    reg_lambda = 0


para = Para()
Model = lightgbm.LGBMClassifier(seed=42,
                                learning_rate=para.learning_rate,
                                max_depth=para.max_depth,
                                min_data_in_leaf=para.min_data_in_leaf,
                                feature_fraction=para.feature_fraction,
                                bagging_fraction=para.bagging_fraction,
                                bagging_freq=para.bagging_freq,
                                min_gain_to_split=para.min_gain_to_split,
                                num_boost_round=para.num_boost_round,
                                num_leaves=para.num_leaves,
                                max_bin=para.max_bin,
                                boosting=para.boosting,
                                reg_alpha=para.reg_alpha,
                                reg_lambda=para.reg_lambda
                                )

para_name = "num_boost_round"
para_name_all = numpy.arange(100, 2600, 500)
# param_grid = [{para_name: para_name_all, "reg_lambda": [0, 0.001, 0.01, 0.03, 0.08, 0.3, 0.5]}]
param_grid = [{para_name: para_name_all}]
scoring = {'AUC': 'roc_auc', 'Accuracy': 'accuracy'}
# X_axis = numpy.arange(1, 49+1)
# scorer = "Accuracy"

estimator = Model
X = X_train
y = Y_train
GSCV = GridSearchCV(estimator, param_grid, scoring=scoring, cv=para.k, refit=False, return_train_score=True)
eval_set = [(X_test, Y_test)]
GSCV.fit(X, y, eval_metric="logloss", eval_set=eval_set, verbose=True)
results = GSCV.cv_results_

# ===========================plot=======================
plt.figure(figsize=(13, 13))
plt.title("Parameter Tuning--"+para_name,
          fontsize=16)
plt.xlabel(para_name)
plt.ylabel("Score")
plt.grid()
ax = plt.axes()
X_axis = numpy.array(results["param_"+para_name].data, dtype=float)

for scorer, color in zip(sorted(scoring), ['g', 'k']):
    for sample, style in (('train', '--'), ('test', '-')):
        sample_score_mean = results['mean_%s_%s' % (sample, scorer)]
        sample_score_std = results['std_%s_%s' % (sample, scorer)]
        ax.fill_between(X_axis, sample_score_mean - sample_score_std,
                        sample_score_mean + sample_score_std,
                        alpha=0.1 if sample == 'test' else 0, color=color)
        ax.plot(X_axis, sample_score_mean, style, color=color,
                alpha=1 if sample == 'test' else 0.7,
                label="%s (%s)" % (scorer, sample))

    best_index = numpy.nonzero(results['rank_test_%s' % scorer] == 1)[0][0]
    best_score = results['mean_test_%s' % scorer][best_index]

    # Plot a dotted vertical line at the best score for that scorer marked by x
    ax.plot([X_axis[best_index], ] * 2, [0, best_score],
            linestyle='-.', color=color, marker='x', markeredgewidth=3, ms=8)

    # Annotate the best score for that scorer
    ax.annotate("%0.2f" % best_score,
                (X_axis[best_index], best_score + 0.005))

plt.legend(loc="best")
plt.grid('off')
plt.show()

# ============================test========================
eval_set = [(X_test, Y_test)]
Model.fit(X_train, Y_train, eval_metric="binary_logloss", eval_set=eval_set, verbose=True)
y_train_predict = Model.predict(X_train)
y_train_proba = Model.predict_proba(X_train)[:, 1]
y_test_predict = Model.predict(X_test)
y_test_proba = Model.predict_proba(X_test)[:, 1]

# AUC, accuracy
print(roc_auc_score(Y_train, y_train_proba))  
print(accuracy_score(Y_train, y_train_predict))
print(roc_auc_score(Y_train, y_test_proba))
print(accuracy_score(Y_train, y_test_predict))





