"""Clase 3: Clasificación"""

# %%
# Imports
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['KERAS_BACKEND'] = 'tensorflow'
# os.environ['KERAS_BACKEND'] = 'torch'

import pandas as pd
import numpy as np
import scipy.stats as sp_stats
import sklearn.model_selection as sk_ms
import sklearn.svm as sk_svm
import sklearn.tree as sk_tree
import sklearn.linear_model as sk_lm
import sklearn.naive_bayes as sk_nb
import sklearn.neighbors as sk_neig
import keras
from scikeras.wrappers import KerasClassifier

from mlfin.plotting import plot_feature_importance, plot_roc_curve
from mlfin.printing import print_classification_metrics


# %%
# Cargo datos
data = pd.read_csv('./data/stocks.csv')
data.dropna(inplace=True)


# %%
# Creamos Training y Testing Sets
y_tk = 'Direction'
x_tks = ['Lag1', 'Lag2', 'Lag3', 'Lag4', 'Lag5', 'Volume']
X_tr, X_te, y_tr, y_te = sk_ms.train_test_split(data.loc[:, x_tks],
                                                data.loc[:, y_tk],
                                                test_size=.1, random_state=123)


# %%
# Entrenamos Logistic Regression
lr = sk_lm.LogisticRegressionCV(cv=5, penalty='l1', solver='liblinear',
                                scoring='roc_auc')
lr.fit(X_tr, y_tr)

plot_roc_curve(lr, X_tr, y_tr)
print_classification_metrics(lr, X_tr, y_tr)

lr_score = np.max(np.mean(lr.scores_[1], axis=0))

best_score = lr_score
best_classifier = lr


# %%
# Entrenamos Naive Bayes
nb = sk_nb.GaussianNB()
nb.fit(X_tr, y_tr)

plot_roc_curve(nb, X_tr, y_tr)
print_classification_metrics(nb, X_tr, y_tr)

nb_score = sk_ms.cross_val_score(nb, X_tr, y_tr, cv=5, scoring='roc_auc').mean()

if nb_score > best_score:
    best_score = nb_score
    best_classifier = nb


# %%
# Entrenamos KNN
param_dict_knn = {'n_neighbors': range(4, 20),
                  'metric': ['euclidean', 'manhattan']}
# Metricas: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.distance_metrics.html

knn = sk_ms.GridSearchCV(sk_neig.KNeighborsClassifier(), param_dict_knn, cv=5,
                         scoring='roc_auc')

knn.fit(X_tr, y_tr)
knn_best = knn.best_estimator_

plot_roc_curve(knn_best, X_tr, y_tr)
print_classification_metrics(knn_best, X_tr, y_tr)

if knn.best_score_ > best_score:
    best_score = knn.best_score_
    best_classifier = knn_best


# %%
# Entrenamos SVM
param_dist_svc = {'C': sp_stats.expon(scale=100),
                  'kernel': ['rbf'], 'gamma': sp_stats.expon(scale=.1)}

svc = sk_ms.RandomizedSearchCV(sk_svm.SVC(), param_dist_svc, n_iter=10, cv=5,
                               scoring='roc_auc')

svc.fit(X_tr, y_tr)
svc_best = svc.best_estimator_

plot_roc_curve(svc_best, X_tr, y_tr)
print_classification_metrics(svc_best, X_tr, y_tr)

if svc.best_score_ > best_score:
    best_score = svc.best_score_
    best_classifier = svc_best


# %%
# SVM lineal usando NN equivalente
def build_nn():
    in_layer = keras.Input(shape=(len(X_tr.columns),))
    out_layer = keras.layers.Dense(1, activation='linear')(in_layer)

    model = keras.Model(in_layer, out_layer)
    model.compile(optimizer='SGD', loss='hinge')

    return model

svc_nn = KerasClassifier(model=build_nn, epochs=5, batch_size=20,
                         verbose=False)

svc_nn.fit(X_tr.values, y_tr)

plot_roc_curve(svc_nn, X_tr, y_tr)
print_classification_metrics(svc_nn, X_tr, y_tr)

svc_nn_score = sk_ms.cross_val_score(svc_nn, X_tr.values, y_tr, cv=5,
                                     scoring='roc_auc').mean()

if svc_nn_score > best_score:
    best_score = svc_nn_score
    best_classifier = svc_nn


# %%
# Entrenamos Classification Tree
param_grid_tree = {'min_samples_split': range(2, 21)}

tree = sk_ms.GridSearchCV(sk_tree.DecisionTreeClassifier(), param_grid_tree,
                          cv=5, scoring='roc_auc')

tree.fit(X_tr, y_tr)
tree_best = tree.best_estimator_

# Graficar feature_importances
plot_feature_importance(tree_best.feature_importances_, x_tks)

plot_roc_curve(tree_best, X_tr, y_tr)
print_classification_metrics(tree_best, X_tr, y_tr)

if tree.best_score_ > best_score:
    best_score = tree.best_score_
    best_classifier = tree_best


# %%
# Testeo el mejor modelo de entrenamiento
print(f'Training Score : {best_score:.2%}')
print(f'Testing Score  : {best_classifier.score(X_te, y_te):.2%}')
print('\nBest estimator:')
print(best_classifier)


# %%
# Entreno best_estimator con todo el dataset para dejarlo listo para uso.
best_classifier.fit(data.loc[:, x_tks], data.loc[:, y_tk]);
print('All done !')
