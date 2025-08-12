"""Clase 3: SVR y Regression Trees"""

# %%
# Imports
import pandas as pd
import scipy.stats as sp_stats
import sklearn.model_selection as sk_ms
import sklearn.svm as sk_svm
import sklearn.tree as sk_tree

from mlfin.plotting import plot_feature_importance


# %%
# Cargo datos
fact_rets = pd.read_csv('./data/F-F_Research_Data_5_Factors_2x3_daily.csv',
                        index_col=0, parse_dates=True) / 100.
spx_rets = pd.read_csv('./data/spx.csv', index_col=0,
                       parse_dates=True).pct_change()

data = pd.concat([spx_rets, fact_rets.iloc[:, :-1]], axis=1, sort=False,
                 join='inner')
data.dropna(inplace=True)


# %%
# Creamos Training y Testing Sets
y_tk = 'SPX'
x_tks = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']
X_tr, X_te, y_tr, y_te = sk_ms.train_test_split(data.loc[:, x_tks],
                                                data.loc[:, y_tk],
                                                test_size=.1, random_state=123)


# %%
# Entrenamos SVR usando GridSearchCV
param_grid_svr = [
    {'C': range(1, 6), 'epsilon': [0.05, 0.1, 0.15], 'kernel': ['linear']},
    {'C': [1, 10, 100, 1000], 'epsilon': [0.05, 0.1, 0.15], 'kernel': ['rbf'],
     'gamma': [0.001, 0.0001]},
]

svr_gs = sk_ms.GridSearchCV(sk_svm.SVR(), param_grid_svr, cv=5, scoring='r2')
# scores: https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter

svr_gs.fit(X_tr, y_tr)

# Para explorar resultados descomentar
# print(svr_gs.cv_results_)

best_score = svr_gs.best_score_
best_estimator = svr_gs.best_estimator_


# %%
# Entrenamos SVR usando RandomizedSearchCV
param_dist_svr = {'C': sp_stats.expon(scale=100),
                  'epsilon': sp_stats.uniform(0.05, 0.15),
                  'kernel': ['rbf'], 'gamma': sp_stats.expon(scale=.1)}

svr_rs = sk_ms.RandomizedSearchCV(sk_svm.SVR(), param_dist_svr, n_iter=10,
                                  cv=5, scoring='r2')

svr_rs.fit(X_tr, y_tr)

if svr_rs.best_score_ > best_score:
    best_score = svr_rs.best_score_
    best_estimator = svr_rs.best_estimator_


# %%
# Entrenamos Regression Tree usando GridSearchCV
param_grid_tree = {'min_samples_split': range(2, 6)}

tree_gs = sk_ms.GridSearchCV(sk_tree.DecisionTreeRegressor(), param_grid_tree,
                             cv=5, scoring='r2')

tree_gs.fit(X_tr, y_tr)

# Graficar feature_importances
plot_feature_importance(tree_gs.best_estimator_.feature_importances_, x_tks)

if tree_gs.best_score_ > best_score:
    best_score = tree_gs.best_score_
    best_estimator = tree_gs.best_estimator_


# %%
# Entrenamos Regression Tree usando RandomizedSearchCV
param_dist_tree = {'min_samples_split': range(2, 100)}

tree_rs = sk_ms.RandomizedSearchCV(sk_tree.DecisionTreeRegressor(),
                                   param_dist_tree, cv=5, n_iter=20,
                                   scoring='r2')

tree_rs.fit(X_tr, y_tr)

if tree_rs.best_score_ > best_score:
    best_score = tree_rs.best_score_
    best_estimator = tree_rs.best_estimator_


# %%
# Testeo el mejor modelo de entrenamiento
print(f'Training Score : {best_score:.2%}')
print(f'Testing Score  : {best_estimator.score(X_te, y_te):.2%}')
print('\nBest estimator:')
print(best_estimator)


# %%
# Entreno best_estimator con todo el dataset para dejarlo listo para uso.
best_estimator.fit(data.loc[:, x_tks], data.loc[:, y_tk])
print('All done !')
