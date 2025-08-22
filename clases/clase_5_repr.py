"""Clase 5: Representación"""

# %%
# Imports
import pandas as pd
import numpy as np
import sklearn.model_selection as sk_ms
import sklearn.pipeline as sk_pl
import sklearn.linear_model as sk_lm
import sklearn.decomposition as sk_de

from mlfin.printing import print_classification_metrics


# %%
# Obtengo datos
train_data = pd.read_hdf('data/central_bank_data.h5', key='bank_defaults_FDIC')
test_data = pd.read_hdf('data/central_bank_data.h5', key='regulated_banks')

fundamentals = ['log_TA', 'NI_to_TA', 'Equity_to_TA', 'NPL_to_TL', 'REO_to_TA',
                'ALLL_to_TL', 'core_deposits_to_TA', 'brokered_deposits_to_TA',
                'liquid_assets_to_TA', 'loss_provision_to_TL', 'NIM',
                'assets_growth']

market_conditions = ['term_spread', 'stock_mkt_growth', 'real_gdp_growth',
                     'unemployment_rate_change', 'treasury_yield_3m',
                     'bbb_spread', 'bbb_spread_change']

label = 'defaulter'
features = fundamentals + market_conditions

X_tr, y_tr = train_data.loc[:, features], train_data.loc[:, label]
X_te, y_te = test_data.loc[:, features], test_data.loc[:, label]


# %%
# Armo caso base
lr_base = sk_lm.LogisticRegressionCV(cv=5, penalty='l1', solver='liblinear',
                                     scoring='roc_auc')
lr_base.fit(X_tr, y_tr)
print(f'Base       -> best CV score: {np.max(np.mean(lr_base.scores_[1.0], axis=0)):.2%}')


# %%
# Usando PCA al 95% de varianza
pca = sk_de.PCA(n_components=0.95).fit(X_tr)
X_tr_pca = pca.transform(X_tr)

lr_pca = sk_lm.LogisticRegressionCV(cv=5, penalty='l1', solver='liblinear',
                                    scoring='roc_auc')
lr_pca.fit(X_tr_pca, y_tr)
print(f'PCA 95%    -> best CV score: {np.max(np.mean(lr_pca.scores_[1.0], axis=0)):.2%}')


# %%
# Usando Kernel-PCA con Pipelines
pipe_kpca = sk_pl.Pipeline([('kpca', sk_de.KernelPCA()),
                            ('lr', sk_lm.LogisticRegressionCV(cv=5, penalty='l1',
                                                              solver='liblinear',
                                                              scoring='roc_auc'))])

pg_kpca = [{'kpca__n_components': range(2, len(features) + 1), ## KPCA__ para darl parametros al modelo     
            'kpca__kernel': ['rbf', 'sigmoid']}]

lr_kpca = sk_ms.GridSearchCV(pipe_kpca, pg_kpca, cv=5, scoring='roc_auc')
lr_kpca.fit(X_tr, y_tr)
print(f'Kernel-PCA -> best CV score: {lr_kpca.best_score_:.2%}')


# %%
# Usando ICA
ica = sk_de.FastICA(max_iter=1000, whiten='unit-variance').fit(X_tr)
X_tr_ica = ica.transform(X_tr)

lr_ica = sk_lm.LogisticRegressionCV(cv=5, penalty='l1', solver='liblinear',
                                    scoring='roc_auc')
lr_ica.fit(X_tr_ica, y_tr)
print(f'ICA        -> best CV score: {np.max(np.mean(lr_pca.scores_[1.0], axis=0)):.2%}')


# %%
# Realizo tesing con Kernel-PCA
print("Testing Results:")
print_classification_metrics(lr_kpca, X_te, y_te)


## --- Pickleo de modelos ---
# Sirve para almacenar un modelo entrenado y luego poder volver a recuperarlo
import pickle
import os

# 1) Creo carpeta si no existe
if not os.path.exists('./trained_models'):
    os.makedirs('./trained_models')

# 2) Almaceno el modelo
if os.path.exists('./trained_models/best_model.pickle'):
    os.remove('./trained_models/best_model.pickle')

with open('./trained_models/best_model.pickle', 'xb') as _file:
    pickle.dump(lr_kpca, _file)

# 3) Recupero el modelo
with open('./trained_models/best_model.pickle', 'rb') as _file:
    awesome_model = pickle.load(_file)
