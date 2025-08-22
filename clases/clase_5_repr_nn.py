"""Clase 5: Representación con NN"""

# %%
# Imports
import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['KERAS_BACKEND'] = 'torch'

import numpy as np
import pandas as pd
import sklearn.preprocessing as sk_pp
import sklearn.linear_model as sk_lm
import keras

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
# Configuro y preproceso
m = len(features)
comps = 8

X_tr = sk_pp.StandardScaler(with_std=False).fit_transform(X_tr)
X_te = sk_pp.StandardScaler(with_std=False).fit_transform(X_te)


# %%
# Linear Autoencoder
# Armo la NN y entreno
input_tensor_l = keras.Input(shape=(m,))
hidden_l = keras.layers.Dense(comps)(input_tensor_l)
output_tensor_l = keras.layers.Dense(m)(hidden_l)

ae_l = keras.Model(input_tensor_l, output_tensor_l)
ae_l.compile(optimizer='Adam', loss='mse')
ae_l.fit(X_tr, X_tr, epochs=10, batch_size=10)

# Extraigo sólo la mitad ya entrenada para armar el encoder.
encoder_l = keras.Model(input_tensor_l, hidden_l)

# Transformo los datos y los utilizo.
pc_l = encoder_l.predict(X_tr, verbose=0)
lr_l = sk_lm.LogisticRegressionCV(cv=5, penalty='l1', solver='liblinear',
                                  scoring='roc_auc')
lr_l.fit(pc_l, y_tr)
print(f'Linear     -> best CV score: {np.max(np.mean(lr_l.scores_[1.0], axis=0)):.2%}')


# %%
# Non-linear Autoencoder
# Armo la NN y entreno
input_tensor_nl = keras.Input(shape=(m,))
hidden_nl_1 = keras.layers.Dense(round((comps + m) / 2), activation='sigmoid')(input_tensor_nl)
hidden_nl_2 = keras.layers.Dense(comps)(hidden_nl_1)
hidden_nl_3 = keras.layers.Dense(round((comps + m) / 2), activation='sigmoid')(hidden_nl_2)
output_tensor_nl = keras.layers.Dense(m)(hidden_nl_3)

ae_nl = keras.Model(input_tensor_nl, output_tensor_nl)
ae_nl.compile(optimizer='Adam', loss='mse')
ae_nl.fit(X_tr, X_tr, epochs=10, batch_size=10)

# Extraigo sólo la mitad ya entrenada para armar el encoder.
encoder_nl = keras.Model(input_tensor_nl, hidden_nl_2)

# Transformo los datos y los utilizo.
pc_nl = encoder_nl.predict(X_tr, verbose=0)
lr_nl = sk_lm.LogisticRegressionCV(cv=5, penalty='l1', solver='liblinear',
                                   scoring='roc_auc')
lr_nl.fit(pc_nl, y_tr)
print(f'Non-linear -> best CV score: {np.max(np.mean(lr_nl.scores_[1.0], axis=0)):.2%}')


# %%
# Realizo tesing con Non-Linear
print("Testing Results:")
# Aplico transformación a los datos originales de testing primero
pc_nl_te = encoder_nl.predict(X_te, verbose=0)
print_classification_metrics(lr_nl, pc_nl_te, y_te)


# Para la implementación de tied weights ver por ejemplo:
# https://medium.com/data-science/build-the-right-autoencoder-tune-and-optimize-using-pca-principles-part-i-1f01f821999b
