"""Clase 2: Regresiones y NN"""

# %%
# Imports & definitions
import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['KERAS_BACKEND'] = 'torch'

import numpy as np
import sklearn.linear_model as sk_lm
import keras
from scikeras.wrappers import KerasRegressor

from mlfin.printing import print_validation_results

rng = np.random.default_rng()


# %%
# Environment cheks
backend = keras.backend.backend()
print(f'Keras backend   : {backend}')

if backend == 'torch':
    import torch
    print(f' -> GPU         : {torch.cuda.is_available()}')
    print(f' -> Silicon     : {torch.mps.is_available()}')

elif backend == 'tensorflow':
    import tensorflow as tf
    print(f' -> GPU/Silicon : {len(tf.config.list_physical_devices('GPU')) > 0}')

else:
    raise RuntimeError(f'Unsupported backend: {backend}')


# %%
# Creando sets de datos
X = rng.standard_normal((1000, 3))
y = -1. + X @ np.array([3., 5., 7.]) + rng.standard_normal(1000) * 0.15


# %%
# API Secuencial
nn_seq = keras.Sequential()
nn_seq.add(keras.layers.Dense(1, activation='linear', use_bias=True))

nn_seq.compile(optimizer='SGD', loss='mse')
nn_seq.fit(X, y, epochs=5, batch_size=20)
print(nn_seq.get_weights())


# %%
# API Funcional
input_tensor = keras.Input(shape=(3,))
output_tensor = keras.layers.Dense(1, use_bias=True)(input_tensor)
nn_func = keras.Model(input_tensor, output_tensor)

nn_func.compile(optimizer='SGD', loss='mse')
nn_func.fit(X, y, epochs=5, batch_size=20)
print(nn_func.get_weights())


# %%
# Usando scikeras wrapper
def build_nn():
    in_layer = keras.Input(shape=(3,))
    out_layer = keras.layers.Dense(1, use_bias=True)(in_layer)

    model = keras.Model(in_layer, out_layer)
    model.compile(optimizer='SGD', loss='mse')

    return model

nn_keras = KerasRegressor(model=build_nn, epochs=5, batch_size=20,
                          verbose=False)


# %%
# Ejecuto Cross-Validation y entreno con API scikit-learn
ols = sk_lm.LinearRegression()

print_validation_results(nn_keras, X, y)
print()
print_validation_results(ols, X, y)

nn_keras.fit(X, y);
ols.fit(X, y);


# %%
# Modelo con 2 capas (1 hidden layer)
input_tensor_2 = keras.Input(shape=(3,))
hidden = keras.layers.Dense(2, activation='relu')(input_tensor_2)
output_tensor_2 = keras.layers.Dense(1)(hidden)

nn_2l = keras.Model(input_tensor_2, output_tensor_2)
nn_2l.compile(optimizer='SGD', loss='mse')

nn_2l.fit(X, y, epochs=5, batch_size=20)
print(nn_2l.get_weights())


# %%
# Realizamos una predicción
print('Predicción para [1., 4., 6.] = ', nn_2l.predict(np.array([[1., 4., 6.]]), verbose=0))


# %%
# Observamos la red
nn_2l.summary()
