"""Utilidades para impresión de resultados."""

import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['KERAS_BACKEND'] = 'torch'

import numpy as np
import sklearn.model_selection as sk_ms
import logging
from scikeras.wrappers import KerasRegressor

from .utils import binary_classification_metrics


def print_validation_results(estimator, X, y, random_state=None):
    """
    Imprime en pantalla resultados de validación.
    Parámetros
        estimator    : Estimador compatible con API scikit-learn
        X            : Valores variables independientes.
                       numpy.ndarray (n filas por k columnas)
        y            : Valores variable dependiente.
                       numpy.ndarray
        random_state : (opcional) Fija seed del generador de números
                       pseudoaleatorios
    """
    log_info(f'Testeando {type(estimator)}')

    np.random.seed(random_state)
    scores = sk_ms.cross_val_score(estimator, X, y, cv=5)

    if isinstance(estimator, KerasRegressor):
        total_mse = ((y - y.mean()) ** 2).mean()
        scores = 1 + scores / total_mse

    log_info('Resultado de Cross-Validation:')
    log_info('Scores  : [' + ', '.join(f'{n:.2%}' for n in scores) + ']')
    log_info(f'Mean    : {scores.mean():.2%}')
    log_info(f'Std Dev : {scores.std():.2%}')


def print_classification_metrics(model, X_test, y_true):
    """
    Computa e imprimer resumen con ROC AUC, Kolmogorov-Smirnov y Accuracy Score
    para un modelo de clasificación binaria dado.
    Parámetros
        model         : Modelo ya fitteado
        X_test        : Dataset con features
        y_true        : Etiquetras correctas para el dataset X_test
    """
    auc, acc, ks = binary_classification_metrics(model, X_test, y_true)

    log_info(f'Accuracy           = {acc:.2%}')
    log_info(f'ROC AUC            = {auc:.2%}')
    log_info(f'Kolmogorov-Smirnov = {ks:.4f}')


def setup_logging(log_level=logging.INFO, log_file='mlfin_logs.txt'):
    """Configura el sistema de logging para escribir a archivo y consola.
    """
    # Se eliminan handlers existentes
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # Se configura el formato del logging
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Handler para archivo (APPEND - agrega al final del archivo)
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setFormatter(formatter)
    
    # Handler para consola
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # Se configura el logger
    logging.basicConfig(
        level=logging.INFO,
        handlers=[file_handler, console_handler],
        format= '%(asctime)s - %(levelname)s - %(message)s',
    )

def log_info(message):
    """Función helper para loggear información."""
    logging.info(message)

def log_debug(message):
    """Función helper para loggear mensajes de debug."""
    logging.debug(message)

def log_warning(message):
    """Función helper para loggear advertencias."""
    logging.warning(message)

def log_error(message):
    """Función helper para loggear errores."""
    logging.error(message)
