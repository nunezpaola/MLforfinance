import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import validate_data, check_is_fitted


class EstimadorOLS(RegressorMixin, BaseEstimator):
    """
    Estimador de regresión lineal por mínimos cuadrados ordinarios (OLS).

    Implementa regresión lineal usando la solución de forma cerrada de
    mínimos cuadrados ordinarios. Compatible con la API de scikit-learn.

    Parameters
    ----------
    fit_intercept : bool, default=True
        Si True, se calcula el intercepto para este modelo. Si se establece
        en False, no se utilizará intercepto en los cálculos (se espera que
        los datos estén centrados).

    Attributes
    ----------
    coef_ : ndarray de dimensión (n_features,)
        Coeficientes estimados para el problema de regresión lineal.
        Si se proporciona solo una variable objetivo, este es un array 1D
        de longitud n_features.
    intercept_ : float
        Constante de intercepto del modelo lineal.

    Notes
    -----
    La implementación utiliza la fórmula de forma cerrada:

    .. math:: \\beta = (X^T X)^{-1} X^T y

    Esta aproximación es computacionalmente eficiente para datasets pequeños
    a medianos, pero puede ser numéricamente inestable o imposible de calcular
    para matrices no invertibles.

    """
    def __init__(self, fit_intercept: bool = True):
        """
        Estimador de regresión lineal por mínimos cuadrados clásicos (OLS).

        Parameters
        ----------
        fit_intercept : bool, default=True
            Si True, se incorpora un intercepto al modelo a estimar.
            Si False, no se ajusta el intercepto y se asume que los datos
            están centrados en cero.

        Attributes
        ----------
        coef_ : ndarray de dimensión (n_features,)
            Coeficientes estimados para las características de entrada.
        intercept_ : float
            Constante de intercepto del modelo lineal.

        """

        self.fit_intercept = fit_intercept

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Ajusta el modelo de regresión lineal a los datos de entrenamiento.

        Utiliza la solución de forma cerrada de mínimos cuadrados ordinarios
        para estimar los coeficientes del modelo lineal.

        Parameters
        ----------
        X : array-like de dimensión (n_samples, n_features)
            Datos de entrenamiento de las features de entrada.
        y : array-like de dimensión (n_samples,)
            Valores objetivo (variable dependiente).

        Returns
        -------
        self : object
            Retorna la instancia ajustada del estimador.

        Raises
        ------
        ValueError
            Si el número de muestras es menor que el número de características
            (más 1 si se ajusta el intercepto), o si la matriz de diseño es
            singular y no puede invertirse.

        Notes
        -----
        El método utiliza la fórmula de forma cerrada:

        .. math:: \\beta = (X^T X)^{-1} X^T y

        donde X es la matriz de diseño (con columna de unos si fit_intercept=True)
        e y es el vector de valores objetivo.

        """
        # Se validan y convierten los datos de entrada
        X, y = validate_data(self, X, y, accept_sparse=False, y_numeric=True)

        # Se valida que tenemos suficientes muestras
        n_samples, n_features = X.shape
        if self.fit_intercept:
            min_samples = n_features + 1
        else:
            min_samples = n_features

        if n_samples < min_samples:
            raise ValueError(
                f"La cantidad de elementos de la muestra (n_samples={n_samples}) debe ser al menos"
                f" igual a la cantidad de features (n_features={n_features})"
                f"{' más 1 para el intercepto' if self.fit_intercept else ''}. "
                f"Se obtuvieron {n_samples} muestras."
            )

        if self.fit_intercept:
            ones = np.ones((X.shape[0], 1))
            X_fit = np.hstack([ones, X])
        else:
            X_fit = X

        # Se usa la forma cerrada (existe bajo rango completo y sin multicolinealidad)
        try:
            beta = np.linalg.inv(X_fit.T @ X_fit) @ (X_fit.T @ y)
        except np.linalg.LinAlgError:
            # Handling del caso en el que la matriz es singular
            raise ValueError(
                "La matriz de diseño es singular y no se puede invertir. "
                "Esto puede ocurrir cuando hay menos muestras que features, "
                "o cuando existe multicolinealidad entre las features."
            )

        if self.fit_intercept:
            self.intercept_ = float(beta[0])
            self.coef_ = beta[1:]
        else:
            self.intercept_ = 0.0  # No hay intercepto
            self.coef_ = beta

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predice usando el modelo de regresión lineal ajustado.

        Parameters
        ----------
        X : array-like de dimensión (n_samples, n_features)
            Muestras de entrada para las cuales se realizarán las predicciones.

        Returns
        -------
        y_pred : ndarray de dimensión (n_samples,)
            Valores predichos por el modelo lineal.

        Raises
        ------
        NotFittedError
            Si el estimador no ha sido ajustado antes de llamar a predict.

        Notes
        -----
        Las predicciones se calculan como:

        .. math:: \\hat{y} = X \\beta + \\beta_0

        donde β son los coeficientes ajustados y β₀ es el intercepto.

        """

        # Se chequea que el modelo esté ajustado
        check_is_fitted(self, ['coef_', 'intercept_'])

        # Valido el input
        X = validate_data(self, X, reset=False)

        return X @ self.coef_ + self.intercept_