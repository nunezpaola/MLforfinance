import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV
import cvxpy as cp
from mlfin.utils import get_allocations
from mlfin.printing import setup_logging, log_info, log_debug, log_warning, log_error
import logging

setup_logging(log_level=logging.INFO, log_file='ps/ps2_logs.txt')


class BalancedPortfolio:
    def __init__(self, asset_returns: pd.DataFrame):
        """
        Inicializa un objeto BalancedPortfolio.

        Parameters
        ----------
        asset_returns : pd.DataFrame
            DataFrame con retornos diarios de activos. Las columnas son nombres de activos
            y el índice son fechas datetime.
        """
        self.asset_returns = asset_returns.copy()

        log_debug("BalancedPortfolio inicializado con retornos de activos:")
        log_debug(self.asset_returns.head())
        log_debug(f"Cantidad de indices activos: {len(self.asset_returns.index)}")

    def get_balanced_portfolio(self, factor_returns: pd.DataFrame):
        """
        Calcula un portafolio balanceado con exposición igual a cada factor usando ML.

        Parameters
        ----------
        factor_returns : pd.DataFrame
            DataFrame con retornos diarios de factores. Las columnas son nombres de factores
            y el índice son fechas datetime.

        Returns
        -------
        opt_self.weights_conv : numpy.ndarray
            Array de pesos óptimos del portafolio (suman 1, no negativos).
        best_exposures_conv : numpy.ndarray
            Array de exposiciones del portafolio a cada factor.
        """

        log_debug("Datos iniciales (factores):")
        log_debug(factor_returns.head())
        log_debug(f"Cantidad de indices factores: {len(factor_returns.index)}")

        # Log fechas que no coinciden
        non_matching_dates = set(self.asset_returns.index) - set(factor_returns.index)
        if non_matching_dates:
            log_warning("Fechas que no coinciden: %s", non_matching_dates)

        # merge para alinear fechas
        df = pd.merge(self.asset_returns, factor_returns, left_index=True, right_index=True)

        # Elimina filas con valores NaN
        df = df.dropna()

        assets = list(self.asset_returns.columns)
        factors = list(factor_returns.columns)
        X = df[factors].values

        # Estima exposiciones de cada activo a los factores usando Lasso optimizado
        exposures = pd.DataFrame(index=assets, columns=factors, dtype=float)
        for asset in assets:
            y = df[asset].values
            model = LassoCV(cv=5, random_state=42).fit(X, y)
            exposures.loc[asset] = model.coef_

        # Construyo matriz de factor-loading B de forma (n_factors, n_assets)
        B = exposures.values.T * 100  # Paso a puntos porcentuales
        n_assets, n_factors = B.shape

        log_debug("Matriz de exposiciones (B):")
        log_debug(B)

        # Objetivo: exposición igual a cada factor
        target_exposure = 1.0 / n_factors
        t = np.full(n_factors, target_exposure)

        log_info(f"Objetivo de exposición por factor: {target_exposure:.1%}")

        # Camino 1: Metodo grilla (get_allocations) para encontrar el mejor portafolio
        log_info("--- Búsqueda en Grilla ---")
        self.weights_grilla = None
        self.score_grilla = float('inf')
        self.exposure_grilla = None

        log_debug(f"Número de activos: {n_assets}, Número de factores: {n_factors}")

        # Algoritmo de búsqueda en grilla
        allocation_count = 0
        for allocation in get_allocations(n_assets):
            allocation_count += 1

            # Convierte allocation a array de numpy
            weights = np.array(allocation)

            # Calcula exposiciones resultantes (exposure del activo * cuanto pondera)
            portfolio_exposures = B @ weights

            # Función objetivo: minimizar diferencia cuadrática con objetivo
            score = np.sum((portfolio_exposures - t) ** 2)

            if score < self.score_grilla:
                self.score_grilla = round(score, 3)
                self.weights_grilla = np.round(weights.copy(), 3)
                self.exposure_grilla = np.round(portfolio_exposures.copy(), 3)
                log_debug(f"Nueva mejor combinación (#{allocation_count}): {allocation}")
                log_debug(f"  Pesos: {weights}")
                log_debug(f"  Score: {score:.6f}")

        log_debug(f"Evaluadas {allocation_count} combinaciones")
        log_info(f"Función objetivo (MSE) en óptimo: {self.score_grilla:.3f}")
        log_info(f"Ponderaciones óptimas: "
                 f"{np.array2string(self.weights_grilla, precision=3)}")
        log_info(f"Exposición óptima a cada factor: "
                 f"{np.array2string(self.exposure_grilla, precision=3)}")

        # Camino 2: Optimización convexa
        log_info("--- Optimización Convexa ---")
        w = cp.Variable(n_assets)
        objective = cp.Minimize(cp.sum_squares(B @ w - t))  # misma función objetivo
        constraints = [cp.sum(w) == 1, w >= 0]  # pesos suman 1 y no negativos
        prob = cp.Problem(objective, constraints)

        # Intenta con diferentes solvers
        solvers_to_try = [cp.CLARABEL, cp.SCS, cp.OSQP]
        solver_used = None

        for solver in solvers_to_try:
            try:
                log_debug(f"Intentando con solver: {solver}")
                prob.solve(solver=solver)
                if prob.status == cp.OPTIMAL:
                    solver_used = solver
                    break
                else:
                    log_info(f"Solver {solver} no encontró solución óptima (status: {prob.status})")
            except Exception as e:
                log_error(f"Solver {solver} falló: {e}")

        if prob.status == cp.OPTIMAL:
            self.weights_conv = np.round(w.value, 3)
            self.exposure_conv = np.round(B @ self.weights_conv, 3)
            self.score_conv = round(prob.value, 3)
            log_info(f"Optimización convexa exitosa con {solver_used}.")
            log_info(f"Función objetivo (MSE) en el óptimo: {self.score_conv:.3f}")
            log_info(f"Ponderaciones óptimas: "
                     f"{np.array2string(self.weights_conv, precision=3)}")
            log_info(f"Exposición óptima a cada factor: "
                     f"{np.array2string(self.exposure_conv, precision=3)}")

            # Usar el mejor resultado
            if self.score_conv < self.score_grilla:
                log_info("--- Resultado óptimo: optimización convexa ---")
                self.result = {
                    'optimal_results': {
                        'weights': np.round(self.weights_conv, 3),
                        'exposiciones': np.round(self.exposure_conv, 3),
                        'score': round(self.score_conv, 3)
                    },
                    'all': {
                        'weights_grilla': np.round(self.weights_grilla, 3),
                        'exposures_grilla': np.round(self.exposure_grilla, 3),
                        'score_grilla': round(self.score_grilla, 3),
                        'weights_convexa': np.round(self.weights_conv, 3),
                        'exposiciones_convexa': np.round(self.exposure_conv, 3),
                        'score_convexa': round(self.score_conv, 3),
                        'solver_used': solver_used,
                        'model_chosen': 'convexa'
                    }
                }
                return np.round(self.weights_conv, 3), np.round(self.exposure_conv, 3)
            else:
                pass
        else:
            log_info(f"Optimización falló con estado: {prob.status}")

        self.result = {
            'optimal_results': {
                'weights': np.round(self.weights_grilla, 3),
                'exposiciones': np.round(self.exposure_grilla, 3),
                'score': round(self.score_grilla, 3)
            },
            'all': {
                'weights_grilla': np.round(self.weights_grilla, 3),
                'exposures_grilla': np.round(self.exposure_grilla, 3),
                'score_grilla': round(self.score_grilla, 3),
                'weights_convexa': (np.round(self.weights_conv, 3)
                                    if hasattr(self, 'weights_conv') else None),
                'exposiciones_convexa': (np.round(self.exposure_conv, 3)
                                         if hasattr(self, 'exposure_conv') else None),
                'score_convexa': (round(self.score_conv, 3)
                                  if hasattr(self, 'score_conv') else None),
                'solver_used': solver_used,
                'model_chosen': 'grilla'
            }
        }

        # Si se llegó hasta acá, usar resultado de grilla
        log_info("--- Resultado óptimo: grilla ---")
        return np.round(self.weights_grilla, 3), np.round(self.exposure_grilla, 3)