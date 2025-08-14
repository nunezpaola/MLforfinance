from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
import logging

from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from mlfin.printing import setup_logging

# setup_logging(log_level=logging.INFO, log_file='ps/ps3_logs.txt')

# Seteo el modelo de datos a recibir
@dataclass
class CVResult:
    name: str
    best_score: float
    best_params: Dict[str, Any]
    estimator: BaseEstimator

class AnalistaDeRiesgo:
    """
    Estima el total de activos en riesgo del sistema financiero ante
    eventuales defaults de entidades financieras. 
    
    Parameters
    ----------
    estimator_parameters : List[Tuple[BaseEstimator, Dict[str, Any]]]
        Lista de tuplas (modelo, grid) para la búsqueda de hiperparámetros.
    scoring : str
        Métrica de evaluación a utilizar durante la validación cruzada.
    """
    def __init__(self,
                 modelos_y_grids: List[Tuple[BaseEstimator, Dict[str, Any]]],
                 scoring: str = "roc_auc",
                 random_state: int = 42) -> None:

        # Guardo los parámetros recibidos
        self.modelos_y_grids = modelos_y_grids
        self.scoring = scoring
        self.random_state = random_state

        # Inicializo atributos adicionales
        self.df_train: Optional[pd.DataFrame] = None
        self.default_col: Optional[str] = None
        self._best: Optional[CVResult] = None

        # Log debug de lo recibido
        logging.debug(f"Modelos y grids recibidos: {modelos_y_grids}")
        logging.debug(f"Métrica de scoring: {scoring}")
        logging.debug(f"Random state: {random_state}")

    def load_data(self, df_train: pd.DataFrame, default_col: str) -> None:
        """
        Carga los datos históricos y la columna objetivo (indicador de default)
        para el análisis.

        Parameters
        ----------
        df_historico : pd.DataFrame
            DataFrame que contiene los datos históricos.
        default_col : str
            Nombre de la columna objetivo (indicador de default).
        """
        # Valido que exista la columna
        if default_col not in df_train.columns:
            raise KeyError(f"default_col '{default_col}' no está en el DataFrame.")

        # Copio el DataFrame y guardo la columna objetivo
        self.df_train = df_train.copy()
        self.default_col = default_col

        # log debug de las caracteristicas de df_train
        logging.debug(f"Características de df_train: {df_train.columns.tolist()}")
        logging.debug(f"Columna objetivo: {default_col}")
        logging.debug(f"Cantidad de observaciones: {df_train.shape[0]}")

    # métodos privados
    def _cv_scheme(self):
        """
        Corre el modelo utilizando un esquema de validación cruzada.
        """
        # 5 folds y 5 repeticiones
        return RepeatedStratifiedKFold(n_splits=5, n_repeats=5,
                                       random_state=self.random_state)

    def _make_preprocessor(self, features: List[str], scale: bool) -> ColumnTransformer:
        """
        Preprocesa columnas numéricas.

        Parameters
        ----------
        features : List[str]
            Lista de nombres de columnas a preprocesar.
        scale : bool
            Indica si se debe aplicar escalado a las características.

        Returns
        -------
        ColumnTransformer

        """
        # Pipe 1. Imputo los datos faltantes usando la mediana
        numeric_pipe = [("impute", SimpleImputer(strategy="median"))]

        # Pipe 2. Escalado (si corresponde)
        if scale:
            numeric_pipe.append(("scale", StandardScaler()))

        # Armo el pipeline final
        numeric = Pipeline(numeric_pipe)

        return ColumnTransformer(
            transformers=[("num", numeric, features)],
            remainder="drop",
        )

    def _wrap_grid(self, grid: Dict[str, Any], clf_name: str = "clf") -> Dict[str, Any]:
        """
        Este método ajusta los nombres de los hiperparámetros para usarlos en un Pipeline
        de scikit-learn ('clf__param')

        Parameters
        ----------
        grid : Dict[str, Any]
            Diccionario de hiperparámetros a ajustar.
        clf_name : str
            Nombre del clasificador en el Pipeline.

        Returns
        -------
        Dict[str, Any]
            Diccionario con los hiperparámetros.
        """
        wrapped = {}
        for k, v in grid.items():
            wrapped[f"{clf_name}__{k}"] = v
        return wrapped

    def _fit_and_select(self,
                        X: pd.DataFrame | pd.Series,
                        y: pd.Series,
                        features: List[str]) -> CVResult:
        """
        Ajusta y selecciona el mejor modelo utilizando validación cruzada.

        Parameters
        ----------
        X : pd.DataFrame | pd.Series
            Matriz de características.
        y : pd.Series
            Vector de etiquetas.
        features : List[str]
            Lista de nombres de columnas a utilizar como características.

        Returns
        -------
        CVResult
            Resultado de la validación cruzada.
        """
        X = X[features].apply(pd.to_numeric, errors="coerce")

        results: List[CVResult] = []

        for est, grid in self.modelos_y_grids:
            # Para modelos sensibles a escala uso scaler; para Tree sin escalar
            needs_scaling = est.__class__.__name__.lower() in ("kneighborsclassifier", "svc")
            pre = self._make_preprocessor(features, scale=needs_scaling)

            pipe = Pipeline(steps=[
                ("pre", pre),
                ("clf", est),
            ])

            param_grid = self._wrap_grid(grid, "clf")

            gs = GridSearchCV(
                estimator=pipe,
                param_grid=param_grid,
                scoring=self.scoring,
                cv=self._cv_scheme(),
                n_jobs=-1, #uso todos los núcleos
                refit=True, #fijo el mejor modelo
                verbose=0,#no muestro el progreso
            )
            gs.fit(X, y)

            best = CVResult(
                name=est.__class__.__name__,
                best_score=float(gs.best_score_),
                best_params=gs.best_params_,
                estimator=gs.best_estimator_,
            )
            results.append(best)
            logging.debug(f"Mejor modelo para {est.__class__.__name__}: {best.best_params}")

        # Elige por mejor score
        results.sort(key=lambda r: r.best_score, reverse=True)
        logging.debug(f"Mejores resultados: {results}")
        return results[0]

    @staticmethod
    def pick_threshold(y_true: np.ndarray,
                       y_proba: np.ndarray,
                       method: str = "fixed",
                       *,
                       fixed: float = 0.5,
                       C_FP: float = 1.0,
                       C_FN: float = 1.0,
                       target_rate: Optional[float] = 0.1,
                       weights: Optional[np.ndarray] = None) -> float:
        """EXTRA 
        Elige umbral de corte para proba de default
        por: 'fixed', 'youden', 'f1', 'cost', 'target_rate'."""
        y_true = np.asarray(y_true).astype(int)
        y_proba = np.asarray(y_proba).astype(float)

        if method == "fixed":
            logging.debug(f"Umbral fijo seleccionado: {fixed}")
            return float(fixed)

        if method.lower() == "youden":
            fpr, tpr, thr = roc_curve(y_true, y_proba)
            j = tpr - fpr
            logging.debug(f"Umbral Youden seleccionado: {thr[j.argmax()]}")
            return float(thr[j.argmax()])

        if method.lower() == "f1":
            prec, rec, thr = precision_recall_curve(y_true, y_proba)
            f1 = 2 * prec * rec / (prec + rec + 1e-12)
            # thr tiene len = len(prec) - 1
            logging.debug(f"Umbral por F1 seleccionado: {thr[np.nanargmax(f1[:-1])]}")
            return float(thr[np.nanargmax(f1[:-1])])

        if method.lower() == "cost":
            w = np.ones_like(y_true) if weights is None else np.asarray(weights, dtype=float)
            ts = np.linspace(0, 1, 501)
            def tot_cost(t):
                pred = (y_proba >= t).astype(int)
                fp = ((pred == 1) & (y_true == 0)) * w
                fn = ((pred == 0) & (y_true == 1)) * w
                return C_FP * fp.sum() + C_FN * fn.sum()
            logging.debug(f"Umbral por costo seleccionado: {min(ts, key=tot_cost)}")
            return float(min(ts, key=tot_cost))

        if method.lower() == "target_rate":
            rate = 0.1 if target_rate is None else float(target_rate)
            rate = min(max(rate, 0.0), 1.0)
            if not (0.0 < rate < 1.0):
                raise ValueError("target_rate debe estar entre 0 y 1 (excluidos).")
            logging.debug(f"Umbral por tasa objetivo seleccionado: {np.quantile(y_proba, 1 - rate)}")
            return float(np.quantile(y_proba, 1 - rate))

        raise ValueError("method inválido. Usá: fixed, youden, f1, cost, target_rate")

    def get_report(self,
                   df_predict: pd.DataFrame,
                   features: List[str],
                   umbral_riesgo: float | str = "fixed",
                   threshold_kwargs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Predice PD para df_predict y calcula métricas agregadas.
        - umbral_riesgo:
            * float -> usa ese umbral fijo
            * str   -> método para elegir umbral con pick_threshold:
                    'fixed' | 'youden' | 'f1' | 'cost' | 'target_rate'
        - threshold_kwargs: parámetros extra para el método (ej:
            {'fixed':0.5} | {'C_FP':1,'C_FN':20} | {'target_rate':0.10})
        """
        threshold_kwargs = threshold_kwargs or {}

        # Valido que se hayan cargado los datos historicos antes
        if self.df_train is None or self.default_col is None:
            raise RuntimeError("Llamá antes a load_data().")

        # Chequeo que los features existan en ambos datasets
        miss_hist = [c for c in features if c not in self.df_train.columns]
        miss_now = [c for c in features if c not in df_predict.columns]
        if miss_hist:
            raise KeyError(f"Faltan features en histórico: {miss_hist}")
        if miss_now:
            raise KeyError(f"Faltan features en actuales: {miss_now}")
        
        # 1) Selecciono el mejor estimador

        # Seteo matriz de features y de variable a predecir
        X_hist = self.df_train[features].copy().apply(pd.to_numeric, errors="coerce")
        y_hist = self.df_train[self.default_col].astype(int).copy()

        self._best = self._fit_and_select(X_hist, y_hist, features)
        best_est = self._best.estimator

        logging.debug(f"Mejor modelo: {self._best.name}")
        logging.debug(f"Mejores parámetros: {self._best.best_params}")

        # Predicción de riesgo (probabilidad de default) en bancos actuales
        X_now = df_predict[features].copy().apply(pd.to_numeric, errors="coerce")
        if hasattr(best_est, "predict_proba"):
            proba_default_hat = best_est.predict_proba(X_now)[:, 1]
            logging.debug(f"Predicción de probabilidad de default (predict_proba): {proba_default_hat}")
        else:
            # fallback para modelos sin predict_proba
            scores = best_est.decision_function(X_now)
            logging.debug(f"Predicción de scores (decision_function): {scores}")
            proba_default_hat = (scores - scores.min()) / (scores.max() - scores.min() + 1e-12)

        # 2) Elegir umbral EXTRA
        if isinstance(umbral_riesgo, (int, float)): #si t es fijo, no hago nada
            t_star = float(umbral_riesgo)
            t_method = "fixed(value)"
        else:
            method = str(umbral_riesgo).lower()
            t_method = method

            if method == "cost":
                # Coste con activos ACTUALES
                C_FP = float(threshold_kwargs.get("C_FP", 1.0))
                C_FN = float(threshold_kwargs.get("C_FN", 1.0))
                assets_now = pd.to_numeric(df_predict.get("total_assets", np.nan), errors="coerce").to_numpy()
                if np.isnan(assets_now).all():
                    logging.warning("No se encontró 'total_assets' en df_predict; 'cost' sin pesos.")
                    assets_now = np.ones_like(proba_default_hat)
                assets_now = np.nan_to_num(assets_now, nan=0.0)

                ts = np.linspace(0, 1, 501)

                def expected_cost(t):
                    pos = (proba_default_hat >= t)
                    neg = ~pos
                    return ( ((1 - proba_default_hat[pos]) * C_FP * assets_now[pos]).sum()
                           + (proba_default_hat[neg] * C_FN * assets_now[neg]).sum() )

                t_star = float(min(ts, key=expected_cost))
                logging.debug(f"Umbral (cost con activos actuales) = {t_star:.4f}  C_FP={C_FP} C_FN={C_FN}")

            elif method == "target_rate":
                rate = float(threshold_kwargs.get("target_rate", 0.10))
                rate = min(max(rate, 0.0), 1.0)
                if not (0.0 < rate < 1.0):
                    raise ValueError("target_rate debe estar entre 0 y 1 (excluidos).")
                t_star = float(np.quantile(proba_default_hat, 1 - rate))
                logging.debug(f"Umbral (target_rate={rate:.2%}) = {t_star:.4f}")

            else:
                # Youden/F1: usar predicciones IN-SAMPLE del histórico¿
                if hasattr(best_est, "predict_proba"):
                    y_proba_train = best_est.predict_proba(X_hist)[:, 1]
                else:
                    scores_tr = best_est.decision_function(X_hist)
                    y_proba_train = (scores_tr - scores_tr.min()) / (scores_tr.max() - scores_tr.min() + 1e-12)

                t_star = self.pick_threshold(
                    y_true=y_hist.to_numpy(),
                    y_proba=y_proba_train,
                    method=method,
                )
                logging.debug(f"Umbral ({method} in-sample histórico) = {t_star:.4f}")

        # Booleano: true si la proba de riesgo es 'demasiado alta'
        riesgo = (proba_default_hat >= t_star).astype(int)

        logging.debug(f"Riesgo calculado (umbral {t_star}): {riesgo}")

        # Calculo metricas
        assets_now = pd.to_numeric(df_predict['total_assets'], errors="coerce")
        total_assets = np.nansum(assets_now)
        risky_assets = np.nansum(assets_now[riesgo == 1])

        pct_entidades_riesgo = 100.0 * riesgo.mean() if len(riesgo) else 0.0
        total_assets_B = total_assets / 1e9 #ajusto a billions
        pct_activos_riesgo = 100.0 * (risky_assets / total_assets) if total_assets > 0 else 0.0
        risky_assets_B = risky_assets / 1e9 

        logging.info(f"Entidades en riesgo de default = {pct_entidades_riesgo:0.5f}%")
        logging.info(f"Activos en riesgo (USD B): {risky_assets_B:,.5f}  ({pct_activos_riesgo:0.5f}%)") 
        logging.info(f"Total de activos del sistema (USD B): {total_assets_B:,.5f}")
        logging.info(f"Umbral usado: {t_star:0.4f}  (método: {t_method})")

        return {
            "best_model_name": self._best.name,
            "best_cv_auc": self._best.best_score,
            "best_params": self._best.best_params,
            "threshold": float(t_star),
            "threshold_method": t_method,
            "pct_entidades_riesgo": float(pct_entidades_riesgo),
            "pct_activos_riesgo": float(pct_activos_riesgo),
            "activos_en_riesgo_usd_B": float(risky_assets_B),  
            "total_activos_usd_B": float(total_assets_B),
        }