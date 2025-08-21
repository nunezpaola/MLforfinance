from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import calinski_harabasz_score
from typing import List, Tuple, Dict, Optional


# ---------------------------------PREPROCESAMIENTO

def _to_numeric(series: pd.Series) -> pd.Series:
    if series.dtype == "O":
        s = series.astype(str).str.strip().str.replace(",", ".", regex=False)
    else:
        s = series
    result = pd.to_numeric(s, errors="coerce")
    if not isinstance(result, pd.Series):
        return pd.Series(result, index=series.index)
    return result


def filtro_outliers(df: pd.DataFrame, cols: List[str], whisker: float = 1.5
                    ) -> pd.DataFrame:
    """
    Filtra filas que caen fuera de [Q1 - whisker*IQR, Q3 + whisker*IQR] en
    cualquiera de las columnas 'cols'.
    """
    mask = pd.Series(True, index=df.index)
    for c in cols:
        # tomo cuantiles
        q1 = df[c].quantile(0.25) 
        q3 = df[c].quantile(0.75)

        # calculo el intercuartil y las bandas
        iqr = q3 - q1
        lo = q1 - whisker * iqr
        hi = q3 + whisker * iqr

        # aplico el filtro
        mask &= df[c].between(lo, hi)

        # print los que se eliminan
        if not df.loc[~mask, c].empty:
            print(f"Eliminados en {c}:\n{df.loc[~mask, c]}\n")

    return df.loc[mask].copy()

def preprocess_for_clustering(
    df: pd.DataFrame,
    feature_cols: List[str],
    id_col: str,
    label_col: Optional[str] = None,
    scale: bool = True,
    filter_outliers: bool = False,
    outlier_whisker: float = 1.5,
    scaler: Optional[object] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Devuelve:
      - work: DataFrame con [id_col, (label_col si existe)] + feature_cols
      - X:    matriz de features (escalada si scale=True)
    """
    work_cols = [id_col] + ([label_col] if label_col else []) + feature_cols
    work = df[work_cols].copy()

    # Formato numérico
    for c in feature_cols:
        work[c] = _to_numeric(work[c])

    # OPCIONAL: filtro outliers
    if filter_outliers:
        work = filtro_outliers(work, feature_cols, whisker=outlier_whisker)

    # Normalizo escala porque uso distancia euclidiana
    X = work[feature_cols].values
    if scale:
        if scaler is None:
            scaler = RobustScaler()
        X = scaler.fit_transform(X)
        X = pd.DataFrame(X, columns=feature_cols, index=work.index)
    else:
        X = pd.DataFrame(X, columns=feature_cols, index=work.index)

    return work, X

# ------------------------- SCORING PARA ELEGIR K ÓPTIMO

def get_clustering_scores(
    X: pd.DataFrame,
    k_min: int = 2,
    k_max: int = 10,
    random_state: int = 42,
) -> Tuple[Dict[int, float], Dict[int, float]]:
    scores_kmeans: Dict[int, float] = {}
    scores_agglo: Dict[int, float] = {}

    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, n_init=20, random_state=random_state)
        labels_km = km.fit_predict(X)
        scores_kmeans[k] = calinski_harabasz_score(X, labels_km)

        ag = AgglomerativeClustering(n_clusters=k, linkage="ward")
        labels_ag = ag.fit_predict(X)
        scores_agglo[k] = calinski_harabasz_score(X, labels_ag)

    return scores_kmeans, scores_agglo

def elegir_k_optimo(scores: Dict[int, float]) -> int:
    return int(max(scores.items(), key=lambda kv: kv[1])[0])

def plot_scores(scores: Dict[int, float], model_name: str, output_dir: str) -> str:
    '''
    Funcion auxiliar para ver como evoluciona el score de Calinski-Harabasz
    en funcion de K.
    '''
    os.makedirs(output_dir, exist_ok=True)
    ks = sorted(scores.keys())
    vals = [scores[k] for k in ks]
    plt.figure(figsize=(7, 5))
    plt.plot(ks, vals, marker="o", color= '#5D76CB')
    plt.title(f"Calinski–Harabasz vs K — {model_name}")
    plt.xlabel("K")
    plt.ylabel("CH score (↑ mejor)")
    plt.grid(True, alpha=0.3)
    out_path = os.path.join(output_dir, f"ps4_ch_{model_name.lower().replace(' ', '_')}.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    return out_path

# ---------------------- FITTING con K optimo + FORZANDO K=3

def fit_kmeans_agglo(
    X: pd.DataFrame,
    k: int,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    km = KMeans(n_clusters=k, n_init=20, random_state=random_state)
    labels_km = km.fit_predict(X)

    ag = AgglomerativeClustering(n_clusters=k, linkage="ward")
    labels_ag = ag.fit_predict(X)

    return labels_km, labels_ag

def _map_clusters_to_majority(y_true: pd.Series, y_pred: np.ndarray) -> Dict[int, str]:
    """
    Considero y_true a la clasficiación que estaba de antemano (MM, EQ, RF)
    y comparo con y_pred (la clasificación obtenida del modelo).
    """
    tmp = pd.DataFrame({"true": y_true.values, "cluster": y_pred})
    mapping = {}
    for cl, sub in tmp.groupby("cluster"):
        maj = sub["true"].value_counts(dropna=False).idxmax()
        mapping[int(cl)] = str(maj)
    return mapping

def forced_k3_differences(
    work: pd.DataFrame,
    X: pd.DataFrame,
    id_col: str,
    label_col: str,
) -> pd.DataFrame:
    """
    Fuerza K=3 en KMeans y Agglomerative y devuelve un DataFrame con los fondos cuyo
    tipo predicho (mayoritario) difiere del label original en al menos un modelo.
    """
    labels_km, labels_ag = fit_kmeans_agglo(X, k=3, random_state=42)
    out, _ = build_comparative(work, labels_km, labels_ag, id_col, label_col, feature_cols=list(X.columns))
    diff = out.loc[out["flag_raro"], [id_col, label_col, "km_pred_tipo", "ag_pred_tipo", "cluster_kmeans", "cluster_agglo"]].copy()
    return diff

def build_comparative(
    work: pd.DataFrame,
    labels_km: np.ndarray,
    labels_ag: np.ndarray,
    id_col: str,
    label_col: str,
    feature_cols: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compara los modelos y me devuelve tabla con diferencias
    """
    out = work[[id_col, label_col] + feature_cols].copy()
    out["cluster_kmeans"] = labels_km
    out["cluster_agglo"] = labels_ag

    map_km = _map_clusters_to_majority(out[label_col], labels_km)
    map_ag = _map_clusters_to_majority(out[label_col], labels_ag)
    out["km_pred_tipo"] = out["cluster_kmeans"].map(map_km)
    out["ag_pred_tipo"] = out["cluster_agglo"].map(map_ag)

    out["km_flag_diff"] = out["km_pred_tipo"] != out[label_col]
    out["ag_flag_diff"] = out["ag_pred_tipo"] != out[label_col]
    out["flag_raro"] = out["km_flag_diff"] | out["ag_flag_diff"]

    tab_km = pd.crosstab(out[label_col], out["cluster_kmeans"], rownames=["tipo"], colnames=["km_cluster"])
    tab_ag = pd.crosstab(out[label_col], out["cluster_agglo"], rownames=["tipo"], colnames=["ag_cluster"])
    tablas = pd.concat({"kmeans": tab_km, "agglo": tab_ag}, axis=1)
    return out, tablas
