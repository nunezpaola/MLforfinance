from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.inspection import permutation_importance
from sklearn.calibration import calibration_curve

from ps.ps3 import AnalistaDeRiesgo
from mlfin.plotting import plot_roc_curve
from mlfin.printing import setup_logging, log_info, log_error, log_warning, log_debug

setup_logging(log_level="INFO", log_file="ps/ps3_logs.txt")

# Importo datos
path = './data/central_bank_data.h5'
df_train = pd.read_hdf(path, key="bank_defaults_FDIC")  # historicos (con labels)
df_predict = pd.read_hdf(path, key="regulated_banks")  # actuales (sin labels)

# target y features
target_col = "defaulter"
features_col = [
    "log_TA", "NI_to_TA", "Equity_to_TA", "NPL_to_TL", "REO_to_TA",
    "ALLL_to_TL", "core_deposits_to_TA", "brokered_deposits_to_TA",
    "liquid_assets_to_TA", "loss_provision_to_TL", "NIM", "assets_growth",
]

# seteo modelos y grids
knn = KNeighborsClassifier()
knn_grid = {"n_neighbors": list(range(3, 31))}
svc = SVC(probability=True, gamma="scale")
svc_grid = {"C": [1, 10, 100, 500, 1000], "kernel": ["linear", "rbf"]}
tree = DecisionTreeClassifier(random_state=42)
tree_grid = {"min_samples_split": list(range(2, 16))}
modelos_y_grids = [(knn, knn_grid), (svc, svc_grid), (tree, tree_grid)]

PALETTE = {
    "fixed":"#6c757d",
    "youden":"#1f77b4",
    "f1":"#ff7f0e",
    "cost":"#2ca02c",
    "target_rate":"#d62728",
}

def plot_pd_hist_with_thresholds(pd_hat, thresholds: dict[str, float],
                                  title="Probabilidad de default en bancos (actual)"):
    plt.figure(figsize=(8,5))
    plt.hist(pd_hat, bins=25, alpha=0.75, edgecolor="white")
    for name, t in thresholds.items():
        color = PALETTE.get(name, None)
        plt.axvline(t, linestyle="--", linewidth=2.0, label=f"{name}: {t:.3f}", color=color)
    plt.xlabel("Probabilidad de default (predicción)")
    plt.ylabel("Frecuencia")
    plt.title(title)
    plt.legend(frameon=False)
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig("ps/figures/pd_hist_with_thresholds.png", dpi=300)
    plt.show()

def plot_feature_importance_topk(feature_importances, feature_names, k: int = 10, title="Top features"):
    imp = np.array(feature_importances, dtype=float)
    names = np.array(feature_names)
    order = np.argsort(imp)[::-1][:k]
    imp_k = imp[order][::-1]     # reversed para barh de abajo hacia arriba
    names_k = names[order][::-1]

    fig, ax = plt.subplots(figsize=(8,5))
    bars = ax.barh(range(len(imp_k)), imp_k, edgecolor="white", color = PALETTE['fixed'])
    ax.set_yticks(range(len(imp_k)))
    ax.set_yticklabels(names_k)
    ax.set_xlabel("Importancia")
    ax.set_title(title + f" (top {k})")
    ax.grid(axis="x", alpha=0.25)

    # anotar valores
    for i, b in enumerate(bars):
        ax.text(b.get_width()*1.01, b.get_y()+b.get_height()/2, f"{imp_k[i]:.3f}",
                va="center", ha="left", fontsize=9)
    plt.tight_layout()
    plt.savefig("ps/figures/feature_importance.png", dpi=300)
    plt.show()

def plot_calibration_insample(model, X, y, title="Calibración (in-sample)"):
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X)[:, 1]
    else:
        s = model.decision_function(X)
        y_proba = (s - s.min())/(s.max()-s.min() + 1e-12)
    frac_pos, mean_pred = calibration_curve(y, y_proba, n_bins=10, strategy="quantile")
    plt.figure(figsize=(6,5))
    plt.plot(mean_pred, frac_pos, marker="o", label="In-sample", color = 'crimson')
    plt.plot([0,1],[0,1],"--", label="Perfect", color= 'grey')
    plt.xlabel("Probabilidad (predicción)")
    plt.ylabel("Frecuencia observada")
    plt.title(title)
    plt.legend(frameon=False)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("ps/figures/calibration_insample.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    # 1) Entreno y estimo
    analista = AnalistaDeRiesgo(modelos_y_grids, scoring="roc_auc", random_state=42)
    analista.load_data(df_train, target_col)

    method_cfgs = [
        ("fixed", {"fixed": 0.50}),
        ("youden", {}),
        ("f1", {}),
        ("cost", {"C_FP": 1.0, "C_FN": 20.0}),
        ("target_rate", {"target_rate": 0.10}),
    ]

    resultados = []
    thresholds = {}

    for method, kwargs in method_cfgs:
        log_info(f"\n==== Evaluando método de umbral: {method} ====")
        if method == "fixed":
            res = analista.get_report(df_predict, features_col,
                                      umbral_riesgo=kwargs["fixed"])
        else:
            res = analista.get_report(df_predict, features_col,
                                      umbral_riesgo=method,
                                      threshold_kwargs=kwargs)
        resultados.append({**res, "method": method})
        thresholds[method] = res["threshold"]

    df_res = pd.DataFrame(resultados)[
        ["method", "best_model_name", "best_cv_auc", "threshold",
         "pct_entidades_riesgo", "pct_activos_riesgo", "total_activos_usd_B"]
    ].sort_values(by="pct_activos_riesgo", ascending=False)

    log_info("\n==== Comparación de métodos de umbral (sobre df_predict) ====")
    log_info(df_res.to_string(index=False))

    # 2) PLOTS

    ###### ROC in-sample del histórico 
    best_pipe = analista._best.estimator  # tomo el último get_report
    plot_roc_curve(best_pipe, df_train[features_col], df_train[target_col].astype(int))

    ###### Calibración in-sample en histórico
    X_hist = df_train[features_col].apply(pd.to_numeric, errors="coerce")
    y_hist = df_train[target_col].astype(int)
    plot_calibration_insample(best_pipe, X_hist, y_hist, title="Calibration (in-sample, historical)")

    ###### Importancia de features
    clf = best_pipe.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        importances = np.abs(np.ravel(clf.coef_))
    else:
        X_hist = df_train[features_col].apply(pd.to_numeric, errors="coerce")
        y_hist = df_train[target_col].astype(int)
        pi = permutation_importance(best_pipe, X_hist, y_hist,
                                    n_repeats=10, random_state=42,
                                    scoring="roc_auc", n_jobs=-1)
        importances = pi.importances_mean

    plot_feature_importance_topk(importances, features_col, k=8, title="Top features")

    # Histograma de PD en actuales
    X_now = df_predict[features_col].apply(pd.to_numeric, errors="coerce")
    if hasattr(best_pipe, "predict_proba"):
        pd_hat_now = best_pipe.predict_proba(X_now)[:, 1]
    else:
        s_now = best_pipe.decision_function(X_now)
        pd_hat_now = (s_now - s_now.min())/(s_now.max()-s_now.min() + 1e-12)

    plot_pd_hist_with_thresholds(pd_hat_now, thresholds,
                                title="Distribución de probabilidad de default (en bancos actuales)")

