from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colormaps as cmaps
from matplotlib.patches import Polygon, Ellipse, Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from ps import ps4  # core
from mlfin.printing import setup_logging, log_info, log_error, log_warning, log_debug  # type: ignore
setup_logging(log_level="INFO", log_file="ps/ps4_logs.txt")

# Constantes
CSV_PATH = "./data/fondos_comunes.csv"
ID_COL = "ID"
LABEL_COL = "TIPO"
FEATURE_COLS = ["VOLAT", "r3m", "r6m"]
OUTPUT_DIR = "./ps/outputs"
K_MIN, K_MAX = 2, 10
APLICAR_FILTRO_OUTLIERS = True
WHISKER = 1.0

# ---------- helpers para superficies ----------
def _ellipsoid_from_points_3d(pts: np.ndarray):
    mu = pts.mean(axis=0)
    cov = np.cov(pts.T) if len(pts) > 1 else np.eye(3)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    radii = 2 * np.sqrt(np.maximum(vals, 1e-12)) 
    return mu, vecs, radii

# ---------- plot 3D ----------


def plot_3d(
    work: pd.DataFrame,
    label_col: str,
    model_labels: np.ndarray,
    xcol: str,
    ycol: str,
    zcol: str,
    title: str,
    outfile: str,
) -> None:
    """
    Superficies 3D por clase original (convex hull si es posible, sino elipsoide 2σ) +
    puntos del modelo coloreados por cluster.
    """
    cmap_poly = cmaps['Set2']
    cmap_pts  = cmaps['tab10']

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    proxy_patches: list[Patch] = []

    # 1) Superficies del df original
    labels = [l for l in work[label_col].dropna().unique()]
    for i, lab in enumerate(labels):
        sub = work[work[label_col] == lab]
        pts = sub[[xcol, ycol, zcol]].values
        face = cmap_poly(i / max(1, len(labels)))
        edge = face

        drew_surface = False
        if len(pts) >= 4:
            try:
                from scipy.spatial import ConvexHull  # opcional
                hull = ConvexHull(pts)
                faces = [pts[s] for s in hull.simplices]
                poly3d = Poly3DCollection(
                    faces, facecolors=face, edgecolors=edge, linewidths=0.4, alpha=0.18, zorder=1
                )
                ax.add_collection3d(poly3d)
                proxy_patches.append(Patch(facecolor=face, edgecolor=edge, alpha=0.25, label=f"orig:{lab}"))
                drew_surface = True
            except Exception:
                pass

        if not drew_surface and len(pts) >= 1:
            # Elipsoide 2σ como fallback
            mu, vecs, radii = _ellipsoid_from_points_3d(pts)
            u = np.linspace(0, 2*np.pi, 40)
            v = np.linspace(0, np.pi, 20)
            xs = np.outer(np.cos(u), np.sin(v))
            ys = np.outer(np.sin(u), np.sin(v))
            zs = np.outer(np.ones_like(u), np.cos(v))
            sphere = np.stack([xs, ys, zs], axis=-1)  # (..., 3)
            # escalar por radios y rotar
            ell = sphere @ np.diag(radii) @ vecs.T + mu
            surf = ax.plot_surface(
                ell[..., 0], ell[..., 1], ell[..., 2],
                rstride=1, cstride=1, linewidth=0, antialiased=True,
                alpha=0.18, color=face
            )
            proxy_patches.append(Patch(facecolor=face, edgecolor=edge, alpha=0.25, label=f"orig:{lab}"))

    # 2) Puntos del modelo por cluster
    clusters = np.unique(model_labels)
    for cl in clusters:
        idx = (model_labels == cl)
        color = cmap_pts(int(cl) % 10)
        ax.scatter(
            work.loc[idx, xcol], work.loc[idx, ycol], work.loc[idx, zcol],
            s=24, c=[color], depthshade=True, edgecolor="k", linewidths=0.2,
            label=f"model:{cl}", zorder=2
        )

    ax.set_xlabel(xcol); ax.set_ylabel(ycol); ax.set_zlabel(zcol)
    ax.set_title(title)
    ax.grid(True, alpha=0.15)
    # Leyenda combinando proxies (superficies) y handles reales (scatters)
    handles, labels_txt = ax.get_legend_handles_labels()
    handles = proxy_patches + handles
    labels_txt = [p.get_label() for p in proxy_patches] + labels_txt
    ax.legend(handles, labels_txt, frameon=False, ncol=2, loc="upper left")
    fig.tight_layout(); fig.savefig(outfile, dpi=150); plt.close(fig)


def main():
    # 1) Cargo datos y creo id
    data = pd.read_csv(CSV_PATH)
    data[ID_COL] = data[LABEL_COL].astype(str) + "_" + data.index.astype(str)

    # 1.1) Analisis xploratorio: scatter para VOL vs r3m y VOL vs r6m usando
    #  clasificacion original
    plt.figure(figsize=(16, 6))
    categories = data['TIPO'].unique()
    custom_colors = ['#FF9EC4', '#BD7EBE', '#5D76CB', '#A9A9A9']
    color_dict = {cat: custom_colors[i % len(custom_colors)] for i, cat in enumerate(categories)}

    # PLOT EXPLOTATORIO
    for i, retorno in enumerate(['r3m', 'r6m']):
        plt.subplot(1, 2, i+1)
        for cat in categories:
            subset = data[data['TIPO'] == cat]
            plt.scatter(
                subset['VOLAT'], subset[retorno],
                color=color_dict[cat], label=cat, alpha=0.90,  linewidth=0.5
            )

        plt.title(f"Volatilidad vs retorno a {retorno[1]} meses")
        plt.xlabel("Volatilidad del último mes (VOLAT)")
        plt.ylabel(f"Retorno a {retorno[1]} meses")
        plt.grid(True, alpha=0.3)
        plt.legend(frameon=False)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/ps4_scatter_volat_comparison.png", dpi=150)
    plt.close()

    # 2) Preproceso
    work, X = ps4.preprocess_for_clustering(
        df=data,
        feature_cols=FEATURE_COLS,
        id_col=ID_COL,
        label_col=LABEL_COL,
        scale=True,
        filter_outliers=APLICAR_FILTRO_OUTLIERS,
        outlier_whisker=WHISKER,
    )

    # 3) Genero el score
    ch_km, ch_ag = ps4.get_clustering_scores(X, k_min=K_MIN, k_max=K_MAX, random_state=42)
    ps4.plot_scores(ch_km, "KMeans", OUTPUT_DIR)
    ps4.plot_scores(ch_ag, "Agglomerative", OUTPUT_DIR)
    k_km = ps4.elegir_k_optimo(ch_km)
    k_ag = ps4.elegir_k_optimo(ch_ag)
    log_info(f"[KMeans] Mejor K por CH: {k_km} (CH={ch_km[k_km]:.2f})")
    log_info(f"[Agglomerative] Mejor K por CH: {k_ag} (CH={ch_ag[k_ag]:.2f})")

    # 4) Elijo por criterio de codo + dominio de variables que k_optimo=4
    k_best = 4

    # 4) Fitteo con el k optimo y con k=3
    labels_km_opt, labels_ag_opt = ps4.fit_kmeans_agglo(X, k=k_best, random_state=42)
    labels_km_3, labels_ag_3 = ps4.fit_kmeans_agglo(X, k=3, random_state=42)
    comparativo, tablas = ps4.build_comparative(work, labels_km_opt, labels_ag_opt, ID_COL, 
                                                LABEL_COL, FEATURE_COLS)

    log_info(f'Comparativo: \n {comparativo}')
    # exporto el comparativo a csv para verlo
    comparativo.to_csv(f"{OUTPUT_DIR}/ps4_comparativo.csv", index=False)
    log_info(f'Tablas: \n {tablas}')

    # calculo diferencias entre kmeans y agglomerative
    tab_kmeans = tablas.loc[:, 'kmeans']
    tab_agglo = tablas.loc[:, 'agglo']
    diferencias = tab_kmeans - tab_agglo
    log_info(f'Diferencias (KMeans - Agglomerative): \n{diferencias}')

    # -------- PLOTS
    plot_3d(
        work, LABEL_COL, labels_km_opt,
        xcol="VOLAT", ycol="r3m", zcol="r6m",
        title=f"Clusterización base vs KMeans (K={k_best})",
        outfile=f"{OUTPUT_DIR}/ps4_3dvs_kmeans_opt.png",
    )
    plot_3d(
        work, LABEL_COL, labels_ag_opt,
        xcol="VOLAT", ycol="r3m", zcol="r6m",
        title=f"Clusterización base vs Agglo (K={k_best})",
        outfile=f"{OUTPUT_DIR}/ps4_3dvs_agglo_opt.png",
    )

    plot_3d(
        work, LABEL_COL, labels_km_3,
        xcol="VOLAT", ycol="r3m", zcol="r6m",
        title="Clusterización base vs KMeans (K=3)",
        outfile=f"{OUTPUT_DIR}/ps4_3dvs_kmeans_k3.png",
    )
    plot_3d(
        work, LABEL_COL, labels_ag_3,
        xcol="VOLAT", ycol="r3m", zcol="r6m",
        title="Clusterización base vs Agglo (K=3)",
        outfile=f"{OUTPUT_DIR}/ps4_3dvs_agglo_k3.png",
    )

    # 5) Diferencias con K=3
    dif_k3 = ps4.forced_k3_differences(work, X, id_col=ID_COL, label_col=LABEL_COL)
    log_info(f"[RESUMEN] Diferencias con K=3: \n {dif_k3}")


if __name__ == "__main__":
    main()
