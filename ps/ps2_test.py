import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ps2 import BalancedPortfolio
from mlfin.printing import setup_logging
from mlfin.utils import get_allocations
from sklearn.linear_model import LassoCV
import logging

# Configurong logging
setup_logging(log_level=logging.INFO, log_file='ps/ps2_logs.txt')

# Desactivo logs DEBUG de matplotlib y otras librerías externas
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)
logging.getLogger('PIL.PngImagePlugin').setLevel(logging.WARNING)


def plot_series_comparison(asset_returns, factor_returns, assets_subset=None):
    """
    Grafica series temporales de ETFs vs factores Fama-French normalizados.

    Parameters
    ----------
    asset_returns : pd.DataFrame
        Retornos de activos
    factor_returns : pd.DataFrame
        Retornos de factores
    assets_subset : list, optional
        Lista de activos a graficar. Si None, usa todos.
    """
    plt.figure(figsize=(15, 10))

    # Lista de colores para las series
    colors = ['#440154', '#3b528b', '#21918c', '#5dc963', '#fde725']

    # Normalizo series (valor inicial = 100)
    asset_cumulative = (1 + asset_returns).cumprod() * 100
    factor_cumulative = (1 + factor_returns).cumprod() * 100

    if assets_subset:
        asset_cumulative = asset_cumulative[assets_subset]

    # Subplot 1: ETFs
    plt.subplot(2, 1, 1)
    for i, col in enumerate(asset_cumulative.columns):
        color_idx = i % len(colors)
        plt.plot(asset_cumulative.index, asset_cumulative[col],
                 label=col, linewidth=2, color=colors[color_idx])
    plt.title('Evolución acumulada de ETFs (Base 100)', fontsize=14,
              fontweight='bold')
    plt.ylabel('Valor acumulado')
    plt.legend(frameon=False)
    plt.grid(True, alpha=0.3)

    # Subplot 2: Factores
    plt.subplot(2, 1, 2)

    # Creo un segundo eje Y (eje derecho)
    ax1 = plt.gca()
    ax2 = ax1.twinx()

    # Grafica la primera columna (exceso de retorno) en el eje derecho
    first_col = factor_cumulative.columns[0]
    ax2.plot(factor_cumulative.index, factor_cumulative[first_col],
             label=f"{first_col} (eje derecho)", linewidth=2,
             color=colors[0], linestyle='--')
    ax2.set_ylabel(f'{first_col} (Base 100)', color=colors[0])
    ax2.tick_params(axis='y', labelcolor=colors[0])

    # Grafica el resto de factores en el eje izquierdo
    other_cols = factor_cumulative.columns[1:]
    for i, col in enumerate(other_cols):
        color_idx = (i + 1) % len(colors)
        ax1.plot(factor_cumulative.index, factor_cumulative[col],
                 label=col, linewidth=2, color=colors[color_idx])

    ax1.set_title('Evolución acumulada de factores Fama-French (Base 100)',
                  fontsize=14, fontweight='bold')
    ax1.set_xlabel('Fecha')
    ax1.set_ylabel('Valor acumulado (resto de factores)')

    # Combino leyendas de ambos ejes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, frameon=False)

    ax1.grid(True, alpha=0.3)

    plt.savefig('ps/figures/etfs_vs_factors.png', dpi=300)


def plot_real_vs_fitted(asset_excess, factor_returns, asset_name):
    """
    Grafica serie real vs fitted para un ETF específico.
    """
    # Merge datos
    df = pd.merge(asset_excess[[asset_name]], factor_returns,
                  left_index=True, right_index=True).dropna()
    X = df[factor_returns.columns].values
    y = df[asset_name].values
    # Ajuste Lasso
    model = LassoCV(cv=5, random_state=42).fit(X, y)
    beta = model.coef_
    y_fit = X.dot(beta)

    plt.figure(figsize=(10, 5))
    plt.plot(df.index, y, label='Real', linewidth=2, color='#3b528b')
    plt.plot(df.index, y_fit, label='Fitted', linewidth=2, linestyle='--',
             color='#21918c')
    plt.title(f'Real vs Fitted: {asset_name}')
    plt.xlabel('Fecha')
    plt.ylabel('Exceso de retorno sobre activo libre de riesgo')
    plt.legend(frameon=False)
    plt.grid(alpha=0.3)
    plt.savefig(f'ps/figures/{asset_name}_real_vs_fitted.png', dpi=300)


def plot_score_surface(asset_excess, factor_returns, n_samples=30):
    """
    Grafica superficie 3D de la función score vs ponderaciones y puntos de grilla.

    Parameters
    ----------
    asset_excess : pd.DataFrame
        Retornos en exceso de activos
    factor_returns : pd.DataFrame
        Retornos de factores
    n_samples : int
        Número de muestras para cada dimensión en el grid
    """
    # Preparo datos (mismo proceso que en BalancedPortfolio)
    df = pd.merge(asset_excess, factor_returns, left_index=True, right_index=True)
    df = df.dropna()

    assets = list(asset_excess.columns)
    factors = list(factor_returns.columns)
    X = df[factors].values

    # Calculo exposiciones usando Lasso
    exposures = pd.DataFrame(index=assets, columns=factors, dtype=float)
    for asset in assets:
        y = df[asset].values
        model = LassoCV(cv=5, random_state=42).fit(X, y)
        exposures.loc[asset] = model.coef_

    B = exposures.values.T * 100
    n_assets, n_factors = B.shape
    target_exposure = 1.0 / n_factors
    t = np.full(n_factors, target_exposure)

    # Creo grid solo para los primeros 3 activos (para visualización 3D)
    if n_assets >= 3:
        # Grid continuo
        w1_range = np.linspace(0, 1, n_samples)
        w2_range = np.linspace(0, 1, n_samples)
        W1, W2 = np.meshgrid(w1_range, w2_range)

        scores = np.zeros_like(W1)

        for i in range(n_samples):
            for j in range(n_samples):
                w1, w2 = W1[i, j], W2[i, j]
                if w1 + w2 <= 1:  # Restricción de suma
                    # Resto de pesos distribuido equitativamente
                    remaining = 1 - w1 - w2
                    w_remaining = remaining / (n_assets - 2) if n_assets > 2 else 0

                    weights = np.array([w1, w2] + [w_remaining] * (n_assets - 2))
                    portfolio_exposures = B @ weights
                    score = np.sum((portfolio_exposures - t) ** 2)
                    scores[i, j] = score
                else:
                    scores[i, j] = np.nan

        # Calcular puntos de grilla usando get_allocations
        grid_points = []
        grid_scores = []

        for allocation in get_allocations(n_assets):
            weights = np.array(allocation)
            portfolio_exposures = B @ weights
            score = np.sum((portfolio_exposures - t) ** 2)
            grid_points.append(weights)
            grid_scores.append(score)

        grid_points = np.array(grid_points)
        grid_scores = np.array(grid_scores)

        # Crear figura con subplots
        fig = plt.figure(figsize=(16, 6))

        # Subplot 1: Superficie 3D
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.plot_surface(W1, W2, scores, cmap='viridis', alpha=0.7)

        # Agregar puntos de grilla
        ax1.scatter(grid_points[:, 0], grid_points[:, 1], grid_scores,
                    c='red', s=50, alpha=0.8, label='Puntos Grilla')

        ax1.set_xlabel(f'Ponderación {assets[0]}')
        ax1.set_ylabel(f'Ponderación {assets[1]}')
        ax1.set_zlabel('Score (MSE)')
        ax1.set_title('Función score vs Ponderaciones\n(Primeros 2 activos)',
                      fontsize=12, fontweight='bold')
        ax1.legend(frameon=False)

        # Subplot 2: Vista 2D (contorno)
        ax2 = fig.add_subplot(122)
        contour = ax2.contour(W1, W2, scores, levels=20, cmap='viridis')
        ax2.clabel(contour, inline=True, fontsize=8)

        # Agregar puntos de grilla en 2D
        scatter = ax2.scatter(grid_points[:, 0], grid_points[:, 1],
                              c=grid_scores, cmap='viridis', s=50, alpha=0.8)

        ax2.set_xlabel(f'Ponderación {assets[0]}')
        ax2.set_ylabel(f'Ponderación {assets[1]}')
        ax2.set_title('Curvas de nivel del score con puntos de grilla',
                      fontsize=12, fontweight='bold')

        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax2)
        cbar.set_label('Score (MSE)')
        plt.savefig('ps/figures/score_surface.png', dpi=300)

    else:
        print("Se necesitan al menos 3 activos para visualización 3D")


if __name__ == "__main__":
    # Levanto datos
    prices = pd.read_csv('./data/selected_etfs.csv', index_col=0,
                         parse_dates=True)
    ff = pd.read_csv('./data/F-F_Research_Data_5_Factors_2x3_daily.csv',
                     index_col=0, parse_dates=True)

    # Calculo retornos
    asset_returns = prices.pct_change(fill_method=None).dropna()

    # Procesa datos de factores Fama-French
    factor_returns = ff.dropna()

    # Calcula exceso de retorno (retornos de activos - tasa libre de riesgo)
    asset_excess = asset_returns.subtract(factor_returns['RF'], axis=0)

    # Elimino la columna risk-free de los factores
    factor_returns = factor_returns.drop(columns=['RF'])

    # Filtro año 2017 y activos específicos
    assets_2017 = ['BOND', 'SUSA', 'DNL', 'XLF', 'XSLV']  # Lista de activos
    asset_returns_2017 = asset_returns.loc['2017-01-01':'2017-12-31', assets_2017]
    asset_excess_2017 = asset_excess.loc['2017-01-01':'2017-12-31', assets_2017]
    factor_returns_2017 = factor_returns.loc['2017-01-01':'2017-12-31']

    # Check que no hay NaN en los datos
    asset_excess_2017 = asset_excess_2017.dropna()
    factor_returns_2017 = factor_returns_2017.dropna()

    # Calculo portafolio balanceado
    bp = BalancedPortfolio(asset_excess_2017)
    optimal_weights, optimal_exposures = bp.get_balanced_portfolio(factor_returns_2017)

    # Display en el formato del trabajo
    print('\n--- Portfolio Balanceado (grilla) ---')
    print('Weights   :', [f"{w*100:.2f}%" for w in bp.weights_grilla])
    print('Exposures :', [f"{e*100:.2f}%" for e in bp.exposure_grilla])

    print('\n--- Portfolio Balanceado (convexa) ---')
    if hasattr(bp, 'weights_conv') and bp.weights_conv is not None:
        print('Weights   :', [f"{w*100:.2f}%" for w in bp.weights_conv])
        print('Exposures :', [f"{e*100:.2f}%" for e in bp.exposure_conv])
    else:
        print('Optimización convexa no disponible')

    # Generar gráficos
    print("\n=== GENERANDO GRÁFICOS ===")

    # Gráfico 1: Series temporales ETFs vs Factores
    print("\nGráfico 1: Evolución temporal de ETFs y Factores...")
    plot_series_comparison(asset_returns_2017, factor_returns_2017, assets_2017)

    # Gráfico 2: Superficie de la función score
    print("\nGráfico 2: Superficie de función score y puntos de grilla...")
    plot_score_surface(asset_excess_2017, factor_returns_2017, n_samples=30)

    # Gráfico 3: Real vs Fitted para un ETF que tiene signo opuesto al del trabajo
    print("\nGráfico 3: Real vs Fitted para XSLV...")
    plot_real_vs_fitted(asset_excess_2017, factor_returns_2017, 'XSLV')
