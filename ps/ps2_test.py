import pandas as pd
from ps2 import BalancedPortfolio
from mlfin.printing import setup_logging
import logging

# Configurar logging
setup_logging(log_level=logging.INFO, log_file='ps/ps2_logs.txt')

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
