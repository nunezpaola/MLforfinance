"""Clase 5: Yield Curve Analysis"""

# %%
# Imports
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.decomposition as sk_de

# Vamos a descomponer los movimientos de ls Yield Curva à la
# "Common Factors Affecting Bond Returns" Litterman-Scheinkman (1991)


# %%
# Cargo datos
yields = pd.read_csv('./data/bonos_usd_ytms.csv', index_col=0, parse_dates=True)
# Hacemos trampa acá porque deberían ser zero-coupon las tasas a utilizar, no
# las YTMs de los bonos, pero para el ejemplo de la técnica sirve.

delta_yields = yields.diff()
# A diferencia de los retornos de precios, el invariante en Fixed-Income es la
# variación de tasas.
delta_yields.dropna(inplace=True)

tgt_data = delta_yields.loc[:'2018-05', :]


# %%
# Armo PCA
pca = sk_de.PCA(n_components=3).fit(tgt_data)
plt.legend(plt.plot(pca.components_.T),
           (f'{v:.2%}' for v in pca.explained_variance_ratio_))
plt.title(f'Primeros {len(pca.components_)} componentes de la Yield Curve')
plt.show()
