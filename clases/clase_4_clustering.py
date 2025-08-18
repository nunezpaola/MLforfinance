"""Clase 4: Clustering"""

# %%
# Imports
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.model_selection as sk_ms
import sklearn.cluster as sk_cl
import sklearn.mixture as sk_mx
import sklearn.metrics as sk_mt


# %%
# Obtengo datos y configuro
prices = pd.read_csv('./data/bonos_usd_prices.csv', index_col=0, parse_dates=True)
rets = prices.pct_change()
rets.dropna(inplace=True)


# %%
# Configuro script
suiza = slice('2018-01-19', '2018-02-16')
hoy = slice('2019-09-13', '2019-10-11')

# Anuncio FMI => 2018-05-08
fmi_pre = slice('2018-04-06', '2018-05-04')
fmi_post = slice('2018-05-11', '2018-06-08')

# PASO => 2019-08-11
paso_pre = slice('2019-07-12', '2019-08-09')
paso_post = slice('2019-08-16', '2019-09-13')


# %%
# Busco cantidad de clusters con los Modelos
data = rets.loc[hoy, :].transpose()  # ACA Jugar con los períodos cambiando 'hoy'
                                     # por el que quieren
data.columns = data.columns.astype(str)

# K-Means
ss_dist = []
K = range(2, 10)
for k in K:
    km = sk_cl.KMeans(n_clusters=k, n_init='auto')
    km.fit(data)  # hm.predict(data) arroja los clusters a los que corresponde
                  # cada observación (ver más abajo fit_predict que hace lo mismo
                  # en un sólo paso.
    ss_dist.append(km.inertia_)

plt.plot(K, ss_dist, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances_to_closest_cluster_center')
plt.title('Elbow Method For Optimal k - K-Means')
plt.show()

# Agglomerative Hierchical Clustering
ch_score = []
for k in K:
    hc = sk_cl.AgglomerativeClustering(n_clusters=k)
    labels = hc.fit_predict(data)  # Acá obtengo clusters para cada observación
    ch_score.append(sk_mt.calinski_harabasz_score(data, labels))

plt.plot(K, ch_score, 'bx-')
plt.xlabel('k')
plt.ylabel('Calinski-Harabasz Score')
plt.title('Calinski-Harabasz Score Method For Optimal k - AHC')
plt.show()

# Gaussian Mixtures
gm_sel = sk_ms.GridSearchCV(sk_mx.GaussianMixture(), cv=5,
                            param_grid={'n_components': K})
gm_sel.fit(data)
print('\n', gm_sel.best_estimator_)
