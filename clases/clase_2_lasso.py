"""Clase 2: Regresión Regularizada Lasso y Ridge"""

# %%
# Imports
import statsmodels.api as sm
import sklearn.datasets as sk_ds
import sklearn.linear_model as sk_lm


# %%
# Creando sets de datos
X_1, y_1, coefs_1 = sk_ds.make_regression(n_samples=1000, n_features=6, bias=5.,
                                          noise=10., coef=True, random_state=1234)

X_2, y_2, coefs_2 = sk_ds.make_regression(n_samples=1000, n_features=6, bias=5.,
                                          noise=10., coef=True, n_informative=4,
                                          random_state=234)


# %%
# Probando con el set 1
ols_1 = sm.OLS(y_1, X_1).fit()

lr_1 = sk_lm.LinearRegression().fit(X_1, y_1)
lr_1_sgd = sk_lm.SGDRegressor(penalty=None).fit(X_1, y_1)

las_1 = sk_lm.Lasso().fit(X_1, y_1)
las_1_sgd = sk_lm.SGDRegressor(penalty='l1', alpha=1.).fit(X_1, y_1)

las_1_cv = sk_lm.LassoCV(cv=20).fit(X_1, y_1)

rid_1 = sk_lm.Ridge().fit(X_1, y_1)
rid_1_cv = sk_lm.RidgeCV(cv=20).fit(X_1, y_1)


print(ols_1.summary())
print('\nVerdaderos         ->', coefs_1)
print(f'\nLinearRegression() ->', lr_1.coef_)
print('SGDRegression      ->', lr_1_sgd.coef_)
print(f'\nLasso(lambda={las_1.alpha:.2f}) ->', las_1.coef_)
print('SGDRegressor       ->', las_1_sgd.coef_)
print(f'Lasso(lambda={las_1_cv.alpha_:.2f}) ->', las_1_cv.coef_)
print(f'\nRidge(lambda={rid_1.alpha:.2f}) ->', rid_1.coef_)
print(f'Ridge(lambda={rid_1_cv.alpha_:.2f}) ->', rid_1_cv.coef_)


# %%
# Probando con el set 2
ols_2 = sm.OLS(y_2, X_2).fit()

lr_2 = sk_lm.LinearRegression().fit(X_2, y_2)
lr_2_sgd = sk_lm.SGDRegressor(penalty=None).fit(X_2, y_2)

las_2 = sk_lm.Lasso().fit(X_2, y_2)
las_2_sgd = sk_lm.SGDRegressor(penalty='l1', alpha=1.).fit(X_2, y_2)

las_2_cv = sk_lm.LassoCV(cv=20).fit(X_2, y_2)

rid_2 = sk_lm.Ridge().fit(X_2, y_2)
rid_2_cv = sk_lm.RidgeCV(cv=20).fit(X_2, y_2)


print(ols_2.summary())
print('\nVerdaderos         ->', coefs_2)
print(f'\nLinearRegression() ->', lr_2.coef_)
print('SGDRegression      ->', lr_2_sgd.coef_)
print(f'\nLasso(lambda={las_2.alpha:.2f}) ->', las_2.coef_)
print('SGDRegressor       ->', las_2_sgd.coef_)
print(f'Lasso(lambda={las_2_cv.alpha_:.2f}) ->', las_2_cv.coef_)
print(f'\nRidge(lambda={rid_2.alpha:.2f}) ->', rid_2.coef_)
print(f'Ridge(lambda={rid_2_cv.alpha_:.2f}) ->', rid_2_cv.coef_)
