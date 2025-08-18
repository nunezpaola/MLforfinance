"""Clase 4: Ensemble Learning para clasificación"""

# %%
# Imports
import pandas as pd
import sklearn.model_selection as sk_ms
import sklearn.ensemble as sk_en
import xgboost as xgb

from mlfin.plotting import plot_roc_curve, plot_feature_importance
from mlfin.printing import print_classification_metrics


# %%
# Cargamos y procesamos los datos
data = pd.read_csv('./data/default.csv')

y_tk = 'default'
X_tks = ['student', 'balance', 'income']
X_tr, X_te, y_tr, y_te = sk_ms.train_test_split(data.loc[:, X_tks],
                                                data.loc[:, y_tk],
                                                test_size=.2, random_state=123)
# Corresponde hacer alguna estandarización?


# %%
# Definimos los modelos
rf_cv = sk_ms.GridSearchCV(sk_en.RandomForestClassifier(),
                           {'n_estimators': [50, 100],
                            'min_samples_split': range(2, 6)},
                           cv=5, scoring='roc_auc')

xgb_cv = sk_ms.GridSearchCV(xgb.XGBClassifier(),  #xgb.XGBClassifier(device="cuda"),
                            {'n_estimators': [50, 100],
                             'max_depth': range(1, 6)},
                            cv=5, scoring='roc_auc')


# %%
# Armamos el Soft VotingClassifier
vc = sk_en.VotingClassifier([('rf', rf_cv), ('xgb', xgb_cv)], voting='soft')
vc.fit(X_tr, y_tr)


# %%
# Hacemos reporte final
print('\nTesting Results:')
plot_roc_curve(vc, X_te, y_te)
print_classification_metrics(vc, X_te, y_te)


# %%
# Feature importance según Random Forest
feat_imp = vc.estimators_[0].best_estimator_.feature_importances_
plot_feature_importance(feat_imp, X_tks)
