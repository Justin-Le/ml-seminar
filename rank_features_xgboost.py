import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from sklearn import cross_validation
import xgboost as xgb
from sklearn.metrics import roc_auc_score

# Required for saving plots
matplotlib.use("Agg")

training = pd.read_csv("./train.csv", index_col=0)
test = pd.read_csv("./test.csv", index_col=0)

print(training.shape)
print(test.shape)

# Replace -999999 in var3 column with most common value 2 
training = training.replace(-999999,2)

X = training.iloc[:,:-1]
y = training.TARGET

# Add zeros per row as extra feature
# X['n0'] = (X == 0).sum(axis=1)

from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_classif, chi2
from sklearn.preprocessing import Binarizer, scale

########################################
# Feature Selection
########################################

# Percentile
p = 75

# Scale to sample mean and unit variance
X_bin = Binarizer().fit_transform(scale(X))

# Chi-squared statistics of non-negative feature
selectF_classif = SelectPercentile(f_classif, percentile=p).fit(X, y)

# ANOVA f-value between label and feature
selectChi2 = SelectPercentile(chi2, percentile=p).fit(X_bin, y)

# Select features
f_classif_selected = selectF_classif.get_support()
f_classif_selected_features = [f for i, f in enumerate(X.columns) if f_classif_selected[i]]
chi2_selected = selectChi2.get_support()
chi2_selected_features = [f for i, f in enumerate(X.columns) if chi2_selected[i]]
selected = chi2_selected & f_classif_selected
features = [f for f, s in zip(X.columns, selected) if s]
X_sel = X[features]

print('F_classif selected {} features {}.'.format(f_classif_selected.sum(), 
       f_classif_selected_features))
print('Chi2 selected {} features {}.'.format(chi2_selected.sum(), 
       chi2_selected_features))
print('Chi2 & F_classif selected {} features'.format(selected.sum()))
print(features)

########################################
# Fitting
########################################

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X_sel, y, random_state=1301, 
                                                                     stratify=y, test_size=0.4)

ratio = float(np.sum(y == 1)) / np.sum(y==0)

clf = xgb.XGBClassifier(missing=9999999999,
                        max_depth = 5,
                        n_estimators=1000,
                        learning_rate=0.1, 
                        nthread=4,
                        subsample=1.0,
                        colsample_bytree=0.5,
                        min_child_weight = 3,
                        scale_pos_weight = ratio,
                        reg_alpha=0.03,
                        seed=1301)
                
clf.fit(X_train, y_train, early_stopping_rounds=50, eval_metric="auc",
        eval_set=[(X_train, y_train), (X_test, y_test)])
        
print('Overall AUC:', 
      roc_auc_score(y, clf.predict_proba(X_sel, ntree_limit=clf.best_iteration)[:,1]))

test['n0'] = (test == 0).sum(axis=1)
sel_test = test[features]    
y_pred = clf.predict_proba(sel_test, ntree_limit=clf.best_iteration)

# submission = pd.DataFrame({"ID": test.index, "TARGET": y_pred[:, 1]})
# submission.to_csv("submission.csv", index=False)

mapFeat = dict(zip(["f" + str(i) for i in range(len(features))], features))
ts = pd.Series(clf.booster().get_fscore())
ts.sort_values()[-15:].plot(kind="barh", title=("features importance"))

featp = ts.sort_values()[-15:].plot(kind='barh', x='feature', y='fscore', 
                                    legend=False, figsize=(6, 10))
plt.title('XGBoost Feature Importance')
fig_featp = featp.get_figure()
fig_featp.savefig('feature_importance_xgb.png', bbox_inches='tight', pad_inches=1)
