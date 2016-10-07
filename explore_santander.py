# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pdb

# Import Seaborn for plotting
# and ignore all warnings
import warnings
warnings.filterwarnings("ignore")
import seaborn as sns

sns.set(style="white", color_codes=True)

# Load data as Pandas dataframes
train = pd.read_csv("./train.csv")
test = pd.read_csv("./test.csv")

train.head()
# Press shift+enter to execute this cell

# Targets: customer satisfaction
# 0 = happy, 1 = unhappy
# Over 96% are happy 
df = pd.DataFrame(train.TARGET.value_counts())
df['Percentage'] = 100*df['TARGET']/train.shape[0]
df

# var3: nationality of customer

# Ten most common values
train.var3.value_counts()[:10]

# -999999 = unknown nationality
train.loc[train.var3==-999999].shape

# Replace -999999 with most common value (nationality = 2) 
train = train.replace(-999999, 2)
train.loc[train.var3==-999999].shape


# Create new feature: number of zeros in a row
X = train.iloc[:, :-1]
y = train.TARGET
X['n0'] = (X==0).sum(axis=1)
train['n0'] = X['n0']

# num_var4: the number of products that a customer has purchased
train.num_var4.hist(bins=100)
plt.xlabel('Number of bank products')
plt.ylabel('Number of customers in train')
plt.title('Most customers have 1 product with the bank')
plt.show()

# Customer satisfaction versus number of products purchased
sns.FacetGrid(train, hue="TARGET", size=6).map(plt.hist, "num_var4").add_legend()
plt.title('Unhappy customers purchased fewer products')
plt.show()
train[train.TARGET==1].num_var4.hist(bins=6)
plt.title('Customer satisfaction versus number of products purchased');

# Var38 is suspected to be the mortage value with the bank. 
# If the mortage is with another bank the national average is used. 
# See https://www.kaggle.com/c/santander-customer-satisfaction/forums/t/19895/var38-is-mortgage-value
# [dmi3kno](https://www.kaggle.com/dmi3kno) says that var38 is value of the customer: [https://www.kaggle.com/cast42/santander-customer-satisfaction/exploring-features/comments#115223](https://www.kaggle.com/cast42/santander-customer-satisfaction/exploring-features/comments#115223)
train.var38.describe()

# var38 for unhappy customers
train.loc[train['TARGET']==1, 'var38'].describe()

# Histogram for var38 is not normal-distributed
train.var38.hist(bins=1000);
train.var38.map(np.log).hist(bins=1000);

# Identify the anomaly: 
# a spike between values 11 and 12 of the distribution
train.var38.map(np.log).mode()

# Most common values for var38
train.var38.value_counts()

# Most common value is close to the mean of the other values
train.var38[train['var38'] != 117310.979016494].mean()

# Excluding the most common value causes the
# distribution to become normal (in log-scale)
train.loc[~np.isclose(train.var38, 117310.979016), 'var38'].value_counts()
train.loc[~np.isclose(train.var38, 117310.979016), 'var38'].map(np.log).hist(bins=100);

# Above plot suggest we split up var38 into two variables
# var38mc == 1 when var38 has the most common value and 0 otherwise
# logvar38 is log transformed feature when var38mc is 0, zero otherwise
train['var38mc'] = np.isclose(train.var38, 117310.979016)
train['logvar38'] = train.loc[~train['var38mc'], 'var38'].map(np.log)
train.loc[train['var38mc'], 'logvar38'] = 0

# Check for NaN
print('Number of nan in var38mc', train['var38mc'].isnull().sum())
print('Number of nan in logvar38',train['logvar38'].isnull().sum())

# var15
# The most important feature for XGBoost is var15. According to [a Kaggle form post](https://www.kaggle.com/c/santander-customer-satisfaction/forums/t/19291/data-dictionary/110414#post110414)
#     var15 is the age of the customer. Let's explore var15
train['var15'].describe()

#Looks more normal, plot the histogram
train['var15'].hist(bins=100);

# Let's look at the density of the age of happy/unhappy customers
sns.FacetGrid(train, hue="TARGET", size=6)    .map(sns.kdeplot, "var15")    .add_legend()
plt.title('Unhappy customers are slightly older');

# Explore the interaction between var15 (age) and var38
sns.FacetGrid(train, hue="TARGET", size=10)    .map(plt.scatter, "var38", "var15")    .add_legend();

sns.FacetGrid(train, hue="TARGET", size=10)    .map(plt.scatter, "logvar38", "var15")    .add_legend()
plt.ylim([0,120]); # Age must be positive ;-)

# Exclude most common value for var38 
sns.FacetGrid(train[~train.var38mc], hue="TARGET", size=10)    .map(plt.scatter, "logvar38", "var15")    .add_legend()
plt.ylim([0,120]);

# What is distribution of the age when var38 has it's most common value ?
sns.FacetGrid(train[train.var38mc], hue="TARGET", size=6)    .map(sns.kdeplot, "var15")    .add_legend();

# What is density of n0 ?
sns.FacetGrid(train, hue="TARGET", size=6)    .map(sns.kdeplot, "n0")    .add_legend()
plt.title('Unhappy customers have a lot of features that are zero');

########################################
# Feature selection
########################################

from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_classif,chi2
from sklearn.preprocessing import Binarizer, scale

p = 3

X_bin = Binarizer().fit_transform(scale(X))
selectChi2 = SelectPercentile(chi2, percentile=p).fit(X_bin, y)
selectF_classif = SelectPercentile(f_classif, percentile=p).fit(X, y)

chi2_selected = selectChi2.get_support()
chi2_selected_features = [ f for i,f in enumerate(X.columns) if chi2_selected[i]]
print('Chi2 selected {} features {}.'.format(chi2_selected.sum(),
   chi2_selected_features))
f_classif_selected = selectF_classif.get_support()
f_classif_selected_features = [ f for i,f in enumerate(X.columns) if f_classif_selected[i]]
print('F_classif selected {} features {}.'.format(f_classif_selected.sum(),
   f_classif_selected_features))
selected = chi2_selected & f_classif_selected
print('Chi2 & F_classif selected {} features'.format(selected.sum()))
features = [ f for f,s in zip(X.columns, selected) if s]
print (features)

# Make a dataframe with the selected features and the target variable
X_sel = train[features + ['TARGET']]

# var36
X_sel['var36'].value_counts()

# var36 is most of the times 99 or [0,1,2,3]
# Let's plot the density in function of the target variabele
sns.FacetGrid(train, hue="TARGET", size=6)    .map(sns.kdeplot, "var36")    .add_legend()
plt.title('If var36 is 0,1,2 or 3 => less unhappy customers');

# In above plot we see that the density of unhappy custormers is lower when var36 is not 99
# var36 in function of var38 (most common value excluded) 
sns.FacetGrid(train[~train.var38mc], hue="TARGET", size=10)    .map(plt.scatter, "var36", "logvar38")    .add_legend();

# Let's seperate that in two plots
sns.FacetGrid(train[(~train.var38mc) & (train.var36 < 4)], hue="TARGET", size=10)    .map(plt.scatter, "var36", "logvar38")    .add_legend()
plt.title('If var36==0, only happy customers');

# Let's plot the density in function of the target variabele, when var36 = 99
sns.FacetGrid(train[(~train.var38mc) & (train.var36 ==99)], hue="TARGET", size=6)    .map(sns.kdeplot, "logvar38")    .add_legend();

# num_var5
train.num_var5.value_counts()
train[train.TARGET==1].num_var5.value_counts()
train[train.TARGET==0].num_var5.value_counts()

sns.FacetGrid(train, hue="TARGET", size=6).map(plt.hist, "num_var5")    .add_legend();
sns.FacetGrid(train, hue="TARGET", size=6).map(sns.kdeplot, "num_var5")    .add_legend();

plt.show()

# Features to build
# num_products_geq_3
# value_is_117310
# age_geq_40
# n0_geq_350
# var36_leq_abs5
# num_var5_is_0_or_3

# num_products = num_var4
# value = var38
# age = var15

"""
sns.pairplot(train[['var15','var36','logvar38','TARGET']], hue="TARGET", size=2, diag_kind="kde");

train[['var15','var36','logvar38','TARGET']].boxplot(by="TARGET", figsize=(12, 6));

# A final multivariate visualization technique pandas has is radviz
# Which puts each feature as a point on a 2D plane, and then simulates
# having each sample attached to those points through a spring weighted
# by the relative value for that feature
from pandas.tools.plotting import radviz
radviz(train[['var15','var36','logvar38','TARGET']], "TARGET");

# # now look at all 8 features together
features

radviz(train[features+['TARGET']], "TARGET");

sns.pairplot(train[features+['TARGET']], hue="TARGET", size=2, diag_kind="kde");
# # Correlations
cor_mat = X.corr()

f, ax = plt.subplots(figsize=(15, 12))
# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(cor_mat,linewidths=.5, ax=ax);

cor_mat = X_sel.corr()

f, ax = plt.subplots(figsize=(15, 12))
# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(cor_mat,linewidths=.5, ax=ax);

# only important correlations and not auto-correlations
threshold = 0.7
important_corrs = (cor_mat[abs(cor_mat) > threshold][cor_mat != 1.0])     .unstack().dropna().to_dict()
unique_important_corrs = pd.DataFrame(
    list(set([(tuple(sorted(key)), important_corrs[key]) \
    for key in important_corrs])), columns=['attribute pair', 'correlation'])
# sorted by absolute value
unique_important_corrs = unique_important_corrs.ix[
    abs(unique_important_corrs['correlation']).argsort()[::-1]]
unique_important_corrs
# Recipe from https://github.com/mgalardini/python_plotting_snippets/blob/master/notebooks/clusters.ipynb
import matplotlib.patches as patches
from scipy.cluster import hierarchy
from scipy.stats.mstats import mquantiles
from scipy.cluster.hierarchy import dendrogram, linkage

# Correlate the data
# also precompute the linkage
# so we can pick up the 
# hierarchical thresholds beforehand

from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler

# scale to mean 0, variance 1
train_std = pd.DataFrame(scale(X_sel))
train_std.columns = X_sel.columns
m = train_std.corr()
l = linkage(m, 'ward')

# Plot the clustermap
# Save the returned object for further plotting
mclust = sns.clustermap(m,
               linewidths=0,
               cmap=plt.get_cmap('RdBu'),
               vmax=1,
               vmin=-1,
               figsize=(14, 14),
               row_linkage=l,
               col_linkage=l)

# Threshold 1: median of the
# distance thresholds computed by scipy
t = np.median(hierarchy.maxdists(l))

# Plot the clustermap
# Save the returned object for further plotting
mclust = sns.clustermap(m,
               linewidths=0,
               cmap=plt.get_cmap('RdBu'),
               vmax=1,
               vmin=-1,
               figsize=(12, 12),
               row_linkage=l,
               col_linkage=l)

# Draw the threshold lines
mclust.ax_col_dendrogram.hlines(t,
                               0,
                               m.shape[0]*10,
                               colors='r',
                               linewidths=2,
                               zorder=1)
mclust.ax_row_dendrogram.vlines(t,
                               0,
                               m.shape[0]*10,
                               colors='r',
                               linewidths=2,
                               zorder=1)

# Extract the clusters
clusters = hierarchy.fcluster(l, t, 'distance')
for c in set(clusters):
    # Retrieve the position in the clustered matrix
    index = [x for x in range(m.shape[0])
             if mclust.data2d.columns[x] in m.index[clusters == c]]
    # No singletons, please
    if len(index) == 1:
        continue

    # Draw a rectangle around the cluster
    mclust.ax_heatmap.add_patch(
        patches.Rectangle(
            (min(index),
             m.shape[0] - max(index) - 1),
                len(index),
                len(index),
                facecolor='none',
                edgecolor='r',
                lw=3)
        )

plt.title('Cluster matrix')

pass
"""
