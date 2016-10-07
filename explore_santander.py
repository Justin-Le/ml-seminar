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

# Targets: customer satisfaction
# 0 = happy, 1 = unhappy
# Over 96% are happy 
df = pd.DataFrame(train.TARGET.value_counts())
df['Percentage'] = 100*df['TARGET']/train.shape[0]
df

########################################
# Explore var3
########################################

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

########################################
# Explore num_var4
########################################

# num_var4: the number of products that a customer has purchased
train.num_var4.hist(bins=100)
plt.xlabel('Number of bank products')
plt.ylabel('Number of customers')
plt.title('Most customers have 1 product with the bank')
plt.show()

# Customer satisfaction versus number of products purchased
sns.FacetGrid(train, hue="TARGET", size=6).map(plt.hist, "num_var4").add_legend()
plt.title('Unhappy customers purchased fewer products')
plt.show()

########################################
# Explore num_var38
########################################

print train.var38.describe()

# var38 for unhappy customers
print train.loc[train['TARGET']==1, 'var38'].describe()

# Distribution of var38 is not Gaussian
train.var38.hist(bins=1000);
plt.show()

# Show distribution in log-scale to clarify
train.var38.map(np.log).hist(bins=1000);
plt.show()

# Identify the anomaly: 
# a spike between values 11 and 12 of the distribution
train.var38.map(np.log).mode()

# Most common values for var38
print train.var38.value_counts()

# Most common value is close to the mean of the other values
print train.var38[train['var38'] != 117310.979016494].mean()

# Excluding the most common value causes the
# distribution to become normal (in log-scale)
print train.loc[~np.isclose(train.var38, 117310.979016), 'var38'].value_counts()
train.loc[~np.isclose(train.var38, 117310.979016), 'var38'].map(np.log).hist(bins=100);
plt.show()

# Split var38
# var38mc == 1 when var38 has the most common value and 0 otherwise
# logvar38 = {log(var38) if var38mc == 0; 0 otherwise}
train['var38mc'] = np.isclose(train.var38, 117310.979016)
train['logvar38'] = train.loc[~train['var38mc'], 'var38'].map(np.log)
train.loc[train['var38mc'], 'logvar38'] = 0

# Check for NaN
print('Number of nan in var38mc', train['var38mc'].isnull().sum())
print('Number of nan in logvar38',train['logvar38'].isnull().sum())

########################################
# Explore num_var15
########################################

# var15 = customer age
# XGBoost gave high importance to var15
print train['var15'].describe()
train['var15'].hist(bins=100);

sns.FacetGrid(train, hue="TARGET", size=6).map(sns.kdeplot, "var15").add_legend()
plt.title('Unhappy customers are slightly older');

# var15 versus var38
sns.FacetGrid(train, hue="TARGET", size=10).map(plt.scatter, "var38", "var15").add_legend();
sns.FacetGrid(train, hue="TARGET", size=10).map(plt.scatter, "logvar38", "var15").add_legend()
plt.ylim([0,120]);

# Exclude most common value of var38 
sns.FacetGrid(train[~train.var38mc], hue="TARGET", size=10).map(plt.scatter, "logvar38", "var15").add_legend()
plt.ylim([0,120]);

# Distribution of the age when var38 has its most common value
sns.FacetGrid(train[train.var38mc], hue="TARGET", size=6).map(sns.kdeplot, "var15").add_legend();

sns.FacetGrid(train, hue="TARGET", size=6).map(sns.kdeplot, "n0").add_legend()
plt.title('Unhappy customers have a lot of features that are zero');
plt.show()

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
print('Chi2 selected {} features {}.\n'.format(chi2_selected.sum(),
   chi2_selected_features))
f_classif_selected = selectF_classif.get_support()
f_classif_selected_features = [ f for i,f in enumerate(X.columns) if f_classif_selected[i]]
print('F_classif selected {} features {}.\n'.format(f_classif_selected.sum(),
   f_classif_selected_features))
selected = chi2_selected & f_classif_selected
print('Chi2 & F_classif selected {} features.\n'.format(selected.sum()))
features = [ f for f,s in zip(X.columns, selected) if s]
print 'Randomly selected features:\n' 
print features
print '\n' 

# Make a dataframe with the selected features and their targets
X_sel = train[features + ['TARGET']]

########################################
# Explore var36
########################################

# var36
X_sel['var36'].value_counts()

# var36 concetrates around 99 and {0,1,2,3}
sns.FacetGrid(train, hue="TARGET", size=6)    .map(sns.kdeplot, "var36").add_legend()
plt.title('If var36 is 0,1,2 or 3 => less unhappy customers');

# Density of unhappy custormers is lower when var36 is not 99
# var36 versus logvar38
sns.FacetGrid(train[~train.var38mc], hue="TARGET", size=10).map(plt.scatter, "var36", "logvar38").add_legend();

# Plot the above separately
sns.FacetGrid(train[(~train.var38mc) & (train.var36 < 4)], hue="TARGET", size=10).map(plt.scatter, "var36", "logvar38").add_legend()
plt.title('If var36==0, only happy customers');

# var36 == 99
sns.FacetGrid(train[(~train.var38mc) & (train.var36 ==99)], hue="TARGET", size=6)    .map(sns.kdeplot, "logvar38").add_legend();

# num_var5
train.num_var5.value_counts()
train[train.TARGET==1].num_var5.value_counts()
train[train.TARGET==0].num_var5.value_counts()

sns.FacetGrid(train, hue="TARGET", size=6).map(plt.hist, "num_var5").add_legend();
sns.FacetGrid(train, hue="TARGET", size=6).map(sns.kdeplot, "num_var5").add_legend();

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
