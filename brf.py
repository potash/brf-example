from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from StringIO import StringIO
import gzip
from urllib import urlopen
import time
import os
import sys

# calculates the precision at the top k examples of the positive class
def precision_curve(y_true, y_score,k=None):
    if k is None:
        k = len(y_true)
    ranks = y_score.argsort()[::-1]
    top_k = ranks[0:k]
    return np.cumsum(y_true[top_k])*1.0/np.arange(1,k+1)

# read a gzipped csv from a url into a pandas dataframe
def csv_from_gzip_url(url):
    f = StringIO(urlopen(url).read())
    df = pd.read_csv(f, compression='gzip')
    return df

# binarize all columns with object dtype
def binarize(df):
    categorical_columns = df.dtypes[df.dtypes == object].index
    for column in categorical_columns:
        categories = df[column].unique()
        for category in categories:
            df[category] = (df[column] == category)
        df.drop(column, axis=1, inplace=True)

# code the specified column as an integer
def code(df, column):
    categories = df[column].unique()
    for i, category in enumerate(categories):
        df.loc[df[column]==category, [column]] = i
    df[column] = df[column].astype(int)


# Load the data 
dirname = os.path.dirname(sys.argv[0])
#http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data_10_percent.gz
kddtrain = pd.read_csv(os.path.join(dirname, 'data/kddcup.data_10_percent.gz'), compression='gzip')  
#'http://kdd.ics.uci.edu/databases/kddcup99/corrected.gz'
kddtest = pd.read_csv(os.path.join(dirname, 'data/corrected.gz'), compression='gzip') 

# rename columns because the csvs don't have headers
kddtrain.columns = range(42)
kddtest.columns = range(42)

# concat the data so that binarization and coding are uniform across train and test
kdd = pd.concat((kddtrain,kddtest))
code(kdd, 41)
binarize(kdd)

# split up into X,y; training, testing
X = kdd.drop(41, axis=1).values
y = (kdd[41].values > 5)
X_train = X[0:len(kddtrain),:]
y_train = y[0:len(kddtrain)]
X_test = X[-len(kddtest):,:]
y_test = y[-len(kddtest):]

print 'baseline: {}'.format(y_train.sum()*1.0 / len(y_train)) # the minority class makes up 1.7% of the training set
print ''

# Define the model parameters to run
common_params={'n_estimators':100, 'criterion':'entropy', 'n_jobs':1}
params = [
    {}, 
    {'class_weight':'balanced'}, 
    {'class_weight':'balanced_subsample'},
    {'class_weight':'balanced_bootstrap'}
]

p = y_train.sum()
n = len(y_train) - p
# a few custom class weights for comparison
# class_weight = [p,n] would be the same as 'auto'. relative to 'auto', the models below range from underweighting to overweighting the minority
for w in [.5,.75,1.25,1.5]:
    params.append({'class_weight': {False:p, True:int(n*w)}})

# Run the models
K = 10000 # the range of precision at k to show
n = 10 # the number of runs per model
plt.figure()

for p in params:
    print 'forest parameters: {}'.format(p)
    clf = RandomForestClassifier(**dict(common_params,**p))

    start = time.clock()
    auc = np.empty(n)
    precision = np.empty((n,K))
    for i in xrange(n):
        clf.fit(X_train,y_train)

        y_score = clf.predict_proba(X_test)[:,1]

        precision[i] = precision_curve(y_test, y_score, K)
        auc[i] = roc_auc_score(y_test, y_score)

    print 'time elapsed: {}'.format( (time.clock() - start) / 10.0)
    print 'precision at {}: {} +/- {}'.format(K, np.mean(precision[:,K-1]), np.std(precision[:,K-1]))
    print 'auc: {} +/- {}'.format(np.mean(auc), np.std(auc))
    print ''

    plt.plot(range(1,K+1), np.mean(precision, axis=0), label=str(p))

plt.xlabel('k')
plt.ylabel('Precision at k')
plt.title('Imbalanced data random forest example')
plt.legend(loc="lower left")
plt.savefig('fig.png')
