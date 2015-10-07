from sklearn import ensemble
from sklearn.datasets import make_classification
from sklearn.cross_validation import train_test_split
from sklearn.metrics import precision_score
import numpy as np

n_samples = 100
n_outputs = 3

Xs = []
ys = []

for i in xrange(n_outputs):
    X, y = make_classification(n_samples=n_samples, weights=[.2])
    Xs.append(X)
    ys.append(y)

X = np.concatenate(Xs, axis=1)
y = np.concatenate(ys).reshape((n_samples,n_outputs), order='F')

X_train, X_test, y_train, y_test = train_test_split(X,y)

estimator = ensemble.RandomForestClassifier(balanced=True)
estimator.fit(X_train,y_train)

y_pred = estimator.predict(X_test)

print precision_score(y_test, y_pred, average='weighted')
