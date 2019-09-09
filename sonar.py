import pandas as pd
import numpy as np
import operator
import matplotlib.pyplot as plt

import warnings

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import roc_curve, auc


warnings.filterwarnings('ignore')

sonar_table = pd.read_csv('sonar.all-data')


# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html
# Label Encoder.
le = LabelEncoder()
le.fit(sonar_table.iloc[:, -1])
sonar_table.iloc[:, -1] = le.transform(sonar_table.iloc[:, -1])

# Code taken from my midterm solutions along with the midterm solutions posted by prof
train, test = train_test_split(sonar_table, test_size=0.2, random_state=42)

pca = PCA(n_components = .9)

scaler = StandardScaler()

train_labels = train.iloc[:, -1]
train_data = train.drop(train.columns[-1], axis=1)
test_labels = test.iloc[:, -1]
test_data = test.drop(test.columns[-1], axis=1)

pca.fit(train_data)
pca_train = pca.transform(train_data)
pca_test = pca.transform(test_data)

svc = svm.SVC(kernel='rbf', gamma='auto')
svc = svc.fit(pca_train, train_labels)

y_pred = svc.predict(pca_test)

fpr, tpr, threshold = roc_curve(test_labels, y_pred)

plt.plot(fpr, tpr, color='b')
plt.plot([0,1],[0,1], color='r')

print("SVC AUC: ", auc(fpr,tpr))

# https://pythonprogramming.net/linear-svc-example-scikit-learn-svm-python/
clf = svm.SVC(kernel='linear', C = 1.0)
clf = clf.fit(pca_train, train_labels)

y_pred_clf = clf.predict(pca_test)

fpr, tpr, threshold = roc_curve(test_labels, y_pred_clf)
plt.plot(fpr, tpr, color='g')

print("SVM AUC: ", auc(fpr,tpr))

plt.show()