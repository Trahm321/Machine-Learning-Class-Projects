import pandas as pd
import numpy as np
import operator
import matplotlib.pyplot as plt

import warnings
import os

from sklearn.externals.six import StringIO
from IPython.display import Image
import graphviz
from sklearn import tree

warnings.filterwarnings('ignore')

wine_data = pd.read_csv('wine.data')

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

# https://scikit-learn.org/stable/modules/tree.html

y_axis = np.c_[wine_data.iloc[:, -1]].astype(np.int)
wine_data = wine_data.drop(wine_data.columns[-1], axis=1)
x_axis = np.c_[wine_data].astype(np.float64)

clf = tree.DecisionTreeClassifier(max_depth=4)
clf.fit(x_axis, y_axis)

dot_data = tree.export_graphviz(clf, out_file=None)
graph = graphviz.Source(dot_data)
graph.view("Wine")