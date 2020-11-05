import os
# os.environ["PATH"] += os.pathsep + 'C:\Program Files (x86)\Graphviz2.38\bin'

from sklearn import datasets, model_selection, metrics, tree
import graphviz
import numpy as np

breastCancer = datasets.load_breast_cancer()


numberOfFeatures = 10
X = breastCancer.data[:, :numberOfFeatures]
y = breastCancer.target
features = breastCancer.feature_names
print np.shape(X), np.shape(y), len(features), features

model = tree.DecisionTreeClassifier(criterion="gini", splitter="best", max_depth=3)
x_train, x_test, y_train, y_test = model_selection.train_test_split(X, y)
model.fit(x_train, y_train)

y_predicted = model.predict(x_test)

print("Recall: %2f" % metrics.recall_score(y_test, y_predicted, average="macro"))
print("Precision: %2f" % metrics.precision_score(y_test, y_predicted, average="macro"))
print("F1: %2f" % metrics.f1_score(y_test, y_predicted, average="macro"))

dot_data = tree.export_graphviz(model, out_file=None, feature_names=breastCancer.feature_names[:numberOfFeatures],
                                class_names=breastCancer.target_names, filled=True, rounded=True, special_characters=True)
graph = graphviz.Source(dot_data)
graph.render("breastCancerDT")