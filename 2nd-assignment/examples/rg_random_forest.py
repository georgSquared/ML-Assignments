# =============================================================================
# !!! NOTE !!!
# The below import is for using Graphviz!!! Make sure you install it in your
# computer somewhere, # after downloading it from here:
# https://graphviz.gitlab.io/_pages/Download/Download_windows.html
# After installation, change the 'C:/Program Files (x86)/Graphviz2.38/bin/'
# from line 11 to the directory that you installed GraphViz (might be the same though).
# =============================================================================
import os
from sklearn import datasets, model_selection, metrics, ensemble, tree
import graphviz

# Import data
breastCancer = datasets.load_breast_cancer()

numberOfFeatures = 10
X = breastCancer.data[:, :numberOfFeatures]
y = breastCancer.target

# RandomForestClassifier is the core of this script. You can customize its functionality
# in various ways, but for now play with the 'criterion' and 'maxDepth' parameters.
# =============================================================================
# 'criterion': Can be either 'gini' (for the Gini impurity) and 'entropy' for the information gain.
# 'n_estimators': The number of trees in the forest. The larger the better, but it will take longer to compute. Also,
#                 there is a critical number after which there is no significant improvement in the results.
# 'max_depth': The maximum depth of the tree. A large depth can lead to overfitting, so start with a maxDepth of
#              e.g. 3, and increase it slowly by evaluating the results each time.
# 'max_features': The size of the random subsets of features to consider when splitting a node
#                 (usually = numberOfFeatures for regression problems).
# =============================================================================
model = ensemble.RandomForestClassifier(criterion="gini", n_estimators=100, max_depth=3, max_features=numberOfFeatures, random_state=0)

# The function below will split the dataset that we have into two subsets. We will use
# the first subset for the training (fitting) phase, and the second for the evaluation phase.
# By default, the train set is 75% of the whole dataset, while the test set makes up for the rest 25%.
x_train, x_test, y_train, y_test = model_selection.train_test_split(X, y, random_state=0)

# Let's train our model.
model.fit(x_train, y_train)

# Ok, now let's predict the output for the second subset
y_predicted = model.predict(x_test)


print("Recall: %2f" % metrics.recall_score(y_test, y_predicted, average="macro"))
print("Precision: %2f" % metrics.precision_score(y_test, y_predicted, average="macro"))
print("F1: %2f" % metrics.f1_score(y_test, y_predicted, average="macro"))


# If you want to visualize a single tree from the forest, just get one from the 'estimators_' property of the model
# and use it in the 'export_graphviz' function below:
estimator = model.estimators_[5]


# Ok, now let's use the 'export_graphviz' function from the 'sklearn.tree' package to visualize the trained
# tree. The are is a variety of options to configure, which can lead to a quite visually pleasant result.
# Also, this will export the graph into a PDF file located within the same folder as this script, from where you can see it.
# If you want to view it from the Python IDE, type 'graph' (without quotes) on the python console after the script has been executed.
dot_data = tree.export_graphviz(estimator, out_file=None, feature_names=breastCancer.feature_names[:numberOfFeatures],
                                class_names=breastCancer.target_names, filled=True, rounded=True, special_characters=True)
graph = graphviz.Source(dot_data)
graph.render("breastCancerRF")
