from sklearn import datasets, metrics, tree, model_selection
import graphviz


def evaluate_model(actual, predicted, print_for_params=False):

    recall_score = metrics.recall_score(actual, predicted, average="micro")
    precision_score = metrics.precision_score(actual, predicted, average="micro")
    f1_score = metrics.f1_score(actual, predicted, average="micro")

    if print_for_params:
        average_params = ['binary', 'micro', 'macro', 'weighted']
        for param in average_params:
            recall_score = metrics.recall_score(actual, predicted, average=param)
            print(f"Precision score: {precision_score:.2f} with average parameter: {param}")

            precision_score = metrics.precision_score(actual, predicted, average=param)
            print(f"Recall score: {recall_score:.2f} with average parameter: {param}")

            f1_score = metrics.f1_score(actual, predicted, average=param)
            print(f"F1 score: {f1_score:.2f} with average parameter: {param} \n")

    accuracy_score = metrics.accuracy_score(actual, predicted)
    if print_for_params:
        print(f"Accuracy score: {accuracy_score:.2f}")

    return accuracy_score, precision_score, recall_score, f1_score


def get_model(criterion, max_depth):
    tree_model = tree.DecisionTreeClassifier(criterion=criterion, splitter="best", max_depth=max_depth)
    return tree_model


def get_predictions(model, x_train, y_train, x_test):
    model.fit(x_train, y_train)
    return model.predict(x_test)


def create_graph(model, dataset):
    dot_data = tree.export_graphviz(model,
                                    out_file=None,
                                    feature_names=dataset.feature_names[:numberOfFeatures],
                                    class_names=dataset.target_names,
                                    filled=True,
                                    rounded=True,
                                    special_characters=True)

    graph = graphviz.Source(dot_data)
    graph.render("breastCancerTreePlot")


if __name__ == "__main__":
    breastCancer = datasets.load_breast_cancer()
    numberOfFeatures = 10
    X = breastCancer.data[:, :numberOfFeatures]
    y = breastCancer.target

    x_train, x_test, y_train, y_test = model_selection.train_test_split(X, y)

    criteria = ["gini", "entropy"]
    max_depth = 3

    for selection_criterion in criteria:
        for tree_depth in range(3, 10):
            tree_model = get_model(selection_criterion, tree_depth)
            y_predicted = get_predictions(tree_model, x_train, y_train, x_test)
            accuracy, precision, recall, f1 = evaluate_model(y_test, y_predicted)

            print("Criterion, Depth, Accuracy, Precision, Recall, F1")
            print(f"{selection_criterion}, {tree_depth}, {accuracy:.2f}, {precision:.2f}, {recall:.2f}, {f1:.2f}")

            print("=======================")
            print("Train Data")
            print("=======================")
            y_predicted_train = get_predictions(tree_model, x_train, y_train, x_train)
            evaluate_model(y_train, y_predicted_train, print_for_params=True)

        print("\n")









