# IMPORT NECESSARY LIBRARIES HERE
from matplotlib import pyplot
from sklearn import datasets, metrics, ensemble, model_selection


def evaluate_model(actual, predicted, print_for_params=False):
    average = "macro"
    recall_score = metrics.recall_score(actual, predicted, average=average)
    precision_score = metrics.precision_score(actual, predicted, average=average)
    f1_score = metrics.f1_score(actual, predicted, average=average)

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


def get_model(criterion, n_estimators, max_depth, max_features):
    forest_model = ensemble.RandomForestClassifier(criterion=criterion,
                                                   n_estimators=n_estimators,
                                                   max_depth=max_depth,
                                                   max_features=max_features,
                                                   random_state=0)
    return forest_model


def get_predictions(model, x_train, y_train, x_test):
    model.fit(x_train, y_train)
    return model.predict(x_test)


def create_graph(estimators, test_data, train_data, title, ylabel):
    """
    Create graphs as number of estimators changes
    :return:
    """
    pyplot.plot(estimators, test_data, label="test-data", color='black')
    pyplot.plot(estimators, train_data, label="train-data", color='red')
    # Create plot
    pyplot.title(title)
    pyplot.xlabel("Number Of Estimator")
    pyplot.ylabel(ylabel)
    pyplot.legend(loc="best")

    pyplot.savefig(title + '.png')

    pyplot.show()


if __name__ == "__main__":
    breastCancer = datasets.load_breast_cancer()
    numberOfFeatures = 10
    X = breastCancer.data[:, :numberOfFeatures]
    y = breastCancer.target

    x_train, x_test, y_train, y_test = model_selection.train_test_split(X, y, random_state=0)

    criteria = ["gini", "entropy"]
    max_depth = 3

    for selection_criterion in criteria:
        accuracy_list = []
        precision_list = []
        recall_list = []
        f1_list = []

        accuracy_list_train = []
        precision_list_train = []
        recall_list_train = []
        f1_list_train = []

        for number_of_trees in range(1, 200, 10):
            forest = get_model(selection_criterion, number_of_trees, max_depth, numberOfFeatures)

            y_predicted = get_predictions(forest, x_train, y_train, x_test)
            accuracy, precision, recall, f1 = evaluate_model(y_test, y_predicted)

            accuracy_list.append(accuracy)
            precision_list.append(precision)
            recall_list.append(recall)
            f1_list.append(accuracy)

            print("Criterion, Number of Estimators, Accuracy, Precision, Recall, F1")
            print(f"{selection_criterion}, {number_of_trees}, {accuracy:.4f}, {precision:.4f}, {recall:.4f}, {f1:.4f}")

            y_predicted_train = get_predictions(forest, x_train, y_train, x_train)
            accuracy_train, precision_train, recall_train, f1_train = evaluate_model(y_train, y_predicted_train)

            accuracy_list_train.append(accuracy_train)
            precision_list_train.append(precision_train)
            recall_list_train.append(recall_train)
            f1_list_train.append(f1_train)

        create_graph(
            range(1, 200, 10),
            accuracy_list,
            accuracy_list_train,
            f"Accuracy - {selection_criterion}",
            "Accuracy Score"
        )

        create_graph(
            range(1, 200, 10),
            precision_list,
            precision_list_train,
            f"Precision - {selection_criterion}",
            "Precision Score"
        )

        create_graph(
            range(1, 200, 10),
            recall_list,
            recall_list_train,
            f"Recall - {selection_criterion}",
            "Recall Score"
        )

        create_graph(
            range(1, 200, 10),
            f1_list,
            f1_list_train,
            f"F1 - {selection_criterion}",
            "F1 Score"
        )

        print("\n")
