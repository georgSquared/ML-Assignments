import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets, model_selection, metrics
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

RANDOM_SEED = 0
NUM_TREES = 100


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


def get_data():
    bc_data = datasets.load_breast_cancer()
    X = bc_data.data
    y = bc_data.target

    return model_selection.train_test_split(X, y, random_state=RANDOM_SEED)


def get_predictions(model, x_train, y_train, x_test):
    model.fit(x_train, y_train)
    return model.predict(x_test)


def get_bagging():
    tree_classifier = DecisionTreeClassifier()
    classifier = BaggingClassifier(base_estimator=tree_classifier, n_estimators=NUM_TREES, random_state=RANDOM_SEED,
                                   n_jobs=-1)
    return classifier


def get_random_forest():
    classifier = RandomForestClassifier(criterion='gini', n_estimators=NUM_TREES, n_jobs=-1)
    return classifier


def create_graph(labels, values1, values2):
    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width / 2, values1, width, label='Bagging')
    rects2 = ax.bar(x + width / 2, values2, width, label='Random Forest')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Scores')
    ax.set_title('Bagging Classifier and Random Forest metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()

    plt.savefig('bc_bagging_random_forest_grouped.png')
    plt.show()


def run():
    x_train, x_test, y_train, y_test = get_data()

    bagging_model = get_bagging()
    forest_model = get_random_forest()

    y_bag = get_predictions(bagging_model, x_train, y_train, x_test)
    y_forest = get_predictions(forest_model, x_train, y_train, x_test)

    labels = ["accuracy", "precision", "recall", "f1"]
    bag_values = [round(val, 2) for val in evaluate_model(y_test, y_bag)]
    forest_values = [round(val, 2) for val in evaluate_model(y_test, y_forest)]

    create_graph(labels, bag_values, forest_values)


if __name__ == "__main__":
    run()
