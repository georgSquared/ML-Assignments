from sklearn import datasets
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import LeaveOneOut


def pprint(d, indent=0):
    for key, value in d.items():
        print('\t' * indent + str(key))
        if isinstance(value, dict):
            pprint(value, indent + 1)
        else:
            print('\t' * (indent + 1) + str(value))


def get_data():
    """
    We choose the breast cancer dataset, that is a binary classification dataset
    :return:
    """
    breast_cancer = datasets.load_breast_cancer()
    return breast_cancer.data, breast_cancer.target


def get_metric_key(actual, predicted):
    if predicted == 0:
        if actual == 0:
            return "TP"
        else:
            return "FP"
    else:
        if actual == 0:
            return "FN"
        else:
            return "TN"


def get_predictions(x_train, y_train, x_test):
    classifier = LogisticRegression(max_iter=10000, n_jobs=-1)
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)

    return y_pred


def validate(x, y):
    evaluation_metrics = {
        "TP": 0,
        "FP": 0,
        "TN": 0,
        "FN": 0,
        "count": 0,
        "accuracy": 0
    }

    cross_validator = LeaveOneOut()
    for train_idx, test_idx in cross_validator.split(x):
        x_train, y_train = x[train_idx], y[train_idx]
        x_test, y_test = x[test_idx], y[test_idx]

        y_pred = get_predictions(x_train, y_train, x_test)

        metric_key = get_metric_key(y_test, y_pred)

        evaluation_metrics[metric_key] += 1
        evaluation_metrics["count"] += 1

    evaluation_metrics["accuracy"] = (evaluation_metrics["TP"] + evaluation_metrics["TN"]) / evaluation_metrics["count"]

    return evaluation_metrics


def run():
    x, y = get_data()
    metrics = validate(x, y)
    pprint(metrics)


if __name__ == "__main__":
    run()
