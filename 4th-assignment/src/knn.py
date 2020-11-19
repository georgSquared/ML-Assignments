import json

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import KNNImputer
from sklearn import model_selection, metrics
from sklearn.preprocessing import MinMaxScaler

random.seed = 42
np.random.seed(666)


def pprint(d, indent=0):
    for key, value in d.items():
        print('\t' * indent + str(key))
        if isinstance(value, dict):
            pprint(value, indent + 1)
        else:
            print('\t' * (indent + 1) + str(value))


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


def preprocess_titanic_data(titanic_df, drop_age=False):
    """
    Select features, cleanup, normalize and fill missing data
    :return The data split into train and test sets
    """

    # Drop non-valuable features. Cabin is interesting, but has too many missing data
    to_drop = ["PassengerId", "Name", "Ticket", "Cabin"]
    if drop_age:
        to_drop.append("Age")

    titanic_df.drop(
        to_drop,
        inplace=True,
        axis=1
    )

    # Merge SibSp(Siblings/Spouse) and Parch (Parents/Children) as Family with a 1/0 value
    # They essentially model the same thing, whether a passenger had someone to care for or not
    titanic_df["Family"] = np.where((titanic_df["SibSp"] + titanic_df["Parch"]) > 0, 1, 0)
    titanic_df.drop(["SibSp", "Parch"], inplace=True, axis=1)

    # Substitute Sex and Embarked with numerical data
    dummy_sex_embarked = pd.get_dummies(titanic_df, columns=['Sex', 'Embarked'], prefix=['Sex', 'Embarked'],
                                        drop_first=True)
    titanic_df.drop(["Sex", "Embarked"], axis=1, inplace=True)
    titanic_df = pd.concat([titanic_df, dummy_sex_embarked], axis=1)

    # Properly select target features
    X = titanic_df.drop(['Survived'], axis=1)
    y = titanic_df['Survived']

    return model_selection.train_test_split(X, y, test_size=0.2, random_state=0)


def normalize_data(train_data, test_data):
    scaler = MinMaxScaler()
    scaler.fit(train_data)
    train = scaler.transform(train_data)
    test = scaler.transform(test_data)

    return train, test


def fill_data(train_data, test_data):
    imputer = KNNImputer(n_neighbors=3, weights='distance')
    imputer.fit(train_data)
    train = imputer.transform(train_data)
    test = imputer.transform(test_data)

    return train, test


def run_tests(x_train, x_test, y_train, y_test):
    neighbors = range(1, 200)
    metric = "minkowski"
    weights = ["uniform", "distance"]
    p_params = [1, 2, 3]

    results = {}
    for weight in weights:
        results[weight] = {}
        for p in p_params:
            # keep max_f1 here
            results[weight][p] = {
                "max_f1": 0,
                "f1_list": []
            }
            for n in neighbors:
                classifier = KNeighborsClassifier(n_neighbors=n, metric=metric, weights=weight, p=p, n_jobs=-1)

                classifier.fit(x_train, y_train)
                y_predicted = classifier.predict(x_test)

                accuracy, precision, recall, f1 = evaluate_model(y_test, y_predicted)

                results[weight][p]["f1_list"].append(f1)

                if f1 > results[weight][p]["max_f1"]:
                    results[weight][p] = {
                        **results[weight][p],
                        **{
                            "max_f1": f1,
                            "accuracy": accuracy,
                            "precision": precision,
                            "recall": recall,
                            "neighbors": n,
                        }
                    }

    return results


def get_data(imputed=True):
    if imputed:
        titanic = pd.read_csv('titanic.csv')
        x_train, x_test, y_train, y_test = preprocess_titanic_data(titanic, drop_age=False)
        x_train, x_test = normalize_data(x_train, x_test)
        x_train, x_test = fill_data(x_train, x_test)
    else:
        titanic = pd.read_csv('titanic.csv')
        x_train, x_test, y_train, y_test = preprocess_titanic_data(titanic, drop_age=True)
        x_train, x_test = normalize_data(x_train, x_test)

    return run_tests(x_train, x_test, y_train, y_test)


def plot_results(imputed, non_imputed):
    for weight in imputed:
        for p in imputed[weight]:
            neighbors = [*range(1, 200)]
            title = f'k-Nearest Neighbors (Weights = {weight}, Metric = minkowski, p ={p})'
            plt.title(title)
            plt.plot(
                neighbors,
                imputed[weight][p]["f1_list"],
                label='with impute', color="red"
            )

            plt.plot(
                neighbors,
                non_imputed[weight][p]["f1_list"],
                label='without impute', color="black"
            )

            plt.legend()
            plt.xlabel('Number of neighbors')
            plt.ylabel('F1')

            plt.savefig(title + '.png')
            plt.show()


if __name__ == "__main__":
    print("Imputed Run")
    imputed_results = get_data(imputed=True)
    pprint(imputed_results)

    print("Non-imputed Run")
    non_imputed_results = get_data(imputed=False)
    pprint(non_imputed_results)

    plot_results(imputed_results, non_imputed_results)
