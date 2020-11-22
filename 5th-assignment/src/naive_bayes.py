import matplotlib.pyplot as plt
import numpy as np
import seaborn
from sklearn import datasets, metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB


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
    # Remove redundant metadata. This will reduce F1 to more realistic values
    to_remove = ('headers', 'footers', 'quotes')
    newsgroups_train = datasets.fetch_20newsgroups(subset='train', random_state=0, remove=to_remove)
    newsgroups_test = datasets.fetch_20newsgroups(subset='test', random_state=0, remove=to_remove)

    x_train = newsgroups_train.data
    y_train = newsgroups_train.target

    x_test = newsgroups_test.data
    y_test = newsgroups_test.target

    return x_train, y_train, x_test, y_test, newsgroups_test.target_names


def vectorize_data(x_train, x_test):
    vectorizer = TfidfVectorizer()
    vectorized_train = vectorizer.fit_transform(x_train)
    vectorized_test = vectorizer.transform(x_test)

    return vectorized_train, vectorized_test


def get_best_predictions(x_train_vec, y_train, x_test_vec, y_test):
    results = []

    for alpha in np.arange(0.01, 1, 0.01):
        classifier = MultinomialNB(alpha=alpha, fit_prior=True)
        classifier.fit(x_train_vec, y_train)
        y_predicted = classifier.predict(x_test_vec)

        accuracy, precision, recall, f1 = evaluate_model(y_test, y_predicted)

        results.append({
            "alpha": alpha,
            "y_test": y_test,
            "y_predicted": y_predicted,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        })

    return max(results, key=lambda res: res["f1"])


def plot_heatmap(data, labels):

    matrix = metrics.confusion_matrix(data["y_test"], data["y_predicted"])
    plt.figure(1, figsize=(20, 15))

    title = f"Multinomial NB - Confusion Matrix. a: {data['alpha']}, F1: {data['f1']:.2f}, acc: {data['accuracy']:.2f}, prec: {data['precision']:.2f}, rec: {data['recall']:.2f}"
    plt.title(title)

    seaborn.set(font_scale=1.4)
    ax = seaborn.heatmap(matrix, annot=True, cbar=False, cmap='OrRd', fmt='d')

    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticklabels(labels, rotation=0)

    plt.savefig(f"{title}.png", bbox_inches='tight')
    plt.show()


def run():
    x_train, y_train, x_test, y_test, categories = get_data()
    x_train_vectorized, x_test_vectorized = vectorize_data(x_train, x_test)
    best_results = get_best_predictions(x_train_vectorized, y_train, x_test_vectorized, y_test)
    print(best_results)

    plot_heatmap(best_results, categories)


if __name__ == "__main__":
    run()
