import Orange
from Orange import classification, evaluation


def load_data(file_name):
    """
    Load a file of the accepted formats as an Orange data Table
    """
    return Orange.data.Table(file_name)


def get_learner(ordered=True, evaluator="entropy"):
    """
    Get a single unparametrized learner
    """
    if ordered:
        learner = Orange.classification.rules.CN2Learner()
    else:
        learner = Orange.classification.rules.CN2UnorderedLearner()

    if ordered and evaluator == "laplace":
        learner.rule_finder.quality_evaluator = Orange.classification.rules.LaplaceAccuracyEvaluator()

    return learner


def get_learners(beam_width, min_covered_examples, max_rule_length):
    """
    Get a list of the 3 required learners, parametrized according to input
    """
    ordered_entropy = Orange.classification.rules.CN2Learner()

    unordered = Orange.classification.rules.CN2UnorderedLearner()

    laplace = Orange.classification.rules.CN2Learner()
    laplace.rule_finder.quality_evaluator = Orange.classification.rules.LaplaceAccuracyEvaluator()

    learners = [ordered_entropy, unordered, laplace]
    labels = ["ordered", "unordered", "laplace"]

    for learner in learners:
        """
        Parametrize the learners
        """
        learner.rule_finder.search_algorithm.beam_width = beam_width
        learner.rule_finder.general_validator.min_covered_examples = min_covered_examples
        learner.rule_finder.general_validator.max_rule_length = max_rule_length

    return learners, labels


def run_tests():
    wine_data = load_data("wine.csv")
    validator = Orange.evaluation.CrossValidation()

    max_values = {
        "ordered": {"accuracy": 0, "f1": 0},
        "unordered": {"accuracy": 0, "f1": 0},
        "laplace": {"accuracy": 0, "f1": 0},
    }

    best = {
        "ordered": {"scores": [], "params": []},
        "unordered": {"scores": [], "params": []},
        "laplace": {"scores": [], "params": []},
    }

    for beam_width in range(2, 11):
        for min_covered_examples in range(7, 21):
            for max_rule_length in range(2, 6):
                learners, labels = get_learners(beam_width, min_covered_examples, max_rule_length)

                results = validator(wine_data, learners)

                accuracy = Orange.evaluation.CA(results)
                precision = Orange.evaluation.Precision(results, average='macro')
                recall = Orange.evaluation.Recall(results, average='macro')
                f1 = Orange.evaluation.F1(results, average='macro')

                print(f"Beam:{beam_width}, Min Covered:{min_covered_examples}, Max Rules:{max_rule_length}")
                print(f"Accuracy:{accuracy}, Precision:{precision}, Recall:{recall}, F1:{f1}")

                for i, label in enumerate(labels):
                    if accuracy[i:i+1] > max_values[label]["accuracy"] or \
                            (accuracy[i:i+1] == max_values[label]["accuracy"] and f1[i:i+1] > max_values[label]["f1"]):
                        best[label] = {
                            "scores": [accuracy[i:i+1], precision[i:i+1], recall[i:i+1], f1[i:i+1]],
                            "params": [beam_width, min_covered_examples, max_rule_length]
                        }

                        max_values[label] = {
                            "accuracy": accuracy[i:i+1],
                            "f1": f1[i:i+1]
                        }
    return best


def print_rules(params):
    learner = None

    for rule in learner.rule_list:
        print(rule)


if __name__ == "__main__":
    best_params = run_tests()
    print_rules(best_params)
