import pandas as pd
from scipy.stats import friedmanchisquare


def run():
    perf_df = pd.read_csv("algo_performance.csv")
    stats, p_value = friedmanchisquare(*[col for name, col in perf_df.iteritems()])
    print(f"Friedman test Statistics: {stats} and P-value: {p_value}")
    for a in [0.01, 0.025, 0.05]:
        print(f"For an a-value of {a}")
        if p_value < a:
            print("There is a significant statistical difference between the algorithm results")
        else:
            print("No significant statistical difference found between the algorithm results")
        print("\n")


if __name__ == "__main__":
    run()
