"""
Read the heavy benchmark data
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

from util.util import read_json

destinations = [5, 10, 50, 100, 300, 500]


def plot_boxplot(exec_times):
    plt.figure(dpi=300)
    plt.title("Heavy benchmark")
    plt.boxplot(exec_times)
    plt.xticks(np.arange(1, len(exec_times) + 1), destinations)
    plt.xlabel("Number of destinations")
    plt.ylabel("Execution time (seconds)")
    plt.grid()
    plt.show()


def plot_means(exec_times):
    means = np.mean(exec_times, axis=1)

    # Do linregress
    slope, intercept, rvalue, *_ = linregress(destinations, means)

    plt.figure(dpi=300)
    plt.title("Heavy benchmark")
    plt.plot(
        (0, 550), (intercept, intercept + 550 * slope), "r", label="r² >= 0.999"
    )  # Regression slope
    plt.plot(destinations, means, "+k")
    plt.xlabel("Number of destinations")
    plt.ylabel("Execution time (seconds)")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    data = read_json("heavy.json")

    plot_means(data)
    plot_boxplot(data)
