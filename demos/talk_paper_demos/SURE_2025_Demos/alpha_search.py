import qmcpy as qp
import numpy as np
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

def alpha_search(dimension, n_alpha, n, sample_wt, coord_wt, disable = False):
    best_alpha = np.empty(dimension)

    # uses product form of kernel to reduce complexity
    previous_k_tilde = np.ones(n, dtype = np.float64)

    # used to calculate choices for alpha
    choices = np.empty(n_alpha, dtype = np.float64)
    n_alpha_array = np.arange(n_alpha, dtype = np.float64)
    double_n_alpha = 2 * n_alpha

    # used to calculate sample points x_i
    points = np.empty(n, dtype = np.float64)
    i = np.arange(n)

    k_tilde_terms = np.empty(n, dtype = np.float64)
    best_k_tilde = np.empty(n, dtype = np.float64)

    # used to calculate squared discrepancy
    n_array = np.arange(1, n + 1)

    for d in tqdm(range(1, dimension + 1), disable = disable):
        calculate_choices(n_alpha_array, n_alpha, double_n_alpha, choices)

        best_wssd = np.inf
        best_choice = 0

        for choice in choices:
            calculate_points(choice, i, points)
            calculate_k_tilde_terms(points, coord_wt[d - 1], previous_k_tilde, k_tilde_terms)

            wssd = np.sum(sample_wt * square_discrepancy(k_tilde_terms, n_array, n))
            if wssd < best_wssd:
                best_wssd = wssd
                best_choice = choice
                np.copyto(best_k_tilde, k_tilde_terms)

        best_alpha[d - 1] = best_choice
        np.copyto(previous_k_tilde, best_k_tilde)
        wssd = best_wssd

    return best_alpha, wssd

def alpha_search_with_start(dimension, n_alpha, n, sample_wt, coord_wt, alpha_start, disable = False):
    best_alpha = np.empty(dimension)
    best_alpha[:alpha_start.size] = alpha_start

    previous_k_tilde = calculate_k_tilde_total(alpha_start, n, coord_wt)

    choices = np.empty(n_alpha, dtype = np.float64)
    n_alpha_array = np.arange(n_alpha, dtype = np.float64)
    double_n_alpha = 2 * n_alpha

    # used to calculate sample points x_i
    points = np.empty(n, dtype = np.float64)
    i = np.arange(n)

    k_tilde_terms = np.empty(n, dtype = np.float64)
    best_k_tilde = np.empty(n, dtype = np.float64)

    # used to calculate squared discrepancy
    n_array = np.arange(1, n + 1)

    for d in tqdm(range(alpha_start.size + 1, dimension + 1), disable = disable):
        calculate_choices(n_alpha_array, n_alpha, double_n_alpha, choices)

        best_wssd = np.inf
        best_choice = 0

        for choice in choices:
            calculate_points(choice, i, points)
            calculate_k_tilde_terms(points, coord_wt[d - 1], previous_k_tilde, k_tilde_terms)

            wssd = np.sum(sample_wt * square_discrepancy(k_tilde_terms, n_array, n))
            if wssd < best_wssd:
                best_wssd = wssd
                best_choice = choice
                np.copyto(best_k_tilde, k_tilde_terms)

        best_alpha[d - 1] = best_choice
        np.copyto(previous_k_tilde, best_k_tilde)
        wssd = best_wssd

    return best_alpha, wssd

# I'm not convinced this is any faster, but could save temporary array creations?
def calculate_choices(n_alpha_array, n_alpha, double_n_alpha, choices):
    np.add(n_alpha_array, np.random.rand(n_alpha), out = choices)
    np.divide(choices, double_n_alpha, out = choices)

# This is faster
def calculate_points(choice, i, points):
    np.multiply(i, choice, out = points)
    np.subtract(points, np.floor(points), out = points)

# I'm not convinced this is any faster, but could save temporary array creations?
def calculate_k_tilde_terms(points, coord_wt, previous_k_tilde, results):
    np.subtract(points, 1, out = results)
    np.multiply(points, results, out = results)
    np.add(1/6, results, out = results)
    np.multiply(coord_wt, results, out = results)
    np.add(1, results, out = results)
    np.multiply(previous_k_tilde, results, out = results)

def square_discrepancy(k_tilde_terms, n_array, n):
    left_sum = np.cumsum(k_tilde_terms[1:]) * n_array[1:]
    right_sum = np.cumsum(n_array[:-1] * k_tilde_terms[1:])

    k_tilde_zero_terms = k_tilde_terms[0] * n_array
    summation = np.zeros(n)
    summation[1:] = left_sum - right_sum
    return (k_tilde_zero_terms + 2 * summation) / (n_array ** 2) - 1

def calculate_k_tilde_total(alpha_start, n, coord_wt):
    points = (np.arange(n).reshape((n, 1)) * alpha_start)
    np.subtract(points, np.floor(points), out = points)

    k_tilde = lambda x, gamma: np.prod(1 + (x * (x - 1) + 1/6) * gamma, axis=1)
    return k_tilde(points, coord_wt[:alpha_start.size])

def plot_discrepancy(dimension, alpha, n = 1e6, gamma = None, trend = True, title = None):
    kronecker = qp.Kronecker(dimension = dimension, alpha = alpha)

    if gamma is None:
        values = kronecker.periodic_discrepancy(int(n))
    else:
        values = kronecker.periodic_discrepancy(int(n), gamma = gamma)

    x = np.arange(1, int(n) + 1)
    y = values[0] / x  # multiply by values[0] to get the same starting point
    y_half = values[0] / np.sqrt(x)
    if trend:
        plt.loglog(x, y_half, label = '$O(n^{-.5})$', linestyle = '--')
        plt.loglog(x, y, label = '$O(n^{-1})$', linestyle = '--')

    plt.loglog(x, values, label = "Kronecker")

    if title is not None:
        plt.title(title)

    plt.xlabel('Number of sample points')
    plt.ylabel('Discrepancy')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    experiment = False
    search = True

    # if experiment:
    #     n_alphas = np.array([2e3, 5e3, 1e4, 2e4, 5e4, 1e5, 2e5, 1e6], dtype = np.uint32)
    #     trials = 6
    #     dim = 20
    #     sample_number = int(1e3)

    #     sample_weights = np.arange(1, sample_number + 1)
    #     coord_weights = 1 / (np.arange(1, dim + 1) ** 2)

    #     results = np.empty(shape = (trials, len(n_alphas)), dtype = object)
    #     for trial in tqdm(range(trials)):
    #         for i, n_al in enumerate(tqdm(n_alphas)):
    #             alpha, wdisc, duration = alpha_search(dimension = dim, n_alpha = n_al,
    #                                                   n = sample_number, sample_wt = sample_weights,
    #                                                   coord_wt = coord_weights, disable = True)
    #             results[trial][i] = (alpha, wdisc, duration)

    #     np.save('n_alpha_results/results_small_n', results)

    if search:
        dim = 100
        n_al = int(2e5)
        sample_number = int(2 ** 20)
        sample_weights = np.arange(1, sample_number + 1)
        coord_weights = 1 / (np.arange(1, dim + 1) ** 2)

        print('starting')
        alpha, wdisc = alpha_search(dimension = dim, n_alpha = n_al,
                                              n = sample_number, sample_wt = sample_weights,
                                              coord_wt = coord_weights)
        print('finished')
        np.savetxt('kron_vector/d100_n2^20_2e5alpha.txt', alpha)