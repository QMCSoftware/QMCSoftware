import qmcpy as qp
import numpy as np
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

def alpha_search(dimension, n_alpha, n, sample_wt=None, coord_wt=None, seed=None):
    if coord_wt is None:
        coord_wt = np.ones(dimension)

    best_alpha = np.empty(dimension)
    previous_k_tilde = 1

    n_alpha_array = np.arange(n_alpha)
    n_array = np.arange(1, n + 1)
    i = np.arange(n)
    double_n_alpha = 2 * n_alpha

    for d in tqdm(range(1, dimension + 1)):
        choices = (n_alpha_array + np.random.rand(n_alpha)) / double_n_alpha
        best_wssd = np.inf
        best_choice = 0
        best_k_tilde = 0
        for choice in choices:
            points = (i * choice) % 1
            k_tilde_terms = (1 + (points * (points - 1) + 1 / 6) * coord_wt[d - 1]) * previous_k_tilde
            del points

            wssd = np.sum(sample_wt * square_discrepancy(k_tilde_terms, n_array, n))
            if wssd < best_wssd:
                best_wssd = wssd
                best_choice = choice
                best_k_tilde = k_tilde_terms

        best_alpha[d - 1] = best_choice
        previous_k_tilde = best_k_tilde
        wssd = best_wssd

    return best_alpha, wssd, previous_k_tilde


def square_discrepancy(k_tilde_terms, n_array, n):
    left_sum = np.cumsum(k_tilde_terms[1:]) * n_array[1:]
    right_sum = np.cumsum(n_array[:-1] * k_tilde_terms[1:])

    k_tilde_zero_terms = k_tilde_terms[0] * n_array
    summation = np.zeros(n)
    summation[1:] = left_sum - right_sum
    return (k_tilde_zero_terms + 2 * summation) / (n_array ** 2) - 1


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
    np.set_printoptions(18)
    dim = 50
    n_al = int(1e5)
    sample_number = int(2 ** 17)
    sample_weights = np.arange(1, sample_number + 1)
    coord_weights = 1 / (np.arange(1, dim + 1) ** 2)
    alpha, wssd, k_tilde = alpha_search(dimension = dim,
                 n_alpha = n_al,
                 n = sample_number,
                 sample_wt = sample_weights,
                 coord_wt = coord_weights)

    np.savetxt('d50_n2^17_1e5alpha', alpha)
    print(k_tilde)
    print(wssd)
    plot_discrepancy(dim, alpha, gamma = coord_weights, title='50 dimensions')