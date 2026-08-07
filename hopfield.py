import numpy as np
import matplotlib.pyplot as plt


def sign(x):
    y = np.sign(x)
    if isinstance(x, (int, float)):  # apparently isinstance() is better than type() for a type check
        y = 1 if y == 0 else y
        return y
    else:  #should be list or np array
        y[y == 0] = 1
    return y


def get_random_patterns(p, N):
    patterns = np.zeros((N, p))
    for i in range(p):
        pattern = np.random.choice([-1, 1], size=N)
        patterns[:, i] = pattern
    return patterns


def get_weights(N, patterns):
    weights = (1/N) * patterns @ patterns.T  #note: cant use np.outer because it flattens inputs first - will give (N*p, N*p) shape
    np.fill_diagonal(weights, 0)
    return weights


def recall(state, weights, N, iterations, recall_type="sync", track_overlap=False, track_energy=False, pattern=None):

    energies = np.zeros(iterations)
    overlaps = np.zeros(iterations)
    state_1 = state.copy()
    state_list = []
    state_list.append(state_1)
    for iteration in range(iterations):

        if recall_type == "sync":
            state_1 = sign(weights @ state_1)
            state_list.append(state_1)

        if recall_type =="async":

            for i in range(N):
                state_1[i] = np.sign(weights[i, :] @ state_1)
                if state_1[i] == 0:   # because np.sign returns values -1, 0 and 1
                    state_1[i] = 1
            state_list.append(state_1)

        if track_overlap is True and pattern is not None:
            overlap = get_overlap(state_1, pattern)
            overlaps[iteration] = overlap

        if track_energy is True:
            energy = get_energy(state_1, weights)
            energies[iteration] = energy
        
        if np.array_equal(state_1, state_list[-2]):
            break
        
    return state_1, energies, overlaps


def get_energy(state, weights):
    return -0.5 * state @ weights @ state


def get_overlap(state, pattern):
    return np.dot(state, pattern) / len(state)


def get_mutual_info(state, pattern):
    m = np.array(get_overlap(state, pattern))
    a, b = (1 + m)/2, (1 - m)/2
    s = a * np.log2(np.where(a > 0, a, 1)) + b* np.log2(np.where(b > 0, b, 1))
    return 1 + s


def corrupt_pattern(pattern, noise):
    rand_indices = np.where(np.random.rand(pattern.shape[0]) < noise, -1, 1)
    corrupted = pattern * rand_indices
    return corrupted


def capacity(N, iterations, pattern_range, corrupt_frac=0.2, recall_type="sync"):

    Mi_list = []

    for j in pattern_range:

        patterns = get_random_patterns(j, N)
        weights = get_weights(N, patterns)
        Mi = []

        for i in range(j):

            corrupted = corrupt_pattern(patterns[:, i], corrupt_frac)
            recovered, _, _ = recall(corrupted, weights, N, iterations, recall_type)
            Mi.append(get_mutual_info(recovered, patterns[:, i]))

        Mi_list.append(np.mean(Mi))

    return pattern_range, Mi_list


if __name__ == "__main__":
    N = 708
    iterations = 200
    pattern_range = np.arange(1, 800, 25)

    pattern_range, distances = capacity(N, iterations, pattern_range)


    plt.scatter(pattern_range, distances)
    plt.xlabel('Number of stored patterns')
    plt.ylabel('MI')
    plt.show()