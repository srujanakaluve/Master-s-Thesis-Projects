import numpy as np
import matplotlib.pyplot as plt
import math
from itertools import combinations


def sign(x):

    y = np.sign(x)

    if type(x) == int or type(x) == float:
        y = 1 if y == 0 else y
        return y
    else:  #should be list or np array
        y[y == 0] = 1
    return y


def topk(x, k):

    topk_x = np.zeros_like(x)

    if x.ndim == 1:
        k_indices = np.argsort(x)[-k:]
        topk_x[k_indices] = 1
 
    elif x.ndim == 2:
        k_indices = np.argsort(x, axis=0)[-k:, :]
        col_indices = np.arange(x.shape[1])
        topk_x[k_indices, col_indices] = 1

    return topk_x


def k_hot_patterns(NL, k):
    idx_combinations = list(combinations(range(NL), k))
    matrix = np.zeros((NL, len(idx_combinations)))
    for i in range(matrix.shape[1]):
        matrix[:, i][list(idx_combinations[i])] = 1
    return matrix


def get_random_patterns(p, NF):
    patterns = np.zeros((NF, p))
    for i in range(p):
        pattern = np.random.choice([-1, 1], size=NF)
        patterns[:, i] = pattern
    return patterns


def corrupt_pattern(pattern, noise):
    rand_indices = np.where(np.random.rand(pattern.shape[0]) < noise, -1, 1)
    corrupted = pattern * rand_indices
    return corrupted


def get_overlap(state, pattern):
    return np.mean(state * pattern, axis=0)


def get_mutual_info(state, pattern):
    m = np.array(get_overlap(state, pattern))
    a, b = (1 + m)/2, (1 - m)/2
    s = a * np.log2(np.where(a > 0, a, 1)) + b* np.log2(np.where(b > 0, b, 1))
    return 1 + s


def scaffold_layers(NL, NH, C, k):

    W_HL = np.random.normal(0, 1, size=(NH, NL))
    L = k_hot_patterns(NL, k)

    # get hidden layer fixed points
    H = sign(W_HL @ L)

    # learn H to L weights
    W_LH = (1/C) * (L @ H.T)

    return L, W_HL, H, W_LH


def scaffold_step(h, W_HL, W_LH, k):
    l = topk(W_LH @ h, k)
    h_step = sign(W_HL @ l)
    return l, h_step


def feature_weights(F, H):
    N_patts = F.shape[1]

    #scaffold has n choose k points but we are only storing N_patts patterns, so learn weights with N_patts learned patterns only
    H_till_Npatts = H[:, :N_patts]

    W_HF = H_till_Npatts @ np.linalg.pinv(F)
    W_FH = F @ np.linalg.pinv(H_till_Npatts)

    return W_HF, W_FH


def mesh_recall(f_cued, k, W_HF, W_LH, W_HL, W_FH):
    f = f_cued
    h = sign(W_HF @ f)
    l = topk(W_LH @ h, k)
    h = sign(W_HL @ l)
    f = sign(W_FH @ h)

    return f


if __name__ == "__main__":

    NL, NH, NF, k = 18, 300, 816, 3
    C = math.comb(NL, k)
    L, W_HL, H, W_LH = scaffold_layers(NL, NH, C, k)

    pattern_range = np.arange(1, 800, 25)
    plot_list = []

    for N_patts in pattern_range:
        F = get_random_patterns(N_patts, NF)
        W_HF, W_FH = feature_weights(F, H)

        #noisy cues
        #F_cued = F * np.where(np.random.rand(F.shape[0], F.shape[1]) < 0.2, -1, 1)
        #F_recalled = mesh_recall(F_cued, k, W_HF, W_LH, W_HL, W_FH)

        #perfect cues 
        F_recalled = mesh_recall(F, k, W_HF, W_LH, W_HL, W_FH)

        #plot overlap
        #overlap = np.mean(get_overlap(F_recalled, F))
        #plot_list.append(overlap)

        #plot mutual information
        mutual_info = np.mean(get_mutual_info(F_recalled, F))
        plot_list.append(mutual_info)

    plt.scatter(pattern_range, plot_list)
    plt.xlabel("Number of stored patterns")
    plt.ylabel("MI")
    plt.axvline(x=NH)
    plt.show()