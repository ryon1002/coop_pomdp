import numpy as np
from scipy.special import softmax

def _max_q_prob(arr):
    ret = (arr == np.max(arr)).astype(np.int)
    return ret / np.sum(ret)

def _exp_q_prob(arr, beta=1):
    ret = np.exp(arr * beta)
    if np.sum(ret) == 0:
        ret = np.ones_like(ret)
    return ret / np.sum(ret)

def exp_prob(arr, beta=1):
    ret = np.exp(arr * beta)
    if np.sum(ret) == 0:
        ret = np.ones_like(ret)
    return ret / np.sum(ret)

def _avg_prob(arr):
    if np.sum(arr) == 0:
        return arr
    return arr / np.sum(arr)


def make_poilcy(q, beta, **kwargs):
    if beta == np.inf:
        return _max_q_prob_one(q)
    return softmax(q, axis=1)

def _max_q_prob_one(arr):
    return np.identity(arr.shape[1])[np.argmax(arr, axis=1)]

