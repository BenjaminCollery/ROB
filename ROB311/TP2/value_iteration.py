#! /usr/bin/env python
import numpy as np


def close_enough(V, Vp, theta):
    """
    Check if |V(S)-Vp(S)| < theta for all S.

    :param V: An array
    :param Vp: Another array
    :type V: numpy.ndarray
    :type Vp: numpy.ndarray
    :return: Whether V and Vp are close close enough
    :rtype: bool"""
    return np.max(np.abs(V - Vp)) < theta

def value_iteration(S, A, P, R, theta, gama):
    """
    Implementation of the value iteration algorithm.

    :param S: The set of states
    :param A: The set of actions
    :param P: The transition matricies
    :param theta: Precision of the result
    :param gama: The gama parameter
    :type S: numpy.ndarray
    :type A: numpy.ndarray
    :type P: numpy.ndarray
    :type theta: float
    :type gama: float
    :returns: The utility of each state and the optimal policy
    :rtype: (numpy.ndarray, numpy.ndarray)
    """
    V = np.zeros(S.shape)
    Vp = (1+theta)*np.ones(S.shape)
    # Compute iteratively the utility function
    while not close_enough(V, Vp, theta):
        V = Vp
        for s in S:
            X = np.zeros(A.shape[0])
            for a in A:
                for sp in S:
                    X[a] += np.sum(P[a][s][sp]*V[sp])
            Vp[s] = R[s] + gama*np.max(X)

    # Compute the optimal policy for each state
    pi = np.zeros(S.shape[0])
    for s in S:
        X = np.zeros(A.shape[0])
        for a in A:
            for sp in S:
                X[a] += np.sum(P[a][s][sp]*V[sp])
        pi[s] = np.argmax(X)

    return V, pi


if __name__ == "__main__":
    x = 0.25
    y = 0.25
    gama = 0.9

    S = np.array([0, 1, 2, 3])
    A = np.array([0, 1, 2])
    # x: action
    # y: état de départ
    # z: état d'arrivé
    # P[x][y][z] = P(z|y,x)
    P = np.array([[[  0,   0,   0,   0],
                   [  0, 1-x,   0,   x],
                   [1-y,   0,   0,   y],
                   [  1,   0,   0,   0]],
                  [[  0,   1,   0,   0],
                   [  0,   0,   0,   0],
                   [  0,   0,   0,   0],
                   [  0,   0,   0,   0]],
                  [[  0,   0,   1,   0],
                   [  0,   0,   0,   0],
                   [  0,   0,   0,   0],
                   [  0,   0,   0,   0]]
    ])
    R = np.array([0, 0, 1, 10])
    theta = 1e-3

    V, pi = value_iteration(S, A, P, R, theta, gama)
    print("Utility of each state: ", V)
    print("Optimal policy for each state: ", pi)
