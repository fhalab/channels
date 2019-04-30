## Tools for GP 

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
import pickle
from scipy.linalg import cho_solve


# ML imports
from sklearn import linear_model
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial import distance
from scipy import optimize, linalg
import scipy

def matern_5_2_kernel(X, X_, hypers):
    """ Calculate the Matern kernel between X and X_.
    Parameters:
        X (np.ndarray):
        X_ (np.ndarray)
        hypers (iterable): default is ell=1.0.
    Returns:
        K (np.ndarray)
    """
    D = euclidean_distances(X, X_)
    D_L = D / hypers[0]
    
    first = (1.0 + np.sqrt(5.0) * D_L) + 5.0 * D_L ** 2 / 3.0
    second = np.exp(-np.sqrt(5.0) * D_L)
    
    K = first * second
    return K

def predict_GP(X_train, y_train, X_test, prams):
    """ Gaussian process regression predictions.
    Parameters:
        X_train (np.ndarray): n x d training inputs
        y_train (np.ndarray): n training observations
        X_test (np.ndarray): m x d points to predict
    Returns:
        mu (np.ndarray): m predicted means
        var (np.ndarray): m predictive variances
    """

    K = matern_5_2_kernel(X_train, X_train, prams[1:])
    Ky = K + np.identity(len(K))*prams[0]
    
    # To invert K_y we use the Cholesky decomposition (L)
    L = np.linalg.cholesky(Ky)
    
    # solve for z=L^-1y
    z = linalg.solve_triangular(L, y_train, lower=True)
    alpha = linalg.solve_triangular(L.T, z, lower=False)
    
    K_star = matern_5_2_kernel(X_train, X_test, prams[1:])
    mu = np.matmul(K_star.T, alpha)
    
    
    # Compute the variance at the test points
    z = linalg.solve_triangular(L, K_star, lower=True)
    alpha = linalg.solve_triangular(L.T, z, lower=False)
    K_star_star = matern_5_2_kernel(X_test, X_test, prams[1:])
    v = np.diag(K_star_star) - np.dot(K_star.T, alpha)
    v = np.diag(v)
    return mu, v

def neg_log_marg_likelihood(log_prams, X, y):
    """ Calculate the negative log marginal likelihood loss.
    We pass the log hypers here because it makes the optimization
    more stable.
    Parameters:
        log_hypers (np.ndarray): natural log of the hyper-parameters
        X (np.ndarray)
        y (np.ndarray)
    Returns:
        (float) The negative log marginal likelihood.
    """
    
    non_log_prams = np.exp(log_prams)
    #print(non_log_prams)
    
    # Evaluate kernel on training data
    K = matern_5_2_kernel(X, X, non_log_prams[1:])

    # To invert K we use the Cholesky decomposition (L), because symmetric and positive  definite    
    n = len(y)
    Ky = K + np.identity(len(K))*non_log_prams[0]
    L = np.linalg.cholesky(Ky)
    z = linalg.solve_triangular(L, y, lower=True)
    alpha = linalg.solve_triangular(L.T, z, lower=False) #dont know about this
    
    first = 0.5 * np.dot(y, alpha)
    second = np.sum(np.log(np.diag(L)))
    third = 0.5 * len(K) * np.log(2 * np.pi)
    
    #log_p_y_X = 0.5*np.matmul(y, alpha) + np.sum(np.log(np.diag(L))) + 0.5*n*np.log(2*np.pi)
    log_p_y_X = (first + second + third)
    return log_p_y_X