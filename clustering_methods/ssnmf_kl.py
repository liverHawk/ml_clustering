from dataclasses import dataclass

import numpy as np


@dataclass
class SSNMFKLConfig:
    n_clusters: int
    lambda_reg: float
    metric: str
    max_iter: int
    tol: float
    random_state: int
    verbose: bool


class SSNMFKL:
    def __init__(self, config: SSNMFKLConfig):
        self.config = config
        
        self.eps = 1e-10

        self.V = None
        self.U = None
        self.H = None

    def _check_input(self, X, Y, L):
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)
        L = np.asarray(L, dtype=float)

        if np.any(X < 0):
            raise ValueError("X must be non-negative.")
        if np.any(Y < 0):
            raise ValueError("Y must be non-negative.")
        if np.any(L < 0) or np.any(L > 1):
            raise ValueError("L must be non-negative.")

        n_samples, n_features = X.shape
        n_labels, n_samples_y = Y.shape
        n_labels_l, n_samples_l = L.shape

        if n_samples != n_samples_y or n_samples != n_samples_l:
            raise ValueError("X, Y, and L must have the same number of samples.")
        if n_labels != n_labels_l:
            raise ValueError("X, Y, and L must have the same number of labels.")

        return X, Y, L

    def _update_frobenius(self, X, Y, L):
        
        if self.H is None:
            raise ValueError("Model is not fitted yet.")
        XH = X @ self.H.T
        VHTH = self.V @ (self.H @ self.H.T)
        VHTH = np.where(VHTH == 0, self.eps, VHTH)
        self.V *= XH / VHTH
        self.V = np.maximum(self.V, self.eps)

        if self.U is None:
            raise ValueError("Model is not fitted yet.")

        VTX = self.V.T @ X
        UTY = self.U.T @ Y
        VTVH = (self.V.T @ self.V) @ self.H
        UTUH = (self.U.T @ self.U) @ self.H

        number = VTX + self.config.lambda_reg * UTY
        denom = VTVH + self.config.lambda_reg * UTUH
        denom = np.where(denom == 0, self.eps, denom)

        self.H *= number / denom
        self.H = np.maximum(self.H, self.eps)

    def _update_kl(self, X, Y, L):
        if self.H is None:
            raise ValueError("Model is not fitted yet.")

        VH = self.V @ self.H
        UH = self.U @ self.H

        ratio_X = X / (VH + self.eps)
        numer_V = ratio_X @ self.H.T
        denom_V = np.ones((X.shape[0], 1)) @ np.ones((1, self.H.shape[0])) @ self.H.T
        denom_V = np.where(denom_V == 0, self.eps, denom_V)
        self.V *= numer_V / (denom_V + self.eps)
        self.V = np.maximum(self.V, self.eps)

        ratio_Y = Y / (UH + self.eps)
        ratio_Y_masked = ratio_Y * L
        numer_U = ratio_Y_masked @ self.H.T
        denom_U = np.ones((Y.shape[0], 1)) @ np.ones((1, self.H.shape[0])) @ self.H.T
        denom_U = np.where(denom_U == 0, self.eps, denom_U)
        self.U *= numer_U / (denom_U + self.eps)
        self.U = np.maximum(self.U, self.eps)

        ratio_X_for_H = X / (VH + self.eps)
        ratio_Y_for_H = Y / (UH + self.eps)

        ratio_Y_for_H_masked = ratio_Y_for_H * L

        numer_H = self.V.T @ ratio_X_for_H + self.config.lambda_reg * self.U.T @ ratio_Y_for_H_masked
        denom_H = np.ones((1, self.V.shape[0])) @ self.V + self.config.lambda_reg * np.ones((1, self.U.shape[0])) @ self.U
        denom_H = denom_H.T @ np.ones((1, self.H.shape[1]))
        denom_H = np.where(denom_H == 0, self.eps, denom_H)

        self.H *= numer_H / (denom_H + self.eps)
        self.H = np.maximum(self.H, self.eps)

    def fit_predict(self, X, Y, L):
        X, Y, L = self._check_input(X, Y, L)

        X_T = X.T

        n_features, n_samples = X_T.shape
        n_labels = Y.shape[0]
        n_components = self.config.n_clusters


