import numpy as np


class PCA:
    def __init__(self):
        self.pc_ls = None
        self.var_ls = None

    def fit(self, X, method="svd"):
        X = X - X.mean(0)

        pc_ls = None
        var_ls = None

        if method == "svd":
            _, sing_val_ls, pc_ls = np.linalg.svd(X / np.sqrt(len(X) - 1))
            var_ls = sing_val_ls ** 2

        elif method == "eig-dec":
            cov = X.T @ X / (len(X) - 1)
            var_ls, pc_ls = np.linalg.eigh(cov)
            sort_idx = np.argsort(var_ls)[::-1]
            var_ls = var_ls[sort_idx]
            pc_ls = pc_ls.T[sort_idx]

        else:
            raise NotImplementedError(f"{method} method not implemented!")
        self.pc_ls, self.var_ls = pc_ls, var_ls

    def transform(self, X, k):
        return X @ self.pc_ls[:k].T