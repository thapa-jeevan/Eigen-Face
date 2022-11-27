import numpy as np


class LDA:
    def __init__(self):
        self.pi_ls = None
        self.u_ls = None
        self.prec = None

    def fit(self, X_train, y_train):
        pi_ls = []
        u_ls = []
        var_ = 0

        N, K = X_train.shape
        categories = np.sort(np.unique(y_train))

        for k in categories:
            k_idx = (y_train == k)
            N_k = sum(k_idx)
            X_k = X_train[k_idx]

            pi_ls.append(N_k / N)
            u_k = X_k.mean(axis=0)
            u_ls.append(u_k)

            var_ += (X_k - u_k).T @ (X_k - u_k) / (N - len(categories))

        self.pi_ls = np.array(pi_ls)
        self.u_ls = np.vstack(u_ls).T
        self.prec = np.linalg.inv(var_)

    def predict(self, X):
        if 1 in X.shape or len(X.shape) == 1:
            X = X.reshape(1, -1)
        post_rel = X @ self.prec @ self.u_ls - \
                   1 / 2 * np.diag(self.u_ls.T @ self.prec @ self.u_ls) + \
                   np.log(self.pi_ls)
        return np.argmax(post_rel, axis=1)

    def __repr__(self):
        return "LDA model"
