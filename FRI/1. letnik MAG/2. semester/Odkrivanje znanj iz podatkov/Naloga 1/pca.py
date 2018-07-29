import numpy as np
import sklearn.decomposition
from numpy import genfromtxt
from sklearn import datasets
from matplotlib import pyplot as plt


def read_data(path):
    data = genfromtxt(path, delimiter=',', skip_header=1)
    y = data[:, 0]
    x = data[:, 1:]
    return [x, y]


def covariance_matrix(x):
    m = x.shape[0]
    return (1/m) * np.dot(np.transpose(x), x)


def projection_v2u(u, v):
    return (np.dot(u, v) / np.dot(u, u)) * u


class EigenPCA:
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.mean_ = None
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None

    def fit(self, x):
        self.mean_ = x.mean(axis=0)
        x -= self.mean_
        cov_mat = covariance_matrix(x)

        eigenvalues, eigenvectors = np.linalg.eigh(cov_mat)
        eigenvalues = np.flip(eigenvalues, axis=0)
        eigenvectors = np.flip(eigenvectors, axis=1)

        self.components_ = np.transpose(eigenvectors[:, :self.n_components])
        self.explained_variance_ = eigenvalues[:self.n_components]
        self.explained_variance_ratio_ = self.explained_variance_ / sum(eigenvalues)
        return self

    def transform(self, x):
        x -= self.mean_
        return np.dot(x, np.transpose(self.components_))


class PowerPCA:
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.mean_ = None
        self.components_ = []
        self.explained_variance_ = []
        self.explained_variance_ratio_ = []

    def power_iterate(self, cov_mat, n, eps=1e-10, max_iterations=1000):
        x = np.random.rand(n)
        e = 1.0
        iteration = 0
        while e >= eps and iteration < max_iterations:
            iteration += 1
            x_new = np.dot(cov_mat, x)
            x_new /= np.linalg.norm(x_new)
            e = np.linalg.norm(x_new - x)
            x = x_new

        eigenvector = x
        eigenvalue = np.dot(np.transpose(eigenvector), np.dot(cov_mat, eigenvector))
        return [eigenvalue, eigenvector]

    def fit(self, x):
        self.mean_ = x.mean(axis=0)
        x = x - self.mean_
        cov_mat = covariance_matrix(x)
        trace_cov_mat = np.trace(cov_mat)
        n = x.shape[1]

        for i in range(self.n_components):
            eigenvalue, eigenvector = self.power_iterate(cov_mat, n)
            self.components_.append(eigenvector)
            self.explained_variance_.append(eigenvalue)
            cov_mat -= eigenvalue * np.dot(eigenvector.reshape(eigenvector.shape[0], 1), eigenvector[np.newaxis])

        self.explained_variance_ = np.array(self.explained_variance_)
        self.explained_variance_ratio_ = self.explained_variance_ / trace_cov_mat
        return self

    def transform(self, x):
        x -= self.mean_
        return np.dot(x, np.transpose(self.components_))


class OrtoPCA:
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.mean_ = None
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None

    def power_iterate(self, cov_mat, n, eps=1e-10, max_iterations=1000):
        u = np.random.rand(n, self.n_components)
        e = 1.0
        iteration = 0
        while e >= eps and iteration < max_iterations:
            iteration += 1
            u_new = np.dot(cov_mat, u)

            # Gram–Schmidt orthonormalization
            for i in range(self.n_components):
                v_i = u_new[:, i]
                u_i = v_i - sum([projection_v2u(u_new[:, j], v_i) for j in range(0, i)])
                u_i /= np.linalg.norm(u_i)
                u_new[:, i] = u_i

            e = np.linalg.norm(u_new - u)
            u = u_new

        eigenvectors = u
        eigenvalues = np.diagonal(np.dot(np.transpose(eigenvectors), np.dot(cov_mat, eigenvectors)))
        return [eigenvalues, eigenvectors]

    def fit(self, x):
        self.mean_ = x.mean(axis=0)
        x = x - self.mean_
        cov_mat = covariance_matrix(x)
        trace_cov_mat = np.trace(cov_mat)
        n = x.shape[1]

        eigenvalues, eigenvectors = self.power_iterate(cov_mat, n)

        self.components_ = np.transpose(eigenvectors)
        self.explained_variance_ = eigenvalues
        self.explained_variance_ratio_ = self.explained_variance_ / trace_cov_mat
        return self

    def transform(self, x):
        x -= self.mean_
        return np.dot(x, np.transpose(self.components_))


if __name__ == "__main__":
    X, y = read_data('train.csv')

    power_pca = PowerPCA()
    power_pca.fit(X)
    transformed_points = power_pca.transform(X)
    x_1 = transformed_points[:, 0]
    x_2 = transformed_points[:, 1]

    distinct_colors = ["#e6194b", "#3cb44b", "#ffe119", "#0082c8", "#f58231",
                       "#911eb4", "#46f0f0", "#f032e6", "#d2f53c", "#008080"]
    colors = list(map(lambda n: distinct_colors[int(n)], y))

    fig = plt.figure()
    for i in range(10):
        indices = np.where(y == i)[0]
        X_i = transformed_points[indices]
        center = np.mean(X_i, axis=0)
        plt.text(center[0], center[1], str(i), fontsize=10, bbox=dict(facecolor=distinct_colors[i], alpha=1))
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.scatter(x_1, x_2, c=colors, marker=".")
    plt.show()

    fig.savefig("pca.pdf", bbox_inches='tight')
