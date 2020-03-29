import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.manifold import SpectralEmbedding
from sklearn.decomposition import PCA
from sklearn.manifold import MDS


class UMAPDimReduction:
    def __init__(self, n_components=2, a=1, b=1, random_state=None, max_iter=300, learning_rate=1):
        """
        NOTE that at this point I will use a=1 and b=1 need to read and understand implications
        since UMAP does curve fit to a and b based on the min_dist hyper-parameter
        :param a:
        :param b:
        """
        self.n_components = n_components
        self.a = a
        self.b = b
        self.random_state = random_state
        self.max_iter = max_iter
        self.learning_rate = learning_rate


    def _prob_low_dim(self, Y):
        """
        Compute matrix of probabilities q_ij in low-dimensional space
        """
        inv_distances = np.power(1 + self.a * np.square(euclidean_distances(Y, Y)) ** self.b, -1)
        return inv_distances

    def _CE(self, P, Y):
        """
        Compute Cross-Entropy (CE) from matrix of high-dimensional probabilities
        and coordinates of low-dimensional embeddings
        """
        Q = self._prob_low_dim(Y)
        # TODO ORM I guess that 0.01 is epsilon, if so take it out to constant - yes
        # return - P * np.log(Q + 0.01) - (1 - P) * np.log(1 - Q + 0.01)
        return - P * np.log(Q+0.00001) - (1 - P) * np.log(1 - Q+0.00001)

    def _CE_gradient(self, P, Y):
        """
        Compute the gradient of Cross-Entropy (CE)
        """
        y_diff = np.expand_dims(Y, 1) - np.expand_dims(Y, 0)
        inv_dist = self._prob_low_dim(Y)
        Q = np.dot(1 - P, np.power(0.001 + np.square(euclidean_distances(Y, Y)), -1))
        np.fill_diagonal(Q, 0)
        # TODO in the UMAP paper there is no normalization here in the implementation that I used there is normalization
        # TODO in row level, I will keep this normalization like I did in the high dimension
        # TODO Pay attention to the normalization which is different from t-SNE
        # TODO ORM this code doesn't work well with different normalization - need to check the derivation
        Q = Q / np.sum(Q, axis=1, keepdims=True)  # Normalization
        # TODO ORM I will do normalization as in the high dim
        # Q = Q / np.sum(Q)
        # Q = (Q + Q.T) / (2*len(Q))  # Symmetry - No need in low dimension!

        fact = np.expand_dims(self.a * P * (1e-8 + np.square(euclidean_distances(Y, Y))) ** (self.b - 1) - Q, 2)
        return 2 * self.b * np.sum(fact * y_diff * np.expand_dims(inv_dist, 2), axis=1)

    def fit_transform(self, X, labels, P):
        if self.random_state is not None:
            np.random.seed(self.random_state)
        # TODO ORM neighbors?
        # TODO ORM for simplicity I will start will PCA initialization or random
        model = SpectralEmbedding(n_components=self.n_components, n_neighbors=5)
        # model = PCA(n_components=self.n_components)
        # model = MDS(n_components=self.n_components)
        # TODO ORM why it was fit transform on log?
        # y = model.fit_transform(np.log(X + 1))
        y = model.fit_transform(X)
        # y = np.random.normal(loc = 0, scale = 1, size = (n, N_LOW_DIMS))
        y = np.random.rand(len(X), 2)

        # TODO ORM debug
        print(labels.astype(int))
        plt.figure(figsize=(20, 15))
        plt.scatter(y[:, 0], y[:, 1], c=labels.astype(int), cmap='tab10', s=50)
        plt.title("UMAP",
                  fontsize=20)
        plt.xlabel("UMAP1", fontsize=20);
        plt.ylabel("UMAP2", fontsize=20)
        plt.savefig('UMAP_Plots/UMAP_iter_' + 'init' + '.png')
        plt.close()


        CE_array = []
        print("Running Gradient Descent: \n")
        for i in range(self.max_iter):
            y = y - self.learning_rate * self._CE_gradient(P, y)

            plt.figure(figsize=(20, 15))
            plt.scatter(y[:, 0], y[:, 1], c=labels.astype(int), cmap='tab10', s=50)
            plt.title("UMAP",
                      fontsize=20)
            plt.xlabel("UMAP1", fontsize=20);
            plt.ylabel("UMAP2", fontsize=20)
            plt.savefig('UMAP_Plots/UMAP_iter_' + str(i) + '.png')
            plt.close()

            CE_current = np.sum(self._CE(P, y)) / 1e+5
            CE_array.append(CE_current)
            if i % 10 == 0:
                print("Cross-Entropy = " + str(CE_current) + " after " + str(i) + " iterations")

        plt.figure(figsize=(20, 15))
        plt.plot(CE_array)
        plt.title("Cross-Entropy", fontsize=20)
        plt.xlabel("ITERATION", fontsize=20);
        plt.ylabel("CROSS-ENTROPY", fontsize=20)
        plt.show()
        return y





