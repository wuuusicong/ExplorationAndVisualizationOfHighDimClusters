import numpy as np
import pandas as pd
import random
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph
from scipy.sparse import csr_matrix
from UMAPDimReduction import UMAPDimReduction


class AMAP:
    def __init__(self, n_components=2, anchors_method='agglomerative', n_intra_anchors=20, add_inter_class_anchors=True,
                 valid_centroids=False, k_penalty_graph=2, k=10, random_state=0, n_random_walks=10, verbose=True):
        """
        TODO ORM note that for small number of k the random walks may not be terminated
        :param n_components:
        :param anchors_method:
        :param n_intra_anchors:
        :param add_inter_class_anchors:
        :param valid_centroids:
        :param k_penalty_graph:
        :param k:
        :param random_state:
        :param verbose:
        """

        # Sanity checks
        if k_penalty_graph > k:
            raise Exception(f'k_penalty_graph={k_penalty_graph} must be less than or equal k={k}')

        # constructor members
        self.n_components = n_components
        self.anchors_method = anchors_method  # TODO validate anchors method
        self.n_intra_anchors = n_intra_anchors
        self.add_inter_class_anchors = add_inter_class_anchors  # TODO
        self.valid_centroids = valid_centroids
        self.k_penalty_graph = k_penalty_graph
        self.k = k  # note that in t-sne paper when they presented this method they used k=20 for mnist
        self.n_random_walks = n_random_walks
        self.verbose = verbose

        # Late initialized members
        self.clusters = None  # clusters within each label
        self.inter_class_anchors = None
        self.inter_class_anchors_labels = None
        self.inter_class_anchors_indices = None  # Note Assuming that if not valid centroids the centroids are concatenated
        self.intra_class_anchors = None
        self.intra_class_anchors_labels = None
        self.intra_class_anchors_indices = None  # Note Assuming that if not valid centroids the centroids are concatenated
        self.low_dim_anchors = None
        self.knng = None
        self.knng_inter_class_graph = None
        self.knng_penalty = None
        self.joint_proba = None

        np.random.seed(random_state)
        random.seed(random_state)

    def print_verbose(self, msg, verbose=False):
        if verbose:
            print(msg)
        elif self.verbose:
            print(msg)

    def fit(self, X, y):
        raise NotImplementedError()

    def fit_transform(self, X, y):
        """
        TODO
        :param X:
        :param y:
        :return:
        """
        # Get Anchors by clustering for each label
        self._get_intra_class_anchors(X, y)
        # Note that in order to perform random walks when valid_centroid is False, the computed centroids must be part
        # of the data. Therefore create an instance of X and y with the new centroids
        if not self.valid_centroids:
            X_with_centroids = np.concatenate((X, self.intra_class_anchors), axis=0)
            y_with_centroids = np.concatenate((y, self.intra_class_anchors_labels))
        else:
            X_with_centroids = X
            y_with_centroids = y

        # TODO For debug
        # self.X_with_centroids = X_with_centroids
        # self.y_with_centroids = y_with_centroids

        # Build knn Graph
        self._build_knng(X, X_with_centroids)
        # TODO ORM debug
        # # Get inter-class anchors
        # self._get_inter_class_anchors(X, y)
        # Get all indices of all anchors (landmarks in t-sne language)
        if self.valid_centroids:
            self.print_verbose(f'Num inter_class_anchors_indices={len(self.inter_class_anchors_indices)} '
                               f'Num intra_class_anchors_indices={len(self.intra_class_anchors_indices)}')
            anchors_indices = self.inter_class_anchors_indices + self.intra_class_anchors_indices
        else:
            # Assuming centroids are concatenated
            # TODO ORM remove comment
            # self.print_verbose(f'Num inter_class_anchors_indices={len(self.inter_class_anchors_indices)} '
            #                    f'Num y_with_centroids={len(y_with_centroids)}')
            anchors_indices = self.inter_class_anchors_indices if self.inter_class_anchors_indices is not None else [] + \
                              [i for i in range(len(y_with_centroids)-len(self.intra_class_anchors_labels), len(y_with_centroids))]
        self.print_verbose(f'len anchors_indices = {len(anchors_indices)}')
        # Calculate probabilities
        # TODO ORM debug
        self._calc_joint_proba(X_with_centroids, anchors_indices)
        # Calculate inter class term
        self._calc_inter_class_term()
        # Dim Reduction
        self._dim_reduction(X_with_centroids, y_with_centroids, anchors_indices)
        return self.low_dim_anchors

    def _get_intra_class_anchors(self, X, y):
        """
        TODO
        :param X:
        :param y:
        :return:
        """
        self.print_verbose(f'finding intra class anchors using {self.anchors_method}')
        if self.anchors_method == 'agglomerative':
            cluster_method = AgglomerativeClustering
        else:
            raise Exception(f'Unsupported anchors method {self.anchors_method}')

        if self.n_intra_anchors is None:
            raise NotImplementedError("Auto number of anchors is not implemented yet")

        self.print_verbose(f'Number of intra_class anchors (centroids) is {self.n_intra_anchors}')

        df = pd.DataFrame(X)
        feature_cols = df.columns
        label_col = 'y'
        cluster_col = 'cluster'
        df[label_col] = y
        for label in df[label_col].unique():
            # Clustering
            cm = cluster_method(n_clusters=self.n_intra_anchors)
            df.loc[df[label_col] == label, cluster_col] = cm.fit_predict(
                df[df[label_col] == label][feature_cols].values)

        # Save clusters for future use
        self.clusters = df[cluster_col].values
        # Find Centroids
        intra_centroids_df = df.groupby([label_col, cluster_col]).mean().reset_index()
        self.print_verbose(f'intra class centroids\n {intra_centroids_df.to_string()}')
        self.intra_class_anchors_labels = intra_centroids_df[label_col].values
        if self.valid_centroids:
            # TODO find valid centroids append them to intra centroid indices
            raise NotImplementedError("Valid centroids is not implemented yet")
        else:
            self.intra_class_anchors = intra_centroids_df[feature_cols].values
        self.print_verbose(f'intra class centroids\n {intra_centroids_df[feature_cols].to_string()}')

    def _build_knng(self, X, X_with_centroids):
        # TODO - very inefficient to compute knng twice!
        # Note that in order to perform random walks when valid_centroid is False, the computed centroids must be part
        # of the data
        self.print_verbose(f'X.shape={X.shape}, X_with_centroids.shape={X_with_centroids.shape}')
        self.knng = kneighbors_graph(X_with_centroids, self.k, mode='distance')
        # TODO ORM DBG
        # self.knng_penalty = kneighbors_graph(X, self.k_penalty_graph, mode='distance')

    def _get_inter_class_anchors(self, X, y):
        edges_x1, edges_x2 = self.knng_penalty.nonzero()
        # Initialize inter class anchors
        inter_class_anchors_indices = set()
        for x1, x2 in zip(edges_x1, edges_x2):
            # x1 is the index of sample x1 in X x2 is the index of sample x2 in X
            # x1 and x2 are neighbors in the KNNG
            # We want to keep only x1 and x2 where y[x1]!=y[x2] meaning points from different classes
            if y[x1] == y[x2]:
                # Neighbors from the same class - continue
                self.print_verbose(f'x1={x1} and x2={x2} are from the same class {y[x1]}={y[x2]}')
                continue
            # Now x1 and x2 are neighbors from different classes
            self.print_verbose(f'x1={x1} and x2={x2} are not from the same class {y[x1]}!={y[x2]}')
            inter_class_anchors_indices.add(x1)
        inter_class_anchors_indices = list(inter_class_anchors_indices)
        self.print_verbose(f'found {len(inter_class_anchors_indices)}')
        self.inter_class_anchors_labels = y[inter_class_anchors_indices]
        self.print_verbose(f'found \n{pd.Series(self.inter_class_anchors_labels).value_counts().to_string()}')
        self.inter_class_anchors = X[inter_class_anchors_indices]
        self.inter_class_anchors_indices = inter_class_anchors_indices

    def _calc_joint_proba(self, X, anchors_indices):
        """
         from t-sne paper: for each of the landmark points, we define a random walk starting at that landmark point and terminating
         as soon as it lands on another landmark point. During a random walk, the probability of choosing
         an edge emanating from node xi to node x j is proportional to e^||xi-xj||^2
         We define pj|i to be the fraction of random walks starting at landmark point xi that terminate at
         landmark point xj.
        :param X:
        :param anchors_indices:
        :return:
        """
        self.print_verbose(f'Num anchors {len(anchors_indices)}')
        self.joint_proba = np.zeros((len(anchors_indices), len(anchors_indices)))
        # note that the distance matrix is not squared
        edges_proba = -1 * self.knng.power(2)
        # note that we are using exp1m since this is sparse matrix!
        edges_proba = edges_proba.expm1()
        edges_proba = edges_proba.multiply(csr_matrix(edges_proba.sum(axis=1)).power(-1))
        self.edges_proba = edges_proba # TODO ORM debug
        # create dict of anchors indices to index in anchors_indices for better performance
        anchors_indices_dict = dict()
        for i in range(len(anchors_indices)):
            anchors_indices_dict[anchors_indices[i]] = i
        # create arange of indices for better performance
        all_points_indices = np.arange(len(X))
        # for each anchor
        for anchor_i in range(len(anchors_indices)):
            self.print_verbose(f'calculating joint probabilities of {anchor_i} using {self.n_random_walks} random walks')
            random_walk_lengths = []
            for rw_i in range(self.n_random_walks):
                # perform random walk
                curr_vertice = anchors_indices[anchor_i]
                self.print_verbose(f'starting random walk {rw_i} from {curr_vertice}')
                next_vertice = None
                walk_length = 0
                self.anchors_indices_dict = anchors_indices_dict  # TODO ORM debug
                while next_vertice not in anchors_indices_dict.keys():
                    next_vertice = np.random.choice(all_points_indices, p=edges_proba[curr_vertice].toarray()[0])
                    self.print_verbose(f'random walk: {rw_i} len {walk_length} moving from {curr_vertice} to {next_vertice} anchors{anchors_indices_dict.keys()}')
                    curr_vertice = next_vertice
                    walk_length += 1
                random_walk_lengths.append(walk_length)
                # now we have valid anchor we can increase probability
                # get the index of the vertices in anchors_indices
                next_vertice_j = anchors_indices_dict[next_vertice]
                self.joint_proba[anchor_i][next_vertice_j] += 1
                self.print_verbose(f'next_vertice {next_vertice}=> next_vertice_j{next_vertice_j}, anchor_i {anchor_i}\n'
                                   f'joint_proba[anchor_i][next_vertice_j] = {self.joint_proba[anchor_i][next_vertice_j]}')
            self.print_verbose(f'avg walk length {sum(random_walk_lengths)/len(random_walk_lengths)}, min {min(random_walk_lengths)}, max {max(random_walk_lengths)}')
        # normalize proba
        self.joint_proba = self.joint_proba / self.joint_proba.sum(axis=1, keepdims=True)

        # Based on UMAP and t-SNE the joint probabilities should be symmetric
        # Therefore we will use the symmetry formula of UMAP
        # Pij = Pi|j + Pj|i - Pi|j*Pj|i
        # TODO: for now I will use simpler form
        # Pij = (Pi|j + Pj|i)/2n
        # self.joint_proba = (self.joint_proba + self.joint_proba.T) / (2*len(self.joint_proba))
        # TODO ORM in t-sne implementation the normalization is by sum
        self.joint_proba = (self.joint_proba + self.joint_proba.T) / (np.sum(self.joint_proba))

    def _calc_inter_class_term(self):
        pass

    def _dim_reduction(self, X, y, anchors_indices):
        """
        NOTE that at this point I will use a=1 and b=1 need to read and understand implications
        since UMAP does curve fit to a and b based on the min_dist hyper-parameter
        :return:
        """
        umap = UMAPDimReduction()
        print(X[anchors_indices])
        print('')
        print(self.joint_proba)
        print(np.argwhere(np.isinf(X[anchors_indices])))
        self.low_dim_anchors = umap.fit_transform(X[anchors_indices], y[anchors_indices], self.joint_proba)





