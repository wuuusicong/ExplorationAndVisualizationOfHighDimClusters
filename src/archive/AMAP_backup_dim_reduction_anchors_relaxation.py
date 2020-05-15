import numpy as np
import pandas as pd
import random
import umap
import alphashape
import shapely
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from sklearn.manifold import MDS, TSNE
from sklearn.decomposition import PCA, LatentDirichletAllocation
from sklearn.cluster import AgglomerativeClustering, KMeans, MeanShift, estimate_bandwidth
from sklearn.neighbors import kneighbors_graph


class AMAP:
    def __init__(self, n_components=2, anchors_method='agglomerative', n_intra_anchors=20,
                 dim_reduction_algo='mds', k=20, self_relation=False, radius_q=0.9,
                 do_relaxation=True, n_iter=1000, batch_size=1, stop_criteria=0.01, loss='Linf', learning_rate=0.01,
                 random_state=None, n_jobs=None, verbose=True):
        """
        TODO ORM note that for small number of k the random walks may not be terminated
        :param n_components:
        :param anchors_method:
        :param n_intra_anchors:
        :param valid_centroids:
        :param k:
        :param random_state:
        :param verbose:
        """

        # Sanity checks

        # constructor members
        self.n_components = n_components
        self.anchors_method = anchors_method  # TODO validate anchors method
        self.n_intra_anchors = n_intra_anchors
        self.k = k  # note that in t-sne paper when they presented this method they used k=20 for mnist
        self.self_relation = self_relation
        self.radius_q = radius_q
        self.dim_reduction_algo = dim_reduction_algo
        self.do_relaxation = do_relaxation
        self.n_iter = n_iter
        self.batch_size = batch_size
        self.stop_criteria = stop_criteria
        if loss == 'Linf':
            self.loss_func = AMAP.l_inf_loss
        else:
            raise Exception(f'Unsupported loss {loss}')
        self.loss = loss
        self.learning_rate = learning_rate
        self.n_jobs = n_jobs
        self.verbose = verbose

        # Late initialized members
        self.num_clusters_each_label = []
        self.y_with_centroids = None
        self.clusters = None  # clusters within each label
        self.intra_class_anchors = None
        self.intra_class_anchors_labels = None
        self.intra_class_anchors_indices = None  # Note Assuming that if not valid centroids the centroids are concatenated
        self.low_dim_anchors = None
        self.low_dim_points = None
        self.knng = None
        self.inter_class_relations = None
        self.inter_class_relations_low_dim = None
        self.anchors_density = None
        self.anchors_radius = None
        self.losses = []  # for visualization

        if random_state is not None:
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
        TODO ORM labels are expected to be 0,1,2,...
        :param X:
        :param y:
        :return:
        """
        # Get Anchors by clustering for each label
        self._get_intra_class_anchors(X, y)
        # Note that in order to perform random walks when valid_centroid is False, the computed centroids must be part
        # of the data. Therefore create an instance of X and y with the new centroids

        X_with_centroids = np.concatenate((X, self.intra_class_anchors), axis=0)
        y_with_centroids = np.concatenate((y, self.intra_class_anchors_labels))
        # save labels for future use
        self.y_with_centroids = y_with_centroids

        # TODO For debug
        # self.X_with_centroids = X_with_centroids
        # self.y_with_centroids = y_with_centroids

        # Assuming centroids are concatenated
        anchors_indices = [i for i in range(len(y_with_centroids)-len(self.intra_class_anchors_labels), len(y_with_centroids))]

        # Build knn Graph
        if self.do_relaxation:
            self._build_knng(X)

        # Calculate inter class relations
        if self.do_relaxation:
            self._calc_inter_class_relations(y)
        # Dim Reduction
        self._dim_reduction(X_with_centroids, y_with_centroids, anchors_indices)
        # DO relaxation in the low dimension
        if self.do_relaxation:
            self.relaxation()


        return self.low_dim_anchors # TODO ORM is this the variable that we want to return?

    def _get_intra_class_anchors(self, X, y):
        """
        TODO
        :param X:
        :param y:
        :return:
        """
        self.print_verbose(f'finding intra class anchors using {self.anchors_method}')
        if self.anchors_method == 'agglomerative':
            cm = AgglomerativeClustering(n_clusters=self.n_intra_anchors)
        elif self.anchors_method == 'kmeans':
            cm = KMeans(n_clusters=self.n_intra_anchors)
        elif self.anchors_method == 'mean_shift':
            # bandwidth = estimate_bandwidth(X, quantile=0.1, n_samples=500)
            # cm = MeanShift(bandwidth=2, bin_seeding=True, cluster_all=False)
            cm = MeanShift(bin_seeding=True, cluster_all=False)  # TODO ORM consider changing to clusterall=False and bin_seeding=True
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
        df[cluster_col] = -1
        for label in sorted(df[label_col].unique()):  # Assumed to be integers 0,1,2,...
            # Clustering
            if self.anchors_method == 'mean_shift':
                bandwidth = estimate_bandwidth(df[df[label_col] == label][feature_cols].values, quantile=0.05)
                cm = MeanShift(bandwidth=bandwidth, bin_seeding=False, cluster_all=False)
                print(f'Bandwidth label: {label} = {bandwidth}')
            df.loc[df[label_col] == label, cluster_col] = cm.fit_predict(
                df[df[label_col] == label][feature_cols].values)
        #     self.num_clusters_each_label.append(self.n_intra_anchors)
        self.num_clusters_each_label = df.groupby(label_col).nunique()[cluster_col].sort_index().values
        self.dbg_df = df

        # Save clusters for future use
        self.clusters = df[cluster_col].values.astype(int)
        # Find Centroids
        intra_centroids_df = df.groupby([label_col, cluster_col]).mean().sort_index().reset_index()
        # self.print_verbose(f'intra class centroids\n {intra_centroids_df.to_string()}')
        self.intra_class_anchors_labels = intra_centroids_df[label_col].values
        self.intra_class_anchors = intra_centroids_df[feature_cols].values
        # self.print_verbose(f'intra class centroids\n {intra_centroids_df[feature_cols].to_string()}')
        # Calc high dim properties
        self._calc_high_dim_clusters_properties(df, label_col, cluster_col, intra_centroids_df)

    def _calc_high_dim_clusters_properties(self, df, label_col, cluster_col, intra_centroids_df):
        df = df.copy()
        intra_centroids_df = intra_centroids_df.copy()
        # Density
        self.anchors_density = df.groupby([label_col])[cluster_col].value_counts().sort_index().values
        # Radius
        # Set index to perform matrix operations per label and cluster
        intra_centroids_df = intra_centroids_df.set_index([label_col, cluster_col])
        df = df.set_index([label_col, cluster_col])
        intra_centroids_df.sort_index(inplace=True)
        df.sort_index(inplace=True)
        radius_df = (df - intra_centroids_df)
        radius_df = radius_df ** 2
        radius_df = radius_df.sum(axis=1)
        radius_df = radius_df ** (0.5)
        radius_df = radius_df.groupby(level=[0, 1]).quantile(q=self.radius_q)
        self.anchors_radius = radius_df.values

    def _build_knng(self, X):
        # Note that in order to perform random walks when valid_centroid is False, the computed centroids must be part
        # of the data
        self.knng = kneighbors_graph(X, self.k, mode='distance', n_jobs=self.n_jobs)

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

    def _sample_index_to_anchor(self, label, cluster):
        # TODO ORM problem with indices
        anchor_index = 0
        for i in range(label):
            anchor_index += self.num_clusters_each_label[i]
        anchor_index += cluster
        return anchor_index

    def anchor_to_label_cluster(self, anchor_index):
        anchors_count = 0
        for i in range(len(self.num_clusters_each_label)-1):
            if anchors_count <= anchor_index < anchors_count + self.num_clusters_each_label[i]:
                return i, anchor_index - anchors_count
            anchors_count += self.num_clusters_each_label[i]
        return len(self.num_clusters_each_label)-1, anchor_index - anchors_count


    def _calc_inter_class_relations(self, y):
        """

        :param X:
        :return:
        """
        self.inter_class_relations = np.zeros((len(self.intra_class_anchors_labels), len(self.intra_class_anchors_labels)))
        edges_x1, edges_x2 = self.knng.nonzero()
        # Initialize inter class anchors
        for x1, x2 in zip(edges_x1, edges_x2):
            anchor_x1 = self._sample_index_to_anchor(y[x1], self.clusters[x1])
            anchor_x2 = self._sample_index_to_anchor(y[x2], self.clusters[x2])
            self.inter_class_relations[anchor_x1][anchor_x2] += 1
        if not self.self_relation:
            np.fill_diagonal(self.inter_class_relations, 0)
        # Normalize for each anchor
        self.inter_class_relations = self.inter_class_relations / self.inter_class_relations.sum(axis=1, keepdims=True)

    def calc_low_dim_inter_class_relations(self, y):
        """
        TODO ORM merge with _calc_inter_class_relations
        :param X:
        :return:
        """

        # low_dim_points_arr = [self.low_dim_anchors]
        #     anchors_series = list(range(len(self.low_dim_anchors)))
        #     labels_series = [self.anchor_to_label_cluster(anchor_index)[0] for anchor_index in range(len(self.low_dim_anchors))]
        #     for anchor_index in range(len(self.low_dim_anchors)):
        #         points = self.random_points_per_cluster(anchor_index)
        #         low_dim_points_arr.append(points)
        #         anchors_series.extend([anchor_index] * len(points))
        #         labels_series.extend([self.anchor_to_label_cluster(anchor_index)[0]] * len(points))
        #     low_dim_points = np.concatenate(low_dim_points_arr)
        #     # calc knng
        #     knng = kneighbors_graph(low_dim_points, self.k, mode='distance', n_jobs=self.n_jobs)
        #     self.inter_class_relations_low_dim = np.zeros(
        #         (len(self.low_dim_anchors), len(self.low_dim_anchors)))
        #     edges_x1, edges_x2 = knng.nonzero()
        #     # Initialize inter class anchors
        #     for x1, x2 in zip(edges_x1, edges_x2):
        #         anchor_x1 = anchors_series[x1]
        #         anchor_x2 = anchors_series[x2]
        #         self.inter_class_relations_low_dim[anchor_x1][anchor_x2] += 1
        #     # TODO ORM do we want symmetry?
        #     # Normalize for each anchor
        #     self.inter_class_relations_low_dim = self.inter_class_relations_low_dim / self.inter_class_relations_low_dim.sum(axis=1, keepdims=True)
        #     if not self.self_relation:
        #         np.fill_diagonal(self.inter_class_relations_low_dim, 0)

        knng = kneighbors_graph(self.low_dim_points, self.k, mode='distance', n_jobs=self.n_jobs)
        self.inter_class_relations_low_dim = np.zeros((len(self.intra_class_anchors_labels), len(self.intra_class_anchors_labels)))
        edges_x1, edges_x2 = knng.nonzero()
        # Initialize inter class anchors
        for x1, x2 in zip(edges_x1, edges_x2):
            anchor_x1 = self._sample_index_to_anchor(y[x1], self.clusters[x1])
            anchor_x2 = self._sample_index_to_anchor(y[x2], self.clusters[x2])
            self.inter_class_relations_low_dim[anchor_x1][anchor_x2] += 1
        if not self.self_relation:
            np.fill_diagonal(self.inter_class_relations_low_dim, 0)
        # Normalize for each anchor
        self.inter_class_relations_low_dim = self.inter_class_relations_low_dim / self.inter_class_relations_low_dim.sum(axis=1, keepdims=True)

    def _dim_reduction(self, X, y, anchors_indices, supervised=False):
        """
        NOTE that at this point I will use a=1 and b=1 need to read and understand implications
        since UMAP does curve fit to a and b based on the min_dist hyper-parameter
        :return:
        """
        if self.dim_reduction_algo == 't-sne':
            dim_reduction_algo_inst = TSNE()
            pass
        elif self.dim_reduction_algo == 'umap':
            dim_reduction_algo_inst = umap.UMAP()
            pass
        elif self.dim_reduction_algo == 'mds':
            dim_reduction_algo_inst = MDS(n_components=self.n_components)
            pass
        elif self.dim_reduction_algo == 'pca':
            dim_reduction_algo_inst = PCA(n_components=self.n_components)
            pass
        elif self.dim_reduction_algo == 'lda':
            dim_reduction_algo_inst = LatentDirichletAllocation(n_components=self.n_components)
            pass
        else:
            raise Exception(f'Dimension reduction algorithm {self.dim_reduction_algo} is not supported')
        # self.print_verbose(X[anchors_indices])
        # self.low_dim_anchors = dim_reduction_algo_inst.fit_transform(X[anchors_indices])
        if supervised:
            self.print_verbose('Supervised Dim Reduction')
            self.low_dim_points = dim_reduction_algo_inst.fit_transform(X, y)
        else:
            self.print_verbose('UnSupervised Dim Reduction')
            self.low_dim_points = dim_reduction_algo_inst.fit_transform(X)

    def random_points_per_cluster(self, anchor_index):
        # TODO ORM for now random points in box instead of circle
        number_of_random_points = self.anchors_density[anchor_index]
        random_points = np.zeros((number_of_random_points, self.n_components))
        minx = self.low_dim_anchors[anchor_index][0] - self.anchors_radius[anchor_index]
        maxx = self.low_dim_anchors[anchor_index][0] + self.anchors_radius[anchor_index]
        miny = self.low_dim_anchors[anchor_index][1] - self.anchors_radius[anchor_index]
        maxy = self.low_dim_anchors[anchor_index][1] + self.anchors_radius[anchor_index]
        if self.n_components > 2:
            minz = self.low_dim_anchors[anchor_index][2] - self.anchors_radius[anchor_index]
            maxz = self.low_dim_anchors[anchor_index][2] + self.anchors_radius[anchor_index]
        for i in range(number_of_random_points):
            random_points[i][0] = np.random.uniform(low=minx, high=maxx)
            random_points[i][1] = np.random.uniform(low=miny, high=maxy)
            if self.n_components > 2:
                random_points[i][2] = np.random.uniform(low=minz, high=maxz)
        return random_points

    # def calc_low_dim_inter_class_relations(self):
    #     # TODO: Note if I am using radius length from the high dimension, probably we need to use MDS
    #     # TODO: otherwise it doesn't make sense at all!
    #     # TODO: calc radius for each anchor in the high dimension once
    #     # initialize with the anchors
    #     low_dim_points_arr = [self.low_dim_anchors]
    #     anchors_series = list(range(len(self.low_dim_anchors)))
    #     labels_series = [self.anchor_to_label_cluster(anchor_index)[0] for anchor_index in range(len(self.low_dim_anchors))]
    #     for anchor_index in range(len(self.low_dim_anchors)):
    #         points = self.random_points_per_cluster(anchor_index)
    #         low_dim_points_arr.append(points)
    #         anchors_series.extend([anchor_index] * len(points))
    #         labels_series.extend([self.anchor_to_label_cluster(anchor_index)[0]] * len(points))
    #     low_dim_points = np.concatenate(low_dim_points_arr)
    #     # calc knng
    #     knng = kneighbors_graph(low_dim_points, self.k, mode='distance', n_jobs=self.n_jobs)
    #     self.inter_class_relations_low_dim = np.zeros(
    #         (len(self.low_dim_anchors), len(self.low_dim_anchors)))
    #     edges_x1, edges_x2 = knng.nonzero()
    #     # Initialize inter class anchors
    #     for x1, x2 in zip(edges_x1, edges_x2):
    #         anchor_x1 = anchors_series[x1]
    #         anchor_x2 = anchors_series[x2]
    #         self.inter_class_relations_low_dim[anchor_x1][anchor_x2] += 1
    #     # TODO ORM do we want symmetry?
    #     # Normalize for each anchor
    #     self.inter_class_relations_low_dim = self.inter_class_relations_low_dim / self.inter_class_relations_low_dim.sum(axis=1, keepdims=True)
    #     if not self.self_relation:
    #         np.fill_diagonal(self.inter_class_relations_low_dim, 0)

    @staticmethod
    def l_inf_loss(X, Y):
        return np.absolute(X-Y).max()

    def get_worst_inter_class_anchors(self):
        diff_mat = self.inter_class_relations - self.inter_class_relations_low_dim
        inter_class_relation_loss = np.absolute(diff_mat)
        src_anchor_index, target_anchor_index = np.unravel_index(inter_class_relation_loss.argmax(),
                                                                 inter_class_relation_loss.shape)
        # if diff is positive it means that the inter relation in the high dimension is higher
        # which means that these anchors should be closer
        # otherwise they should be distant
        if diff_mat[src_anchor_index][target_anchor_index] > 0:
            direction = 1
        else:
            direction = -1
        return src_anchor_index, target_anchor_index, direction

    def relax_anchor(self, src_anchor, target_anchor, direction):
        direction_vec = target_anchor - src_anchor
        return src_anchor + direction * self.learning_rate * direction_vec

    def relaxation(self):
        for i in range(self.n_iter):
            self.calc_low_dim_inter_class_relations()
            loss = self.loss_func(self.inter_class_relations, self.inter_class_relations_low_dim)
            if loss < self.stop_criteria:
                self.print_verbose(f'loss {loss} < stopping criteria {self.stop_criteria} nothing to do')
                return
            self.losses.append(loss)
            self.print_verbose(f'Starting iteration {i+1} loss = {loss}')
            for j in range(self.batch_size):
                src_anchor_index, target_anchor_index, direction = self.get_worst_inter_class_anchors()
                self.print_verbose(f'Updating anchors {src_anchor_index} => {target_anchor_index} in direction {direction}')
                new_src_anchor = self.relax_anchor(self.low_dim_anchors[src_anchor_index],
                                                   self.low_dim_anchors[target_anchor_index],
                                                   direction)
                # update src anchor
                self.low_dim_anchors[src_anchor_index] = new_src_anchor
            if self.verbose:
                # df = pd.DataFrame(data=self.low_dim_anchors, columns=['x', 'y'])
                # df['label'] = self.intra_class_anchors_labels
                df = pd.DataFrame(data=self.low_dim_points, columns=['x', 'y'])
                df['label'] = self.y_with_centroids
                AMAP.vis_2d(df, 'x', 'y', 'label', f'iter{i}', f'relaxation_images/iter{i}.png')

    @staticmethod
    def vis_2d(_df, _x, _y, _color, _title='', file_name=''):
        fig, ax = plt.subplots(figsize=(10,10))
        _df['legend'] = _df[_color].transform(lambda x: f'label_{x}')
        sns.scatterplot(data=_df, x=_x, y=_y, hue='legend', ax=ax)
        ax.set_title(_title)
        ax.set_xlim((-0.2, 0.2))
        ax.set_ylim((-0.2, 0.2))
        # fig = go.Figure()
        # color = iter(ClusterPlot.DEFAULT_PLOTLY_COLORS)
        # for y in sorted(_df[_y].unique()):
        #     c = next(color)
        #     fig.add_trace(go.Scatter(x=_df[_df[_y] == y][_x].values, y=_df[_df[_y]==y][_y].values, mode='markers',
        #                              color=c))
        # # fig.show()
        # # fig = px.scatter(_df, x=_x, y=_y, color=_color)
        #
        # fig.update_layout(title=f'{_title}: 2D Visualization')
        if file_name == '':
            fig.show()
        else:
            # fig.write_image(file_name)
            fig.savefig(file_name)
        plt.close()

    DEFAULT_PLOTLY_COLORS = ['rgb(31, 119, 180)', 'rgb(255, 127, 14)',
                             'rgb(44, 160, 44)', 'rgb(214, 39, 40)',
                             'rgb(148, 103, 189)', 'rgb(140, 86, 75)',
                             'rgb(227, 119, 194)', 'rgb(127, 127, 127)',
                             'rgb(188, 189, 34)', 'rgb(23, 190, 207)']

    def anchors_to_contour(self):
        # TODO ORM DBG
        # self.anchors_radius = 0.3
        #######################
        x_plus_arr = self.low_dim_anchors.copy()
        x_plus_arr[:, 0] = self.low_dim_anchors[:, 0] + self.anchors_radius
        x_minus_arr = self.low_dim_anchors.copy()
        x_minus_arr[:, 0] = self.low_dim_anchors[:, 0] - self.anchors_radius
        y_plus_arr = self.low_dim_anchors.copy()
        y_plus_arr[:, 1] = self.low_dim_anchors[:, 1] + self.anchors_radius
        y_minus_arr = self.low_dim_anchors.copy()
        y_minus_arr[:, 1] = self.low_dim_anchors[:, 1] - self.anchors_radius
        if self.n_components > 2:
            z_plus_arr = self.low_dim_anchors.copy()
            z_plus_arr[:, 2] = self.low_dim_anchors[:, 2] + self.anchors_radius
            z_minus_arr = self.low_dim_anchors.copy()
            z_minus_arr[:, 2] = self.low_dim_anchors[:, 2] - self.anchors_radius
        anchors_radius = np.concatenate([x_plus_arr, x_minus_arr, y_plus_arr, y_minus_arr])
        if self.n_components > 2:
            anchors_radius = np.concatenate([anchors_radius, z_plus_arr, z_minus_arr])
        n_points_per_anchor = self.n_components * 2
        labels = []
        for i in range(n_points_per_anchor):
            labels.extend(self.intra_class_anchors_labels)

        anchors_df = pd.DataFrame(anchors_radius, columns=['x', 'y', 'z'] if self.n_components > 2 else ['x', 'y'])
        anchors_df['label'] = labels
        return anchors_df

    @staticmethod
    def smooth_poly_Chaikins_corner_cutting_iter(poly, iter=1):
        new_poly = poly[:]
        for i in range(iter):
            new_poly = AMAP.smooth_poly_Chaikins_corner_cutting(new_poly, True)
        return new_poly

    @staticmethod
    def smooth_poly_Chaikins_corner_cutting(poly, append_first_point):
        """
        poly is list of lists
        example: poly1 = [
        [3,3],
        [4,4],
        [5,4],
        [5,7],
        [6,8],
        [7,5],
        [6,3],
        [5,2],
        [4,2],
        [3,3]
        ]
        Based on https://stackoverflow.com/questions/27642237/smoothing-a-2-d-figure
        Q(i) = (3/4)P(i) + (1/4)P(i+1)
        R(i) = (1/4)P(i) + (3/4)P(i+1)
        """
        new_poly = []
        for i in range(len(poly) - 1):
            q_i = [0.75 * poly[i][0] + 0.25 * poly[i + 1][0], 0.75 * poly[i][1] + 0.25 * poly[i + 1][1]]
            r_i = [0.25 * poly[i][0] + 0.75 * poly[i + 1][0], 0.25 * poly[i][1] + 0.75 * poly[i + 1][1]]
            new_poly.extend([q_i, r_i])
        # append first point for smoothness
        if append_first_point:
            new_poly.append(new_poly[0])
        return new_poly

    def get_concave_hull(self, points, alpha):
        alpha_shape = alphashape.alphashape(points.tolist(), alpha)
        smooth_shapes = []
        if isinstance(alpha_shape, shapely.geometry.polygon.Polygon):
            alpha_shape = [alpha_shape]
        else:  # Multipolygon
            alpha_shape = list(alpha_shape)
        for shape in list(alpha_shape):
            x, y = shape.exterior.coords.xy
            smooth_shape = np.array(self.smooth_poly_Chaikins_corner_cutting_iter(list(zip(x, y)), 3))
            smooth_shapes.append(smooth_shape)
        return smooth_shapes

    def anchors_plot(self, alpha):
        # TODO ORM need to support n_components
        # TODO ORM replace strings col
        color = iter(self.DEFAULT_PLOTLY_COLORS)
        anchors_agg_df_ = pd.DataFrame(data=self.low_dim_anchors, columns=['x', 'y'])
        anchors_agg_df_['label'] = self.intra_class_anchors_labels
        contours_df = self.anchors_to_contour()
        fig = go.Figure()
        for label in sorted(anchors_agg_df_['label'].unique()):
            points = contours_df[contours_df['label'] == label][['x', 'y']].values
            concave_hulls = self.get_concave_hull(points, alpha)

            anchors_tmp = anchors_agg_df_[anchors_agg_df_['label'] == label][['x', 'y']].values
            c = next(color)
            fig.add_trace(go.Scatter(x=anchors_tmp[:, 0], y=anchors_tmp[:, 1],
                                     mode='markers',
                                     marker_color=c,
                                     name=f'digit_{label}'))
            for concave_hull in concave_hulls:
                fig.add_trace(go.Scatter(x=concave_hull[:, 0],
                                         y=concave_hull[:, 1],
                                         fill='toself',
                                         marker_color=c,
                                         name=f'digit_{label}'))
        fig.show()






