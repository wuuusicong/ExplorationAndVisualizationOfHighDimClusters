import os
import numpy as np
import pandas as pd
import random
import umap
import alphashape
import shapely
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import imageio
from datetime import datetime
from matplotlib import patches
from matplotlib.path import Path
from plotly.subplots import make_subplots
from sklearn.manifold import MDS, TSNE
from sklearn.decomposition import PCA, LatentDirichletAllocation
from sklearn.cluster import AgglomerativeClustering, KMeans, MeanShift, estimate_bandwidth, Birch
from sklearn.neighbors import kneighbors_graph
from scipy.interpolate import splprep, splev
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from pygifsicle import optimize

# Global Configurations
sns.set_style("darkgrid")


class AMAP:
    def __init__(self, n_components: int = 2, anchors_method: str = 'birch', n_intra_anchors: int = None,
                 birch_threshold: float = 0.3, birch_branching_factor: int = None,
                 dim_reduction_algo: str = 'umap', supervised: bool = False, reduce_all_points: bool = False,
                 uniform_points_per: str = 'anchor',
                 k: int = 20, self_relation: bool = False, radius_q: float = 1, do_relaxation: bool = True,
                 top_greedy: int = 1, magnitude_step: bool = False, n_iter: int = 10, batch_size: int = 1,
                 stop_criteria: float = 0.00001, loss: str = 'mse', learning_rate: float = None, random_points_in_box: bool = False,
                 class_to_label: dict = None, random_state: int = None, n_jobs: int = None, verbose: bool = True,
                 dataset: str = 'default', show_fig: bool = True, save_fig: bool = True, is_plotly: bool = False,
                 do_animation=True, use_spline: bool = False, alpha: float = 1):
        """
        :param n_components:
        :param anchors_method:
        :param n_intra_anchors:
        :param valid_centroids:
        :param k:
        :param random_state:
        :param verbose:
        """
        # start with verbose level
        self.verbose = verbose

        # Sanity checks
        if n_components > 2:
            raise Exception(f'n_components > 2 not supported currently')

        supported_dim_reduction_algos = ['t-sne', 'umap', 'mds', 'pca', 'lda']
        if dim_reduction_algo not in supported_dim_reduction_algos:
            raise Exception(f'Unsupported dim_reduction_algo {dim_reduction_algo}, supported algos: {supported_dim_reduction_algos}')

        supported_anchors_methods = ['agglomerative', 'kmeans', 'birch']
        if anchors_method not in supported_anchors_methods:
            raise Exception(f'Unsupported anchors method {anchors_method}, supported methods: {supported_anchors_methods}')

        if anchors_method in ['birch']:
            if n_intra_anchors is not None:
                self.print_verbose(f'Anchors method {anchors_method} does not require number of anchors '
                                   f'but n_intra_anchors is {n_intra_anchors}')
        else:
            if not n_intra_anchors:
                raise Exception(f'n_intra_anchors is required for {anchors_method}')

        if dim_reduction_algo not in ['umap', 'lda'] and supervised:
            raise Exception(f'dim_reduction_algo {dim_reduction_algo} does not support supervision')

        supported_uniform_points_per = ['label', 'anchor']
        if uniform_points_per not in supported_uniform_points_per:
            raise Exception(f'Unsupported uniform_points_per {uniform_points_per} , supported methods: {supported_uniform_points_per}')

        if k < 0:
            raise Exception(f'k must be greater than 0')

        if not 0 < radius_q <=1:
            raise Exception(f'radius_q must bi in (0, 1]')

        if top_greedy < 0:
            raise Exception(f'top_greedy must be greater than 0')

        if not magnitude_step and learning_rate is None:
            raise Exception('Learning rate must be defined if magnitude step is disabled')

        if n_iter < 0:
            raise Exception(f'n_iter must be greater than 0')

        if batch_size < 0:
            raise Exception(f'batch_size must be greater than 0')

        if stop_criteria < 0:
            raise Exception(f'stop_criteria must be greater than 0')

        # constructor members
        self.n_components = n_components
        self.anchors_method = anchors_method
        self.n_intra_anchors = n_intra_anchors
        self.birch_threshold = birch_threshold
        self.birch_branching_factor = birch_branching_factor
        self.dim_reduction_algo = dim_reduction_algo
        self.supervised = supervised
        self.reduce_all_points = reduce_all_points
        self.uniform_points_per = uniform_points_per
        self.k = k  # note that in t-sne paper when they presented this method they used k=20 for mnist
        self.self_relation = self_relation
        self.radius_q = radius_q
        self.do_relaxation = do_relaxation
        self.top_greedy = top_greedy
        self.magnitude_step = magnitude_step
        self.n_iter = n_iter
        self.batch_size = batch_size
        self.stop_criteria = stop_criteria
        if loss == 'Linf':
            self.loss_func = AMAP.l_inf_loss
        elif loss == 'mse':
            self.loss_func = AMAP.mse_loss
        else:
            raise Exception(f'Unsupported loss {loss}')
        self.loss = loss
        self.learning_rate = learning_rate
        self.random_points_in_box = random_points_in_box
        self.class_to_label = class_to_label
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.dataset = dataset
        self.show_fig = show_fig
        self.save_fig = save_fig
        self.is_plotly = is_plotly
        self.do_animation = do_animation
        self.use_spline = use_spline
        self.alpha = alpha

        # create output dir
        now = datetime.now()
        date_time = now.strftime("%Y-%d-%m_%H-%M-%S")
        self.output_dir = f'./plots/{self.dataset}/{date_time}/'
        if self.save_fig:
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
            with open(f'{self.output_dir}/run_info.txt', 'w+') as f:
                f.write(str(self))

        # Late initialized members
        self.num_clusters_each_label = []
        self.X_with_centroids = None
        self.y_with_centroids = None
        self.clusters = None  # clusters within each label
        self.intra_class_anchors = None
        self.intra_class_anchors_labels = None
        self.intra_class_anchors_indices = None  # Note Assuming that if not valid centroids the centroids are concatenated
        self.anchors_indices = None
        self.low_dim_anchors = None
        self.low_dim_points = None
        self.knng = None
        self.inter_class_relations = None
        self.inter_class_relations_low_dim = None
        self.anchors_density = None
        self.anchors_radius = None
        self.losses = []  # for visualization

        # assistance variable
        self.label_col = 'label'
        self.cluster_col = 'cluster'
        self.anchor_col = 'anchor'
        self.x_col = 'x'
        self.y_col = 'y'

    def __repr__(self):
        return "{klass}\n=================\naddr:@{id:x}\n{attrs}\n\n".format(
            klass=self.__class__.__name__,
            id=id(self) & 0xFFFFFF,
            attrs="\n".join("{}={!r}".format(k, v) for k, v in self.__dict__.items()),
        )

    def print_verbose(self, msg, verbose=False):
        if verbose:
            print(msg)
        elif self.verbose:
            print(msg)

    def fit(self, X, y):
        raise NotImplementedError()

    def fit_transform(self, X, y):
        """
        labels are expected to be 0,1,2,...
        :param X:
        :param y:
        :return:
        """
        # sanity check
        if not isinstance(X, np.ndarray):
            raise Exception(f'X must be an instance of numpy ndarray')

        if not isinstance(y, np.ndarray):
            raise Exception(f'y must be an instance of numpy array')

        if X.shape[0] != y.shape[0]:
            raise Exception(f'X and y are not in the same length ({X.shape[0]} != {y.shape[0]}')

        if not np.array_equal(np.unique(y), np.arange(len(np.unique(y)))):
            raise Exception(f'label values must be integers starting from 0 to last label minus 1')

        if self.random_state is not None:
            np.random.seed(self.random_state)
            random.seed(self.random_state)

        # Get Anchors by clustering for each label
        self._get_intra_class_anchors(X, y)

        self.X_with_centroids = np.concatenate((X, self.intra_class_anchors), axis=0)
        self.y_with_centroids = np.concatenate((y, self.intra_class_anchors_labels))

        # Assuming centroids are concatenated too in _get_intra_class_anchors
        self.anchors_indices = [i for i in range(len(self.y_with_centroids)-len(self.intra_class_anchors_labels),
                                            len(self.y_with_centroids))]
        # Build knn Graph
        if self.do_relaxation:
            self._build_knng(self.X_with_centroids)

        # Calculate inter class relations
        if self.do_relaxation:
            self._calc_inter_class_relations()
        # Dim Reduction
        self._dim_reduction()
        # DO relaxation in the low dimension
        if self.do_relaxation:
            self.relaxation()


        return self.low_dim_anchors

    def _get_intra_class_anchors(self, X, y):
        """
        :param X:
        :param y:
        :return:
        """
        self.print_verbose(f'finding intra class anchors using {self.anchors_method}')
        if self.anchors_method == 'agglomerative':
            cm = AgglomerativeClustering(n_clusters=self.n_intra_anchors)
        elif self.anchors_method == 'kmeans':
            cm = KMeans(n_clusters=self.n_intra_anchors)
        elif self.anchors_method == 'birch':
            if self.birch_branching_factor is None:
                # no branching factor
                branching_factor = X.shape[0]
            else:
                branching_factor = self.birch_branching_factor
            cm = Birch(threshold=self.birch_threshold, branching_factor=branching_factor, n_clusters=None)

        else:
            raise Exception(f'Unsupported anchors method {self.anchors_method}')

        if self.n_intra_anchors is None and self.anchors_method != 'birch':
            raise NotImplementedError("Auto number of anchors is not implemented yet")

        df = pd.DataFrame(X)
        feature_cols = df.columns
        df[self.label_col] = y
        df[self.cluster_col] = -1
        anchors_dfs = []
        for label in sorted(df[self.label_col].unique()):  # Assumed to be integers 0,1,2,...
            # Clustering
            df.loc[df[self.label_col] == label, self.cluster_col] = cm.fit_predict(
                df[df[self.label_col] == label][feature_cols].values)
            if self.anchors_method == 'birch':
                subcluster_centers = cm.subcluster_centers_
            else:
                subcluster_centers = df[df[self.label_col] == label].groupby([self.label_col, self.cluster_col]).mean().sort_index().values
            tmp_df = pd.DataFrame(subcluster_centers, columns=feature_cols)
            tmp_df[self.label_col] = label
            tmp_df.reset_index(inplace=True)
            tmp_df[self.cluster_col] = tmp_df['index']
            anchors_dfs.append(tmp_df)

        self.num_clusters_each_label = df.groupby(self.label_col).nunique()[self.cluster_col].sort_index().values
        self.dbg_df = df

        # Save clusters for future use
        self.clusters = df[self.cluster_col].values.astype(int)
        intra_centroids_df = pd.concat(anchors_dfs)

        self.intra_class_anchors_labels = intra_centroids_df[self.label_col].values
        self.intra_class_anchors = intra_centroids_df[feature_cols].values
        # concat clusters of centroids to clusters variable NOTE this is a strong assumption
        # The anchors and anchors labels are concatenated later also to X and y
        anchors_clusters = []
        for num_anchors in self.num_clusters_each_label:
            anchors_clusters.extend(list(range(num_anchors)))
        self.clusters = np.concatenate((self.clusters, np.asarray(anchors_clusters)))
        # Calc high dim properties
        self._calc_high_dim_clusters_properties(df, self.label_col, self.cluster_col, intra_centroids_df)

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
        anchor_index = 0
        for i in range(label):
            anchor_index += self.num_clusters_each_label[i]
        anchor_index += cluster
        return anchor_index

    def anchor_to_label_cluster(self, anchor_index, visualization=False):
        anchors_count = 0
        for i in range(len(self.num_clusters_each_label)-1):
            if anchors_count <= anchor_index < anchors_count + self.num_clusters_each_label[i]:
                if visualization and self.class_to_label is not None:
                    return self.class_to_label[i], anchor_index - anchors_count
                else:
                    return i, anchor_index - anchors_count
            anchors_count += self.num_clusters_each_label[i]
        if visualization and self.class_to_label is not None:
            return self.class_to_label[len(self.num_clusters_each_label) - 1], anchor_index - anchors_count
        else:
            return len(self.num_clusters_each_label)-1, anchor_index - anchors_count

    def _calc_inter_class_relations(self):
        """

        :param X:
        :return:
        """
        self.inter_class_relations = np.zeros((len(self.intra_class_anchors_labels), len(self.intra_class_anchors_labels)))
        edges_x1, edges_x2 = self.knng.nonzero()
        # Initialize inter class anchors
        for x1, x2 in zip(edges_x1, edges_x2):
            anchor_x1 = self._sample_index_to_anchor(self.y_with_centroids[x1], self.clusters[x1])
            anchor_x2 = self._sample_index_to_anchor(self.y_with_centroids[x2], self.clusters[x2])
            self.inter_class_relations[anchor_x1][anchor_x2] += 1
        if not self.self_relation:
            np.fill_diagonal(self.inter_class_relations, 0)
        # Normalize for each anchor
        sum_row = self.inter_class_relations.sum(axis=1, keepdims=True)
        # replace with 1 where the sum is 0 to avoid division by 0
        sum_row[sum_row == 0] = 1
        self.inter_class_relations = self.inter_class_relations / sum_row

    def calc_low_dim_inter_class_relations(self):
        """
        :param X:
        :return:
        """
        knng = kneighbors_graph(self.low_dim_points, self.k, mode='distance', n_jobs=self.n_jobs)
        self.inter_class_relations_low_dim = np.zeros((len(self.intra_class_anchors_labels), len(self.intra_class_anchors_labels)))
        edges_x1, edges_x2 = knng.nonzero()
        # Initialize inter class anchors
        for x1, x2 in zip(edges_x1, edges_x2):
            anchor_x1 = self._sample_index_to_anchor(self.y_with_centroids[x1], self.clusters[x1])
            anchor_x2 = self._sample_index_to_anchor(self.y_with_centroids[x2], self.clusters[x2])
            self.inter_class_relations_low_dim[anchor_x1][anchor_x2] += 1
        if not self.self_relation:
            np.fill_diagonal(self.inter_class_relations_low_dim, 0)

        # Normalize for each anchor
        sum_row = self.inter_class_relations_low_dim.sum(axis=1, keepdims=True)
        # replace with 1 where the sum is 0 to avoid division by 0
        sum_row[sum_row == 0] = 1
        self.inter_class_relations_low_dim = self.inter_class_relations_low_dim / sum_row

    def _dim_reduction(self):
        """
        NOTE that at this point I will use a=1 and b=1 need to read and understand implications
        since UMAP does curve fit to a and b based on the min_dist hyper-parameter
        :return:
        """
        if self.dim_reduction_algo == 't-sne':
            dim_reduction_algo_inst = TSNE()
        elif self.dim_reduction_algo == 'umap':
            dim_reduction_algo_inst = umap.UMAP(n_components=self.n_components, min_dist=1)
        elif self.dim_reduction_algo == 'mds':
            dim_reduction_algo_inst = MDS(n_components=self.n_components)
        elif self.dim_reduction_algo == 'pca':
            dim_reduction_algo_inst = PCA(n_components=self.n_components)
        elif self.dim_reduction_algo == 'lda':
            dim_reduction_algo_inst = LatentDirichletAllocation(n_components=self.n_components)
        else:
            raise Exception(f'Dimension reduction algorithm {self.dim_reduction_algo} is not supported')
        if self.supervised:
            self.print_verbose('Supervised Dim Reduction')
            if self.reduce_all_points:
                self.print_verbose('Dim Reduction all points')
                self.low_dim_points = dim_reduction_algo_inst.fit_transform(self.X_with_centroids, self.y_with_centroids)
            else:
                self.print_verbose('Dim Reduction only anchors')
                self.low_dim_anchors = dim_reduction_algo_inst.fit_transform(self.X_with_centroids[self.anchors_indices],
                                                                             self.y_with_centroids[self.anchors_indices])
        else:
            self.print_verbose('UnSupervised Dim Reduction')
            if self.reduce_all_points:
                self.print_verbose('Dim Reduction all points')
                self.low_dim_points = dim_reduction_algo_inst.fit_transform(self.X_with_centroids)
            else:
                self.print_verbose('Dim Reduction only anchors')
                self.low_dim_anchors = dim_reduction_algo_inst.fit_transform(self.X_with_centroids[self.anchors_indices])
        if self.reduce_all_points:
            self.low_dim_anchors = self.low_dim_points[self.anchors_indices]
        else:
            # generate random points for each anchor in the low dimension within radius
            self.print_verbose(f'Dim Reduction only anchors - generate random points in low dim per {self.uniform_points_per}')
            low_dim_points = [None] * self.X_with_centroids.shape[0]
            for i in range(self.X_with_centroids.shape[0]):
                if i in self.anchors_indices:
                    low_dim_points[i] = self.low_dim_anchors[self.anchors_indices.index(i)]
                else:
                    # generate random point
                    # TODO ORM doesn't support 3d
                    if self.uniform_points_per == 'anchor':  # uniform points per anchor
                        anchor_index = self._sample_index_to_anchor(self.y_with_centroids[i], self.clusters[i])
                        low_dim_points[i] = self.random_points_per_cluster(anchor_index, 1)[0]
                    elif self.uniform_points_per == 'label':
                        contours_df = self.anchors_to_contour()
                        points = contours_df[contours_df[self.label_col] == self.y_with_centroids[i]][[self.x_col, self.y_col]].values
                        concave_hulls = self.get_concave_hull(points, alpha=self.alpha, spline=self.use_spline)
                        polygons = []
                        minxs = []
                        minys = []
                        maxxs = []
                        maxys = []
                        for concave_hull in concave_hulls:
                            # create shapely objects
                            poly = Polygon(concave_hull)
                            polygons.append(poly)
                            # get bounding box of polygon
                            minx, miny, maxx, maxy = poly.bounds
                            minxs.append(minx)
                            minys.append(miny)
                            maxxs.append(maxx)
                            maxys.append(maxy)
                        minx, miny, maxx, maxy = min(minxs), min(minys), max(maxxs), max(maxys)
                        # generate random points within the bounding box
                        in_poly = False
                        while not in_poly:
                            x_cord = np.random.uniform(low=minx, high=maxx)
                            y_cord = np.random.uniform(low=miny, high=maxy)
                            for p in polygons:
                                if p.contains(Point(x_cord, y_cord)):
                                    low_dim_points[i] = np.array([x_cord, y_cord])
                                    in_poly = True
                                    break
                    else:
                        raise Exception(f'Unsupported uniform_points_per: {self.uniform_points_per}')
            self.low_dim_points = np.asarray(low_dim_points)

    def random_points_per_cluster(self, anchor_index, number_of_random_points=None):
        if number_of_random_points is not None:
            number_of_random_points = self.anchors_density[anchor_index]
        if self.random_points_in_box:
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
        else:
            # generate the points
            # TODO ORM not supporting 3d
            radius = self.anchors_radius[anchor_index]
            theta = np.random.rand(number_of_random_points) * (2 * np.pi)
            r = radius * np.sqrt(np.random.rand(number_of_random_points))
            x, y = r * np.cos(theta), r * np.sin(theta)
            x = x + self.low_dim_anchors[anchor_index][0]
            y = y + self.low_dim_anchors[anchor_index][1]
            random_points = np.array(list(zip(x, y)))
        return random_points

    @staticmethod
    def l_inf_loss(X, Y):
        return np.absolute(X-Y).max()

    @staticmethod
    def mse_loss(X, Y):
        return np.square(X - Y).mean()

    def get_top_anchors_to_relax(self):
        diff_mat = self.inter_class_relations - self.inter_class_relations_low_dim
        inter_class_relation_loss = np.absolute(diff_mat)
        # if self relation is enable there is no point to calculate the loss from anchor to itself
        # because it is impossible to relax this anchor with respect to itself
        if self.self_relation:
            np.fill_diagonal(inter_class_relation_loss, 0)
        src_anchor_indices, target_anchor_indices = np.unravel_index(np.argsort(inter_class_relation_loss, axis=None),
                                                                     inter_class_relation_loss.shape)
        # revers indices
        src_anchor_indices = src_anchor_indices[::-1]
        target_anchor_indices = target_anchor_indices[::-1]
        directions = np.full_like(src_anchor_indices, -1)
        magnitudes = inter_class_relation_loss[src_anchor_indices, target_anchor_indices]
        # if diff is positive it means that the inter relation in the high dimension is higher
        # which means that these anchors should be closer
        # otherwise they should be distant
        directions[diff_mat[src_anchor_indices, target_anchor_indices] > 0] = 1
        return src_anchor_indices[:self.top_greedy], target_anchor_indices[:self.top_greedy], directions[:self.top_greedy], magnitudes[:self.top_greedy]

    def relax_anchor_cluster(self, src_anchor_index, target_anchor_index, direction, magnitude):
        src_anchor, target_anchor = self.low_dim_anchors[src_anchor_index], self.low_dim_anchors[target_anchor_index]
        direction_vec = target_anchor - src_anchor
        # update low dim points of all points in the same label cluster
        label, cluster = self.anchor_to_label_cluster(src_anchor_index)
        label_indices = np.argwhere(self.y_with_centroids == label)
        cluster_indices = np.argwhere(self.clusters == cluster)
        label_cluster_indices = np.intersect1d(label_indices, cluster_indices)
        if not self.magnitude_step:
            magnitude = 1
        if self.learning_rate is not None:
            magnitude = magnitude * self.learning_rate
        self.low_dim_points[label_cluster_indices] = self.low_dim_points[label_cluster_indices] + direction * magnitude * direction_vec
        # update also the low dim anchors
        self.low_dim_anchors[src_anchor_index] = self.low_dim_anchors[src_anchor_index] + direction * magnitude * direction_vec

    def relaxation(self):
        for i in range(self.n_iter):
            self.calc_low_dim_inter_class_relations()
            loss = self.loss_func(self.inter_class_relations, self.inter_class_relations_low_dim)
            if loss < self.stop_criteria:
                self.print_verbose(f'loss {loss} < stopping criteria {self.stop_criteria} nothing to do')
                return
            self.losses.append(loss)
            self.print_verbose(f'Starting iteration {i+1} loss = {loss}')
            is_first = True
            for j in range(self.batch_size):
                src_anchor_indices, target_anchor_indices, directions, magnitudes = self.get_top_anchors_to_relax()
                for ra_i in range(len(src_anchor_indices)):
                    self.relax_anchor_cluster(src_anchor_indices[ra_i], target_anchor_indices[ra_i], directions[ra_i],
                                              magnitudes[ra_i])

            if self.is_plotly:
                self.anchors_plot_plotly(i)
            else:
                self.anchors_plot_sns(i, is_first)

        if self.do_animation:
            gif_path = f'{self.output_dir}/animation.gif'
            with imageio.get_writer(gif_path, mode='I') as writer:
                for i in range(self.n_iter):
                    for j in range(5):
                        writer.append_data(imageio.imread(f'{self.output_dir}/iter{i}.png'))
                        import time
            # optimize(gif_path)

    def anchors_plot_sns(self, i, is_first):
        df = pd.DataFrame(data=self.low_dim_points, columns=[self.x_col, self.y_col])
        df[self.label_col] = self.y_with_centroids
        df[self.cluster_col] = self.clusters
        df[self.anchor_col] = False
        df.loc[df.index.isin(self.anchors_indices), self.anchor_col] = True
        fig = plt.figure(figsize=(50, 25))

        # low dim plot
        shape = (4, 8)
        df[self.label_col] = df[self.label_col].transform(lambda x: f'label_{x}')

        self.points_anchors_patches_plot(shape, fig, df, i, is_first)

        # loss
        self.loss_plot(shape, fig)

        # loss squared diff heatmap plot
        self.loss_heatmap_plot(shape, fig)

        # low dim relation
        self.low_dim_relations_plot(shape, fig)

        # high dim relations
        self.high_dim_relations_plot(shape, fig)

        fig.suptitle(f'{self.dim_reduction_algo}_{self.anchors_method}')
        if self.save_fig:
            fig.savefig(f'{self.output_dir}/iter{i}.png')
        if self.show_fig:
            plt.show()
        plt.close()

    def high_dim_relations_plot(self, shape, fig):
        ax = plt.subplot2grid(shape, (2, 6), colspan=2, rowspan=2)
        sns.heatmap(self.inter_class_relations, ax=ax, annot=False, square=True, cmap='Blues',
                    vmin=0, vmax=1,
                    xticklabels=[str(self.anchor_to_label_cluster(i, visualization=True)) for i in
                                 range(self.inter_class_relations.shape[0])],
                    yticklabels=[str(self.anchor_to_label_cluster(i, visualization=True)) for i in
                                 range(self.inter_class_relations.shape[0])])
        ax.set_title('High-Dim Relations')

        if self.save_fig:
            # Save just the portion _inside_ the second axis's boundaries
            extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            fig.savefig(f'{self.output_dir}/high_dim_relations_plot.png', bbox_inches=extent.expanded(1.3, 1.3))

    def low_dim_relations_plot(self, shape, fig):
        ax = plt.subplot2grid(shape, (0, 6), colspan=2, rowspan=2)
        sns.heatmap(self.inter_class_relations_low_dim, ax=ax, annot=False, square=True, cmap='Blues',
                    vmin=0, vmax=1,
                    xticklabels=[str(self.anchor_to_label_cluster(i, visualization=True)) for i in
                                 range(self.inter_class_relations.shape[0])],
                    yticklabels=[str(self.anchor_to_label_cluster(i, visualization=True)) for i in
                                 range(self.inter_class_relations.shape[0])])
        ax.set_title('Low-Dim Relations')

        if self.save_fig:
            # Save just the portion _inside_ the second axis's boundaries
            extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            fig.savefig(f'{self.output_dir}/low_dim_relations_plot.png', bbox_inches=extent.expanded(1.3, 1.3))

    def loss_plot(self, shape, fig):
        ax = plt.subplot2grid(shape, (0, 0), colspan=2, rowspan=2)
        sns.lineplot(x=list(range(len(self.losses))), y=self.losses, ax=ax)
        ax.set_xlim((0, self.n_iter + 3))
        ax.set_ylim((-0.1, 6.1))
        ax.set_title('Loss')

        if self.save_fig:
            # Save just the portion _inside_ the second axis's boundaries
            extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            fig.savefig(f'{self.output_dir}/loss_plot.png', bbox_inches=extent.expanded(1.3, 1.3))

    def loss_heatmap_plot(self, shape, fig):
        # Squared Diff Relations
        ax = plt.subplot2grid(shape, (2, 0), colspan=2, rowspan=2)
        a_min = 0
        a_max = 0.5
        sns.heatmap(np.clip(np.square(self.inter_class_relations - self.inter_class_relations_low_dim), a_min=a_min,
                            a_max=a_max),
                    ax=ax, annot=False, square=True, cmap='Blues',
                    vmin=a_min, vmax=a_max,
                    xticklabels=[str(self.anchor_to_label_cluster(i, visualization=True)) for i in
                                 range(self.inter_class_relations.shape[0])],
                    yticklabels=[str(self.anchor_to_label_cluster(i, visualization=True)) for i in
                                 range(self.inter_class_relations.shape[0])])
        ax.set_title('Squared Diff Relations')

        if self.save_fig:
            # Save just the portion _inside_ the second axis's boundaries
            extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            fig.savefig(f'{self.output_dir}/loss_heatmap_plot.png', bbox_inches=extent.expanded(1.3, 1.3))

    def points_anchors_patches_plot(self, shape, fig, df, i, is_first):
        ax = plt.subplot2grid(shape, (0, 2), colspan=4, rowspan=4)
        filled_markers = ('o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X',
                          'o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X',
                          'o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X',
                          'o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X',
                          'o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X',
                          'o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X',
                          'o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X',
                          'o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X',
                          'o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X',
                          'o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X')
        # Plot points
        sns.scatterplot(data=df[df['anchor'] == False], x=self.x_col, y=self.y_col, hue=self.label_col,
                        style=self.cluster_col, ax=ax,
                        alpha=0.2, legend=False, markers=filled_markers)
        # Plot anchors
        sns.scatterplot(data=df[df['anchor'] == True], x=self.x_col, y=self.y_col, hue=self.label_col,
                        style=self.cluster_col, ax=ax,
                        alpha=1, markers=filled_markers)
        # For each anchor, we add a text above the anchor
        for j, anchor_i in enumerate(self.anchors_indices):
            ax.text(self.low_dim_points[anchor_i][0], self.low_dim_points[anchor_i][1],
                    f'{str(self.anchor_to_label_cluster(j, visualization=True))}',
                    horizontalalignment='center', size='medium',
                    color='black', weight='semibold')
        ax.set_title(f'iter{i}')
        if is_first:
            xlim = (self.low_dim_points[:, 0].min() - 0.1, self.low_dim_points[:, 0].max() + 0.1)
            ylim = (self.low_dim_points[:, 1].min() - 0.1, self.low_dim_points[:, 1].max() + 0.1)
        # Add patches
        contours_df = self.anchors_to_contour()
        import itertools
        palette = itertools.cycle(sns.color_palette())
        for label in sorted(contours_df[self.label_col].unique()):
            points = contours_df[contours_df[self.label_col] == label][[self.x_col, self.y_col]].values
            concave_hulls = self.get_concave_hull(points, alpha=self.alpha, spline=self.use_spline)
            c = next(palette)
            c_darker = c[0], 1 - 1.6 * (1 - c[1]), c[2]
            for concave_hull in concave_hulls:
                coords = concave_hull
                line_cmde = [Path.MOVETO] + [Path.LINETO] * (len(coords) - 2) + [Path.CLOSEPOLY]
                path = Path(coords, line_cmde)
                patch = patches.PathPatch(path, facecolor=c, alpha=0.5, linewidth=5, edgecolor=c_darker)
                ax.add_patch(patch)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        # Modify legend
        num_labels = contours_df[self.label_col].nunique()
        current_handles, current_labels = plt.gca().get_legend_handles_labels()
        plt.legend(current_handles[:num_labels + 1], current_labels[:num_labels + 1])

        if self.save_fig:
            # Save just the portion _inside_ the second axis's boundaries
            extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            fig.savefig(f'{self.output_dir}/points_anchors_patches_plot.png', bbox_inches=extent.expanded(1.3, 1.3))

    DEFAULT_PLOTLY_COLORS = ['rgb(31, 119, 180)', 'rgb(255, 127, 14)',
                             'rgb(44, 160, 44)', 'rgb(214, 39, 40)',
                             'rgb(148, 103, 189)', 'rgb(140, 86, 75)',
                             'rgb(227, 119, 194)', 'rgb(127, 127, 127)',
                             'rgb(188, 189, 34)', 'rgb(23, 190, 207)']

    def anchors_to_contour(self):
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

        anchors_df = pd.DataFrame(anchors_radius, columns=[self.x_col, self.y_col, 'z'] if self.n_components > 2 else [self.x_col, self.y_col])
        anchors_df[self.label_col] = labels
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

    @staticmethod
    def get_concave_hull(points, alpha, spline=False):
        alpha_shape = alphashape.alphashape(points.tolist(), alpha)
        smooth_shapes = []
        if isinstance(alpha_shape, shapely.geometry.polygon.Polygon):
            alpha_shape = [alpha_shape]
        else:  # Multipolygon
            alpha_shape = list(alpha_shape)
        for shape in list(alpha_shape):
            x, y = shape.exterior.coords.xy
            if not spline:
                smooth_shape = np.array(AMAP.smooth_poly_Chaikins_corner_cutting_iter(list(zip(x, y)), 3))
            else:
                tck, u = splprep([np.array(x), np.array(y)], s=3)
                new_points = splev(u, tck)
                x, y = new_points[0], new_points[1]
                smooth_shape = np.array(list(zip(x,y)))
            smooth_shapes.append(smooth_shape)
        return smooth_shapes

    def anchors_plot_plotly(self, i):
        # TODO ORM need to support n_components
        anchors_agg_df_ = pd.DataFrame(data=self.low_dim_points[self.anchors_indices], columns=[self.x_col, self.y_col])
        anchors_agg_df_[self.label_col] = self.y_with_centroids[self.anchors_indices]

        color = iter(self.DEFAULT_PLOTLY_COLORS)
        contours_df = self.anchors_to_contour()
        fig = make_subplots(
            rows=2, cols=2,
            specs=[[{}, {}],
                   [{"colspan": 2}, None]],
            subplot_titles=("First Subplot", "Second Subplot", "Third Subplot"))

        fig.add_trace(go.Heatmap(x=[str(self.anchor_to_label_cluster(i, visualization=True)) for i in
                                    range(self.inter_class_relations.shape[0])],
                                 y=[str(self.anchor_to_label_cluster(i, visualization=True)) for i in
                                    range(self.inter_class_relations.shape[0])],
                                 z=self.inter_class_relations),
                      row=1, col=1)
        fig.add_trace(go.Heatmap(x=[str(self.anchor_to_label_cluster(i, visualization=True)) for i in
                                    range(self.inter_class_relations.shape[0])],
                                 y=[str(self.anchor_to_label_cluster(i, visualization=True)) for i in
                                    range(self.inter_class_relations.shape[0])],
                                 z=self.inter_class_relations_low_dim),
                      row=1, col=2)

        for label in sorted(anchors_agg_df_[self.label_col].unique()):
            points = contours_df[contours_df[self.label_col] == label][[self.x_col, self.y_col]].values
            concave_hulls = self.get_concave_hull(points, alpha=self.alpha, spline=self.use_spline)

            anchors_tmp = anchors_agg_df_[anchors_agg_df_[self.label_col] == label][[self.x_col, self.y_col]].values
            c = next(color)
            fig.add_trace(go.Scatter(x=anchors_tmp[:, 0], y=anchors_tmp[:, 1],
                                     mode='markers',
                                     marker_color=c,
                                     name=f'digit_{label}'),
                          row=2, col=1)
            for concave_hull in concave_hulls:
                fig.add_trace(go.Scatter(x=concave_hull[:, 0],
                                         y=concave_hull[:, 1],
                                         fill='toself',
                                         marker_color=c,
                                         name=f'digit_{label}'),
                              row=2, col=1)
        fig.update_layout(height=700)
        if self.save_fig:
            fig.write_image(f'{self.output_dir}/iter{i}.png')
        if self.show_fig:
            fig.show()
