# TODO Split library code from code that uses the including utils functions and classes
# TODO Examples only 3D and MNIST without sampling
# TODO Readme
# TODO Auto choose birch threshold
# TODO Auto choose LOF
# TODO pytest

import os
import logging
import itertools
import numpy as np
import pandas as pd
import random
import umap
import alphashape
import shapely
import seaborn as sns
import imageio
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines
from datetime import datetime
from matplotlib.transforms import Bbox, TransformedBbox
from matplotlib.legend_handler import HandlerBase
from matplotlib.image import BboxImage
from matplotlib import patches
from matplotlib.path import Path
from sklearn.manifold import MDS, TSNE
from sklearn.decomposition import PCA, LatentDirichletAllocation
from sklearn.cluster import AgglomerativeClustering, KMeans, Birch
from sklearn.neighbors import kneighbors_graph
from scipy.interpolate import splprep, splev
from shapely.geometry import Point
from shapely.geometry import Point, Polygon, MultiPolygon
from scipy.spatial import Voronoi
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import LocalOutlierFactor
from sklearn.neighbors import KNeighborsClassifier


class HandlerLineImage(HandlerBase):

    def __init__(self, path, space=15, offset=10):
        self.space = space
        self.offset = offset
        self.image_data = plt.imread(path)
        super(HandlerLineImage, self).__init__()

    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        line = matplotlib.lines.Line2D([xdescent + self.offset, xdescent + (width - self.space) / 3. + self.offset],
                                       [ydescent + height / 2., ydescent + height / 2.])
        line.update_from(orig_handle)
        line.set_clip_on(False)
        line.set_transform(trans)

        height = height * 1.6
        bb = Bbox.from_bounds(xdescent + (width + self.space) / 3. + self.offset,
                              ydescent,
                              height * self.image_data.shape[1] / self.image_data.shape[0],
                              height)

        tbb = TransformedBbox(bb, trans)
        image = BboxImage(tbb)
        image.set_data(self.image_data)

        self.update_prop(image, orig_handle, legend)
        return [line, image]


def check_greater_than_zero(arg, arg_name):
    if arg < 0:
        raise Exception(f'{arg_name} must be greater than 0')


class ClusterPlot:
    def __init__(self, n_components: int = 2, anchors_method: str = 'birch', n_intra_anchors: int = None,
                 birch_threshold: float = 0.3, birch_branching_factor: int = None,
                 dim_reduction_algo: str = 'umap', supervised: bool = False, umap_n_neighbors: int = 15,
                 umap_min_dist: int = 1, tsne_perplexity: float = 30.0, reduce_all_points: bool = False,
                 uniform_points_per: str = 'anchor',
                 k: int = 20, proximity_k: int = 3, self_relation: bool = False, radius_q: float = None,
                 do_relaxation: bool = True,
                 top_greedy: int = 1, magnitude_step: bool = False, n_iter: int = 10, batch_size: int = 1,
                 stop_criteria: float = 0.00001, loss: str = 'Linf', only_inter_relations: bool = False,
                 learning_rate: float = None, mask_sparse_subcluster: int = None, random_points_method: str = 'voronoi',
                 class_to_label: dict = None, random_state: int = None, n_jobs: int = None, verbose: int = logging.INFO,
                 dataset: str = 'default', show_fig: bool = True, save_fig: bool = True, figsize: tuple = (21, 15),
                 do_animation=False, use_spline: bool = False, alpha: float = None, remove_outliers_k: float = None,
                 douglas_peucker_tolerance: float = 0.6,
                 smooth_iter: int = 3, skip_polygons_with_area: float = 0.01,
                 mask_relation_in_same_label: bool = True, save_fig_every: int = 1,
                 show_anchors: bool = False, show_points: bool = False, show_blobs: bool = True,
                 show_inner_blobs: bool = False,
                 show_label_level_plots: bool = True, show_anchor_level_plot: bool = False, show_loss_plot: bool = True,
                 every_matrix_in_single_plot: bool = True, default_max_val: float = 0.1,
                 mat_figsize: tuple = (14, 10), mat_annot_kws_size: int = 35, mat_label_size: int = 25,
                 main_plot_fig_size: tuple = (26, 13), show_anchors_annotation: bool = False,
                 vmax_overlap: float = 0.5, vmax_proximity: float = 0.5,
                 annotate_images: bool = True, k_annot_clf: int = 40, orig_images: list = None,
                 transpose_low_dim: bool = False):
        """

        :param n_components: number of components reduce dimension to. Currently only 2 is supported due to voronoi
        :param anchors_method: Algorithm to use for finding anchors, the recomendation is to use birch, supported types
        ['agglomerative', 'kmeans', 'birch']
        :param n_intra_anchors: number of anchors to find, ignored if anchors_method is birch
        :param birch_threshold: birch threshold ignored if anchors_method is not birch
        :param birch_branching_factor: birch_branching_factor ignored if anchors_method is not birch
        :param dim_reduction_algo: which dimensionality reduction to use, defule UMAP, supported algos: 
        ['t-sne', 'umap', 'mds', 'pca', 'lda']
        :param supervised: Whether to use supervised dimensionality reduction, available only for umap and 
        lda dim_reduction_algo
        :param umap_n_neighbors: number of neighbors to use in umap, ignored if dim_reduction_algo is not umap
        :param umap_min_dist: umap_min_dist to use in umap, ignored if dim_reduction_algo is not umap
        :param reduce_all_points: whether to reduce dimension of all the points or just the anchors, use this paramter
        to compare the dim reduction algo to clusters-plot
        :param uniform_points_per: 'anchor' or 'label' how to generate random points in 2D in case that
        reduce_all_points is False.
        'label' will generate points in the entire bounding polygon of the class. 'anchor' depends on the flag
        random_points_method.
        :param k: number of neighbors to use for the k nearest neighbors graph for measuring the overlap between
        sub-clusters
        :param proximity_k: number of neighbors to use for the k nearest neighbors graph for measuring the proximity
        between sub-clusters
        :param self_relation: allow edges between sub-cluster to itself or not
        :param radius_q: NOT RECOMENDED TO USE, which quantile to take from the radius while using random_points_method
        'radius'
        :param do_relaxation: Whether to do relaxation or not
        :param top_greedy: How many anchors to relax in each iteration
        :param magnitude_step: Whether to do relaxation in steps corresponding to loss value or fixed value
        induced by the learning rate
        :param n_iter: number of relaxation iterations
        :param batch_size: how many steps of relaxation to do each iteration
        :param stop_criteria: stop criteria on the loss function
        :param loss: loss function for relaxation, supported types: 'Linf', 'mse'
        :param only_inter_relations: NOT RECOMENDED TO USE, keep only edges between different labels in the overlap
        graph. This flag zero out the diagonal on the overlap matrix before the normaliation. it is recommended to use
        mask_relation_in_same_label instead of this flag which mask the diagonal after the normalization.
        :param learning_rate: learning rate for the relaxation, steps will be taken in relative learning rate of the
        distance between the source and the target anchor
        :param mask_sparse_subcluster: Skip relaxation on subclusters with lower than mask_sparse_subcluster points.
        The logic is to ignore sub-clusters which are actually outliers and not interesting in the optimization
        :param random_points_method: in case that reduce_all_points is False and uniform_points_per is 'anchor',
        how to generate random points in 2D.
        supported types: ['box', 'radius', 'voronoi']. 'box' generates random points uniformly in the bounding box of
        the sub-cluster in 2D based on the radius of the sub-cluster measured in the high-dim,
        'radius' generates points randomly in the radius of each anchor as measured in the high dimension,
        'voronoi' generates random points uniformly in the voronoi diagram induces by each anchor in 2D.
        it is not recommended to use different option rather than 'voronoi'
        :param class_to_label: dict mapping from class number to label for visualization purposes
        :param random_state: seed for repruducability
        :param n_jobs: number of jobs to use, not suported for all flow
        :param verbose: self.logger level as defined in self.logger module
        :param dataset: directory name corresponding with the dataset, this folder will be the output directory for the
        plots and info
        :param show_fig: show main figure or not
        :param save_fig: save main figure or not
        :param figsize: tuple figure size of main plot
        :param do_animation: make animation gif of the relaxation
        :param use_spline: use spline to find the bounding polygon - NOT RECOMMENDED
        :param alpha: alpha for alpha shape algorithm. it is used to find the concave hull of each class in 2D.
        this parameter can be a list with alpha for every class separately or float which will be used for all the
        classes. 0 is convec hull, increasing the alpha will result in more tight counding polygon.
        :param douglas_peucker_tolerance: - tolerance for the douglas peucker algorithm used for smoothing the bounding
        polygon of each class
        :param smooth_iter: number of smoothing iterations to use for smoothing the polygons using
        Chaikins_corner_cutting algorithm
        :param skip_polygons_with_area: do not print small polygons with area smaller than skip_polygons_with_area
        :param mask_relation_in_same_label: mask overlap between sub-cluster of same label. it is recommended to use
        this flag and not only_inter_relations
        :param save_fig_every: save figure every save_fig_every iterations
        :param show_anchors: show the anchors in the 2D plot or not
        :param show_points: show the points in the 2D plot or not. use this flag for benchmarking and comparison of
        clusters-plot to other methods
        :param show_blobs: show the polygons in the 2D plot, set to False for benchmarking and comparison of
        clusters-plot to other methods
        :param show_inner_blobs: show the inner blobs created by the voronoi regions induced by the anchors in 2D.
        :param show_label_level_plots: show overlap and proximity matrices normalized to label level
        :param show_anchor_level_plot: show overlap and proximity matrices in anchor (sub-cluster) granularity
        :param show_loss_plot: show loss value per iteration plot
        :param every_matrix_in_single_plot: plot every matrix in single plot or all of matrices in same plot
        :param default_max_val: Default maximum value for matrices heatmaps. default 0.1
        :param mat_figsize: Tuple size of f=matrices figure
        :param mat_annot_kws_size: size of text annotation on matrices
        :param mat_label_size: size of label in matrices plot
        :param main_plot_fig_size: main plot fig size
        :param show_anchors_annotation: annotate anchors label and subcluster on the plot
        :param vmax_overlap: upper bound for overlap heatmaps
        :param vmax_proximity: upper bound for proximity heatmap
        :param annotate_images: print candidates for image annotations or not, if set to True orig_images must be use
        :param k_annot_clf: K for KNN classifier used for finding the image annotation candidates.
        :param orig_images: list of pathes for every image, pathes should be corresponding to the X matrix passed to
        fit_transform method
        """
        # start with verbose level
        self.verbose = verbose
        self.logger = logging.getLogger(f'ClusterPlot-{np.random.randint(0, 65534)}')
        self.logger.setLevel(self.verbose)
        ch = logging.StreamHandler()
        ch.setLevel(self.verbose)
        # create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        # add the handlers to the logger
        self.logger.addHandler(ch)

        # Sanity checks
        if n_components > 2:
            raise Exception(f'n_components > 2 not supported currently')

        supported_dim_reduction_algos = ['t-sne', 'umap', 'mds', 'pca', 'lda']
        if dim_reduction_algo not in supported_dim_reduction_algos:
            raise Exception(f'Unsupported dim_reduction_algo {dim_reduction_algo}, '
                            f'supported algos: {supported_dim_reduction_algos}')

        supported_anchors_methods = ['agglomerative', 'kmeans', 'birch']
        if anchors_method not in supported_anchors_methods:
            raise Exception(f'Unsupported anchors method {anchors_method}, '
                            f'supported methods: {supported_anchors_methods}')

        supported_random_points_method = ['box', 'radius', 'voronoi']
        if random_points_method not in supported_random_points_method:
            raise Exception(f'Unsupported random_points_method {random_points_method}, '
                            f' methods: {supported_random_points_method}')

        if anchors_method in ['birch']:
            if n_intra_anchors is not None:
                self.logger.warning(f'Anchors method {anchors_method} does not require number of anchors '
                                    f'but n_intra_anchors is {n_intra_anchors}')
        else:
            if not n_intra_anchors:
                raise Exception(f'n_intra_anchors is required for {anchors_method}')

        if dim_reduction_algo not in ['umap', 'lda'] and supervised:
            raise Exception(f'dim_reduction_algo {dim_reduction_algo} does not support supervision')

        supported_uniform_points_per = ['label', 'anchor']
        if uniform_points_per not in supported_uniform_points_per:
            raise Exception(f'Unsupported uniform_points_per {uniform_points_per} , '
                            f'supported methods: {supported_uniform_points_per}')

        check_greater_than_zero(k, 'k')
        check_greater_than_zero(proximity_k, 'proximity_k')
        check_greater_than_zero(top_greedy, 'top_greedy')
        check_greater_than_zero(n_iter, 'n_iter')
        check_greater_than_zero(batch_size, 'batch_size')
        check_greater_than_zero(stop_criteria, 'stop_criteria')
        check_greater_than_zero(douglas_peucker_tolerance, 'douglas_peucker_tolerance')
        check_greater_than_zero(smooth_iter, 'smooth_iter')
        check_greater_than_zero(skip_polygons_with_area, 'skip_polygons_with_area')
        check_greater_than_zero(save_fig_every, 'save_fig_every')
        check_greater_than_zero(vmax_overlap, 'vmax_overlap')
        check_greater_than_zero(vmax_proximity, 'vmax_proximity')
        check_greater_than_zero(k_annot_clf, 'k_annot_clf')

        if radius_q is not None and not 0 < radius_q <= 1:
            raise Exception(f'radius_q must bi in (0, 1]')

        if not magnitude_step and learning_rate is None:
            raise Exception('Learning rate must be defined if magnitude step is disabled')

        if len(main_plot_fig_size) != 2:
            raise Exception(f'main_plot_fig_size of len {len(main_plot_fig_size)} must be of len 2')

        # constructor members
        self.n_components = n_components
        self.anchors_method = anchors_method
        self.n_intra_anchors = n_intra_anchors
        self.birch_threshold = birch_threshold
        self.birch_branching_factor = birch_branching_factor
        self.dim_reduction_algo = dim_reduction_algo
        self.supervised = supervised
        self.umap_n_neighbors = umap_n_neighbors
        self.umap_min_dist = umap_min_dist
        self.tsne_perplexity = tsne_perplexity
        self.reduce_all_points = reduce_all_points
        self.uniform_points_per = uniform_points_per
        self.k = k  # note that in t-sne paper when they presented this method they used k=20 for mnist
        self.proximity_k = proximity_k
        self.self_relation = self_relation
        self.radius_q = radius_q
        self.do_relaxation = do_relaxation
        self.top_greedy = top_greedy
        self.magnitude_step = magnitude_step
        self.n_iter = n_iter
        self.batch_size = batch_size
        self.stop_criteria = stop_criteria
        if loss == 'Linf':
            self.loss_func = ClusterPlot._l_inf_loss
        elif loss == 'mse':
            self.loss_func = ClusterPlot._mse_loss
        else:
            raise Exception(f'Unsupported loss {loss}')
        self.loss = loss
        self.only_inter_relations = only_inter_relations
        self.learning_rate = learning_rate
        self.mask_sparse_subcluster = mask_sparse_subcluster
        self.random_points_method = random_points_method
        self.class_to_label = class_to_label
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.dataset = dataset
        self.show_fig = show_fig
        self.save_fig = save_fig
        self.figsize = figsize
        self.do_animation = do_animation
        self.use_spline = use_spline
        self.alpha = alpha
        self.remove_outliers_k = remove_outliers_k
        self.douglas_peucker_tolerance = douglas_peucker_tolerance
        self.smooth_iter = smooth_iter
        self.skip_polygons_with_area = skip_polygons_with_area
        self.mask_relation_in_same_label = mask_relation_in_same_label
        self.save_fig_every = save_fig_every
        self.show_points = show_points
        self.show_anchors = show_anchors
        self.show_blobs = show_blobs
        self.show_inner_blobs = show_inner_blobs
        self.show_label_level_plots = show_label_level_plots
        self.show_anchor_level_plot = show_anchor_level_plot
        self.show_loss_plot = show_loss_plot
        self.every_matrix_in_single_plot = every_matrix_in_single_plot
        self.default_max_val = default_max_val
        self.mat_figsize = mat_figsize
        self.mat_annot_kws_size = mat_annot_kws_size
        self.mat_label_size = mat_label_size
        self.main_plot_fig_size = main_plot_fig_size
        self.show_anchors_annotation = show_anchors_annotation
        self.vmax_overlap = vmax_overlap
        self.vmax_proximity = vmax_proximity
        self.annotate_images = annotate_images
        self.k_annot_clf = k_annot_clf
        if self.annotate_images and orig_images is None:
            raise Exception(f'Annotate images is enabled but no orig images')
        self.orig_images = orig_images
        self.transpose_low_dim = transpose_low_dim

        # create output dir
        now = datetime.now()
        date_time = now.strftime("%Y-%d-%m_%H-%M-%S")
        self.output_dir = f'../plots/{self.dataset}/{date_time}/'
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
        self.intra_class_anchors_indices = None  # Note Assuming that centroids are concatenated
        self.anchors_indices = None
        self.low_dim_anchors = None
        self.low_dim_points = None
        self.knng = None
        self.inter_class_relations = None
        self.inter_class_relations_low_dim = None
        self.inter_class_relations_label_level = None
        self.inter_class_relations_low_dim_label_level = None
        self.anchors_density = None
        self.anchors_radius = None
        self.losses = []  # for visualization
        self.anchor_voronoi_regions = []
        self.anchor_voronoi_regions_label = []
        self.cand_samples_to_plot = dict()
        # <anchor_index>: {
        #     'outlier': <>,
        #     'pure': <>,
        #     'target_anchor': <>,
        #     'src_label': <>,
        #     'target_label': <>
        # }
        self.overlapped_anchor_per_label = None
        self.pure_anchor_per_label = None
        self.high_dim_proximity_matrix = None
        self.low_dim_proximity_matrix = None

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

    def fit(self, X, y):
        """
        Not implemented
        :param X: Ignore
        :param y: Ignore
        :return:
        """
        raise NotImplementedError('Use fit_transform')

    def fit_transform(self, X, y):
        """
        labels are expected to be 0,1,2,...
        :param X: {array-like, sparse matrix} of shape (n_samples, n_features)
            Training data
        :param y: array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values. Will be cast to X's dtype if necessary
        :return: low dim anchors
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

        if isinstance(self.alpha, list):
            if len(self.alpha) != len(np.unique(y)):
                raise Exception(f'alpha is a list => it must be of the length of number of labels')
        else:
            self.alpha = [self.alpha] * len(np.unique(y))

        if self.random_state is not None:
            np.random.seed(self.random_state)
            random.seed(self.random_state)

        # Get Anchors by clustering for each label
        self._get_intra_class_anchors(X, y)

        self.X_with_centroids = np.concatenate((X, self.intra_class_anchors), axis=0)
        self.y_with_centroids = np.concatenate((y, self.intra_class_anchors_labels))

        # Assuming centroids are concatenated too in _get_intra_class_anchors
        self.anchors_indices = [i for i in range(len(self.y_with_centroids) - len(self.intra_class_anchors_labels),
                                                 len(self.y_with_centroids))]
        # Build knn Graph
        if self.do_relaxation:
            self._build_knng(self.X_with_centroids)

        # Calculate inter class relations
        if self.do_relaxation:
            self._calc_inter_class_relations()
            self._calc_proximity_matrix('high')

        # Dim Reduction
        self._dim_reduction()
        # DO relaxation in the low dimension
        if self.do_relaxation:
            self._relaxation()

        return self.low_dim_anchors

    def cluster_plot(self, figsize: tuple = None,
                     show_blobs: bool = None,
                     show_points: bool = None,
                     show_anchors: bool = None,
                     show_anchors_annotation: bool = None,
                     show_inner_blobs: bool = None,
                     douglas_peucker_tolerance: float = None,
                     smooth_iter: int = None,
                     show_label_level_plots: bool = None,
                     show_anchor_level_plot: bool = None,
                     show_loss_plot: bool = None,
                     default_max_val: float = None,
                     mat_figsize: tuple = None,
                     mat_annot_kws_size: int = None,
                     mat_label_size: int = None
                     ):
        """
        Plot Cluster Plot
        All default values of argument are None, None means taking the value from the constructor, otherwise the value
        is overriden.
        :param figsize: Figure size of main plot
        :param show_blobs: Show ClusterPlot blobs
        :param show_points: Show points on main plot or not
        :param show_anchors: Show anchors on main plot or not
        :param show_anchors_annotation: Show annotation labels ans subcluster on anchors
        :param show_inner_blobs: Show Voronoi regions
        :param douglas_peucker_tolerance: tolerance of Douglas Peucker smoothing algorithm
        :param smooth_iter: How many smoothing iterations using cutting corner algorithm
        :param show_label_level_plots: Show relations matrices in label level
        :param show_anchor_level_plot: Show relation matrices in anchor level
        :param show_loss_plot: Show loss plot or not
        :param default_max_val: Maximum default value for heatmaps in matrices plot
        :param mat_figsize: Figure size of relation matrices
        :param mat_annot_kws_size: Size of annotations on matrices plot
        :param mat_label_size: Size of label in matrices plot
        :return: None
        """

        if figsize is None:
            figsize = self.figsize
        if show_points is None:
            show_points = self.show_points
        if show_blobs is None:
            show_blobs = self.show_blobs
        if show_anchors is None:
            show_anchors = self.show_anchors
        if show_anchors_annotation is None:
            show_anchors_annotation = self.show_anchors_annotation
        if show_inner_blobs is None:
            show_inner_blobs = self.show_inner_blobs
        if douglas_peucker_tolerance is None:
            douglas_peucker_tolerance = self.douglas_peucker_tolerance
        if smooth_iter is None:
            smooth_iter = self.smooth_iter
        if show_label_level_plots is None:
            show_label_level_plots = self.show_label_level_plots
        if show_anchor_level_plot is None:
            show_anchor_level_plot = self.show_anchor_level_plot
        if show_loss_plot is None:
            show_loss_plot = self.show_loss_plot
        if default_max_val is None:
            default_max_val = self.default_max_val
        if mat_figsize is None:
            mat_figsize = self.mat_figsize
        if mat_annot_kws_size is None:
            mat_annot_kws_size = self.mat_annot_kws_size
        if mat_label_size is None:
            mat_label_size = self.mat_label_size

        self._cluster_plot_set_sns(self.n_iter, figsize=figsize,
                                   show_blobs=show_blobs,
                                   show_points=show_points,
                                   show_anchors=show_anchors,
                                   show_anchors_annotation=show_anchors_annotation,
                                   show_inner_blobs=show_inner_blobs,
                                   douglas_peucker_tolerance=douglas_peucker_tolerance,
                                   smooth_iter=smooth_iter,
                                   show_label_level_plots=show_label_level_plots,
                                   show_anchor_level_plot=show_anchor_level_plot,
                                   show_loss_plot=show_loss_plot,
                                   default_max_val=default_max_val,
                                   mat_figsize=mat_figsize,
                                   mat_annot_kws_size=mat_annot_kws_size,
                                   mat_label_size=mat_label_size
                                   )

    def _get_intra_class_anchors(self, X, y):
        """
        Perform Clustering for each class to get sub-clusters and anchors
        :param X: {array-like, sparse matrix} of shape (n_samples, n_features)
            Training data
        :param y: array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values. Will be cast to X's dtype if necessary
        :return: None
        """
        self.logger.info(f'finding intra class anchors using {self.anchors_method}')
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
                subcluster_centers = df[df[self.label_col] == label].groupby([self.label_col, self.cluster_col]).mean() \
                    .sort_index().values
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
        """
        Calc density and radius of each subcluster in high-dim
        :param df: data pandas Dataframe
        :param label_col: label column
        :param cluster_col: cluster column
        :param intra_centroids_df: anchors dataframe
        :return: None
        """
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
        radius_df = radius_df ** 0.5
        radius_df = radius_df.groupby(level=[0, 1]).quantile(q=self.radius_q if self.radius_q is not None else 1)
        self.anchors_radius = radius_df.values

    def _build_knng(self, X):
        """
        Build K-Nearest-Neighbors-Graph
        :param X: Traning data
        :return: None
        """
        self.knng = kneighbors_graph(X, self.k, mode='distance', n_jobs=self.n_jobs)

    def _sample_index_to_anchor(self, label, cluster):
        """
        Convert sample index to anchor index
        :param label: label of sample
        :param cluster: sub-cluster of sample
        :return: anchor index
        """
        anchor_index = 0
        for i in range(label):
            anchor_index += self.num_clusters_each_label[i]
        anchor_index += cluster
        return anchor_index

    def _anchor_to_label_cluster(self, anchor_index, visualization=False):
        """
        Get label and sub-cluster by anchor_index
        :param anchor_index: index of anchor
        :param visualization: for vis or not
        :return: Tuple (label, sub-cluster)
        """
        anchors_count = 0
        for i in range(len(self.num_clusters_each_label) - 1):
            if anchors_count <= anchor_index < anchors_count + self.num_clusters_each_label[i]:
                if visualization and self.class_to_label is not None:
                    return self.class_to_label[i], anchor_index - anchors_count
                else:
                    return i, anchor_index - anchors_count
            anchors_count += self.num_clusters_each_label[i]
        if visualization and self.class_to_label is not None:
            return self.class_to_label[len(self.num_clusters_each_label) - 1], anchor_index - anchors_count
        else:
            return len(self.num_clusters_each_label) - 1, anchor_index - anchors_count

    def _calc_proximity_matrix(self, dim):
        """
        Calc proximity matrix
        :param dim: high or low dim
        :return: None
        """
        if dim == 'high':
            anchors = self.intra_class_anchors
        else:
            anchors = self.low_dim_anchors
        num_labels = len(np.unique(self.intra_class_anchors_labels))
        proximity_matrix = np.zeros((num_labels, num_labels))

        # dist matrix on anchors
        dist_mat = pairwise_distances(anchors)
        # take to inf anchors from same label
        for i in range(dist_mat.shape[0]):
            for j in range(dist_mat.shape[1]):
                if self.intra_class_anchors_labels[i] == self.intra_class_anchors_labels[j]:
                    dist_mat[i][j] = np.inf
        # for each anchor find 3 closest anchors from other labels
        edges = np.argpartition(dist_mat, kth=self.proximity_k, axis=1)[:, :self.proximity_k]

        for i in range(edges.shape[0]):
            for j in range(edges.shape[1]):
                anchor_i = i
                anchor_j = edges[i][j]
                if dist_mat[anchor_i][anchor_j] < np.inf:
                    label_i = self.intra_class_anchors_labels[anchor_i]
                    label_j = self.intra_class_anchors_labels[anchor_j]
                    proximity_matrix[label_i][label_j] += 1

        # Normalize for each anchor
        sum_row = proximity_matrix.sum(axis=1, keepdims=True)
        # replace with 1 where the sum is 0 to avoid division by 0
        sum_row[sum_row == 0] = 1
        proximity_matrix = proximity_matrix / sum_row

        if dim == 'high':
            self.high_dim_proximity_matrix = proximity_matrix
        else:
            self.low_dim_proximity_matrix = proximity_matrix

    def _calc_inter_class_relations(self):
        """
        Calc proximity and overlap
        :return: None
        """
        self.inter_class_relations = np.zeros((len(self.intra_class_anchors_labels),
                                               len(self.intra_class_anchors_labels)))
        num_labels = len(np.unique(self.intra_class_anchors_labels))
        self.inter_class_relations_label_level = np.zeros((num_labels, num_labels))
        edges_x1, edges_x2 = self.knng.nonzero()
        # Initialize inter class anchors
        for x1, x2 in zip(edges_x1, edges_x2):
            anchor_x1 = self._sample_index_to_anchor(self.y_with_centroids[x1], self.clusters[x1])
            anchor_x2 = self._sample_index_to_anchor(self.y_with_centroids[x2], self.clusters[x2])
            self.inter_class_relations[anchor_x1][anchor_x2] += 1
            self.inter_class_relations_label_level[self.y_with_centroids[x1]][self.y_with_centroids[x2]] += 1
        if not self.self_relation:
            np.fill_diagonal(self.inter_class_relations, 0)

        if self.only_inter_relations:
            # fill zeros in indices where anchors are from the same label
            for i in range(len(self.intra_class_anchors_labels)):
                for j in range(len(self.intra_class_anchors_labels)):
                    label_i = self.intra_class_anchors_labels[i]
                    label_j = self.intra_class_anchors_labels[j]
                    # label_i = self.y_with_centroids[self.anchors_indices[i]]
                    # label_j = self.y_with_centroids[self.anchors_indices[j]]
                    if label_i == label_j:
                        self.inter_class_relations[i][j] = 0

            for i in range(len(self.inter_class_relations_label_level)):
                self.inter_class_relations_label_level[i][i] = 0

        # Normalize for each anchor
        sum_row = self.inter_class_relations.sum(axis=1, keepdims=True)
        sum_row_label_level = self.inter_class_relations_label_level.sum(axis=1, keepdims=True)
        # replace with 1 where the sum is 0 to avoid division by 0
        sum_row[sum_row == 0] = 1
        sum_row_label_level[sum_row_label_level == 0] = 1
        self.inter_class_relations = self.inter_class_relations / sum_row
        self.inter_class_relations_label_level = self.inter_class_relations_label_level / sum_row_label_level

        if self.annotate_images:
            self.overlapped_anchor_per_label = [None] * num_labels
            self.pure_anchor_per_label = [[] for _ in range(num_labels)]
            # build KNN classifier
            knn_classifier = KNeighborsClassifier(n_neighbors=self.k_annot_clf)
            knn_classifier.fit(self.X_with_centroids, self.y_with_centroids)
            y_predicted = knn_classifier.predict(self.X_with_centroids)
            # outliers
            #    find all mistakes
            #    for each label
            for label_i in range(num_labels):
                #       find first mistake for this label
                for sample_index in np.argwhere(y_predicted != self.y_with_centroids):
                    sample_index = sample_index[0]
                    src_label, target_label = self.y_with_centroids[sample_index], y_predicted[sample_index]
                    anchor = self._sample_index_to_anchor(int(self.y_with_centroids[sample_index]),
                                                          self.clusters[sample_index])
                    if src_label == label_i:
                        self.cand_samples_to_plot[anchor] = dict()
                        self.cand_samples_to_plot[anchor]['outlier'] = sample_index
                        self.cand_samples_to_plot[anchor]['outlier_is_valid'] = True
                        self.cand_samples_to_plot[anchor]['src_label'] = src_label
                        self.cand_samples_to_plot[anchor]['target_label'] = target_label
                        self.overlapped_anchor_per_label[src_label] = anchor
                        break
                for sample_index in np.argwhere(y_predicted == self.y_with_centroids):
                    sample_index = sample_index[0]
                    src_label, target_label = self.y_with_centroids[sample_index], y_predicted[sample_index]
                    anchor = self._sample_index_to_anchor(int(self.y_with_centroids[sample_index]),
                                                          self.clusters[sample_index])
                    if src_label == label_i:
                        if anchor not in self.cand_samples_to_plot:
                            self.cand_samples_to_plot[anchor] = dict()
                        self.cand_samples_to_plot[anchor]['pure'] = sample_index
                        self.cand_samples_to_plot[anchor]['src_label'] = src_label
                        self.logger.debug(
                            f'Adding anchor {anchor} to label {src_label} = {self.pure_anchor_per_label[src_label]}')
                        self.pure_anchor_per_label[src_label].append(anchor)
                        if len(self.pure_anchor_per_label[src_label]) == 10:
                            break

    def _calc_low_dim_inter_class_relations(self):
        """
        Calc low dim proximity and overlap
        :return: None
        """
        knng = kneighbors_graph(self.low_dim_points, self.k, mode='distance', n_jobs=self.n_jobs)
        self.inter_class_relations_low_dim = np.zeros((len(self.intra_class_anchors_labels),
                                                       len(self.intra_class_anchors_labels)))
        num_labels = len(np.unique(self.intra_class_anchors_labels))
        self.inter_class_relations_low_dim_label_level = np.zeros((num_labels, num_labels))
        edges_x1, edges_x2 = knng.nonzero()
        # Initialize inter class anchors
        for x1, x2 in zip(edges_x1, edges_x2):
            anchor_x1 = self._sample_index_to_anchor(self.y_with_centroids[x1], self.clusters[x1])
            anchor_x2 = self._sample_index_to_anchor(self.y_with_centroids[x2], self.clusters[x2])
            self.inter_class_relations_low_dim[anchor_x1][anchor_x2] += 1
            self.inter_class_relations_low_dim_label_level[self.y_with_centroids[x1]][self.y_with_centroids[x2]] += 1
        if not self.self_relation:
            np.fill_diagonal(self.inter_class_relations_low_dim, 0)

        if self.only_inter_relations:
            # fill zeros in indices where anchors are from the same label
            for i in range(len(self.intra_class_anchors_labels)):
                for j in range(len(self.intra_class_anchors_labels)):
                    label_i = self.y_with_centroids[self.anchors_indices[i]]
                    label_j = self.y_with_centroids[self.anchors_indices[j]]
                    if label_i == label_j:
                        self.inter_class_relations_low_dim[i][j] = 0

            for i in range(len(self.inter_class_relations_label_level)):
                self.inter_class_relations_low_dim_label_level[i][i] = 0

        # Normalize for each anchor
        sum_row = self.inter_class_relations_low_dim.sum(axis=1, keepdims=True)
        sum_row_label_level = self.inter_class_relations_low_dim_label_level.sum(axis=1, keepdims=True)
        # replace with 1 where the sum is 0 to avoid division by 0
        sum_row[sum_row == 0] = 1
        sum_row_label_level[sum_row_label_level == 0] = 1
        self.inter_class_relations_low_dim = self.inter_class_relations_low_dim / sum_row
        self.inter_class_relations_low_dim_label_level = \
            self.inter_class_relations_low_dim_label_level / sum_row_label_level

    def _dim_reduction(self):
        """
        Reduce dimension of points or anchors
        :return:
        """
        if self.dim_reduction_algo == 't-sne':
            dim_reduction_algo_inst = TSNE(random_state=self.random_state, perplexity=self.tsne_perplexity)
        elif self.dim_reduction_algo == 'umap':
            dim_reduction_algo_inst = umap.UMAP(n_components=self.n_components, min_dist=self.umap_min_dist,
                                                n_neighbors=self.umap_n_neighbors,
                                                random_state=self.random_state)
        elif self.dim_reduction_algo == 'mds':
            dim_reduction_algo_inst = MDS(n_components=self.n_components, random_state=self.random_state)
        elif self.dim_reduction_algo == 'pca':
            dim_reduction_algo_inst = PCA(n_components=self.n_components, random_state=self.random_state)
        elif self.dim_reduction_algo == 'lda':
            dim_reduction_algo_inst = LatentDirichletAllocation(n_components=self.n_components,
                                                                random_state=self.random_state)
        else:
            raise Exception(f'Dimension reduction algorithm {self.dim_reduction_algo} is not supported')
        if self.supervised:
            self.logger.info('Supervised Dim Reduction')
            if self.reduce_all_points:
                self.logger.info('Dim Reduction all points')
                self.low_dim_points = dim_reduction_algo_inst.fit_transform(self.X_with_centroids,
                                                                            self.y_with_centroids)
            else:
                self.logger.info('Dim Reduction only anchors')
                self.low_dim_anchors = dim_reduction_algo_inst.fit_transform(
                    self.X_with_centroids[self.anchors_indices],
                    self.y_with_centroids[self.anchors_indices])
        else:
            self.logger.info('UnSupervised Dim Reduction')
            if self.reduce_all_points:
                self.logger.info('Dim Reduction all points')
                self.low_dim_points = dim_reduction_algo_inst.fit_transform(self.X_with_centroids)
            else:
                self.logger.info('Dim Reduction only anchors')
                self.low_dim_anchors = dim_reduction_algo_inst.fit_transform(
                    self.X_with_centroids[self.anchors_indices])
        if self.reduce_all_points:
            self.low_dim_anchors = self.low_dim_points[self.anchors_indices]
        else:
            # generate random points for each anchor in the low dimension within radius
            self.logger.info(
                f'Dim Reduction only anchors - generate random points in low dim per {self.uniform_points_per}')

            # Support random points in Voronoi regions
            label_to_contour_df = dict()
            contours_df = self._get_contour_df()
            for label in sorted(contours_df[self.label_col].unique()):
                points = contours_df[contours_df[self.label_col] == label][[self.x_col, self.y_col]].values
                label_to_contour_df[label] = self._get_concave_hull(points, alpha=self.alpha[label],
                                                                    remove_outliers_k=self.remove_outliers_k,
                                                                    spline=self.use_spline,
                                                                    douglas_peucker_tolerance=self.douglas_peucker_tolerance,
                                                                    smooth_iter=self.smooth_iter)
                if len(label_to_contour_df[label]) > 1:
                    polygon = MultiPolygon([Polygon(p) for p in label_to_contour_df[label]])
                else:
                    polygon = Polygon(label_to_contour_df[label][0])
                points = []
                # if cluster label is broken to multipolygon to voronoi is invalid need to handle every polygon by itself
                anchor_index = -1  # debug only
                for anchor, anchor_label in zip(self.low_dim_anchors, self.intra_class_anchors_labels):
                    anchor_index += 1
                    if anchor_label != label:
                        continue
                    p = Point(anchor)
                    if polygon.contains(p) or polygon.intersects(p):
                        points.append(anchor)
                    else:
                        self.logger.debug(
                            f'ANCHOR: {anchor_index}: {p} is outside of polygon ({type(polygon)}): {polygon}. Skipping anchor')
                # handle regions with one or two points
                if len(points) < 3:
                    for i in range(len(points)):
                        point = points[i]
                        points.append([point[0] + 0.01, point[1]])
                        points.append([point[0] - 0.01, point[1]])
                        points.append([point[0], point[1] + 0.01])
                        points.append([point[0], point[1] - 0.01])
                points = np.array(points)

                vor = Voronoi(points)

                regions, vertices = ClusterPlot._voronoi_finite_polygons_2d(vor)

                mask = polygon
                for region in regions:
                    polygon_region = vertices[region]
                    shape = list(polygon_region.shape)
                    shape[0] += 1
                    p = Polygon(np.append(polygon_region, polygon_region[0]).reshape(*shape)).intersection(mask)
                    if p.is_empty or isinstance(p, Point):
                        self.logger.warning('Intersection of voronoy with polygon is empty or in single point')
                    self.anchor_voronoi_regions.append(p)
                    self.anchor_voronoi_regions_label.append(label)

                    # Comment out if you want to see plot of the regions a long with the anchors
            #         poly = np.array(list(zip(p.boundary.coords.xy[0][:-1], p.boundary.coords.xy[1][:-1])))
            #         new_vertices.append(poly)
            #         plt.fill(*zip(*poly), alpha=0.4)
            #     plt.plot(points[:, 0], points[:, 1], 'ko')
            # plt.title("Clipped Voronois")
            # plt.show()

            low_dim_points = [None] * self.X_with_centroids.shape[0]
            from tqdm import tqdm
            for i in tqdm(range(self.X_with_centroids.shape[0])):
                if self.uniform_points_per == 'label':
                    concave_hulls = label_to_contour_df[self.y_with_centroids[i]]
                else:
                    concave_hulls = None
                self._random_point_low_dim(i, low_dim_points, concave_hulls)
            self.low_dim_points = np.array(low_dim_points)

        # Rotate low dim points and anchors
        if self.transpose_low_dim:
            self.low_dim_points[:, [0, 1]] = self.low_dim_points[:, [1, 0]]
            self.low_dim_anchors[:, [0, 1]] = self.low_dim_anchors[:, [1, 0]]

    @staticmethod
    def _voronoi_finite_polygons_2d(vor, radius=None):
        """
        Reconstruct infinite voronoi regions in a 2D diagram to finite
        regions.

        Parameters
        ----------
        vor : Voronoi
            Input diagram
        radius : float, optional
            Distance to 'points at infinity'.

        Returns
        -------
        regions : list of tuples
            Indices of vertices in each revised Voronoi regions.
        vertices : list of tuples
            Coordinates for revised Voronoi vertices. Same as coordinates
            of input vertices, with 'points at infinity' appended to the
            end.

        """

        if vor.points.shape[1] != 2:
            raise ValueError("Requires 2D input")

        new_regions = []
        new_vertices = vor.vertices.tolist()

        center = vor.points.mean(axis=0)
        if radius is None:
            radius = vor.points.ptp().max()

        # Construct a map containing all ridges for a given point
        all_ridges = {}
        for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
            all_ridges.setdefault(p1, []).append((p2, v1, v2))
            all_ridges.setdefault(p2, []).append((p1, v1, v2))

        # Reconstruct infinite regions
        for p1, region in enumerate(vor.point_region):
            vertices = vor.regions[region]

            if all(v >= 0 for v in vertices):
                # finite region
                new_regions.append(vertices)
                continue

            # reconstruct a non-finite region
            ridges = all_ridges[p1]
            new_region = [v for v in vertices if v >= 0]

            for p2, v1, v2 in ridges:
                if v2 < 0:
                    v1, v2 = v2, v1
                if v1 >= 0:
                    # finite ridge: already in the region
                    continue

                # Compute the missing endpoint of an infinite ridge

                t = vor.points[p2] - vor.points[p1]  # tangent
                t /= np.linalg.norm(t)
                n = np.array([-t[1], t[0]])  # normal

                midpoint = vor.points[[p1, p2]].mean(axis=0)
                direction = np.sign(np.dot(midpoint - center, n)) * n
                far_point = vor.vertices[v2] + direction * radius

                new_region.append(len(new_vertices))
                new_vertices.append(far_point.tolist())

            # sort region counterclockwise
            vs = np.asarray([new_vertices[v] for v in new_region])
            c = vs.mean(axis=0)
            angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
            new_region = np.array(new_region)[np.argsort(angles)]

            # finish
            new_regions.append(new_region.tolist())

        return new_regions, np.asarray(new_vertices)

    def _random_point_low_dim(self,
                              i,
                              low_dim_points,
                              concave_hulls):
        """
        Generate virtual random points in low dim after dim reduction
        :param i: sample index
        :param low_dim_points: points in low dim
        :param concave_hulls: voronoi shapes
        :return: None
        """
        anchor_index = self._sample_index_to_anchor(self.y_with_centroids[i], self.clusters[i])
        if i in self.anchors_indices:
            low_dim_points[i] = self.low_dim_anchors[self.anchors_indices.index(i)]
        else:
            # generate random point
            # TODO ORM doesn't support 3d
            if self.uniform_points_per == 'anchor':  # uniform points per anchor
                low_dim_points[i] = self._random_points_per_cluster(anchor_index, 1)[0]
            elif self.uniform_points_per == 'label':
                polygons = []
                areas = []
                minxs = []
                minys = []
                maxxs = []
                maxys = []
                for concave_hull in concave_hulls:
                    # create shapely objects
                    poly = Polygon(concave_hull)
                    polygons.append(poly)
                    areas.append(poly.area)
                    # get bounding box of polygon
                    minx, miny, maxx, maxy = poly.bounds
                    minxs.append(minx)
                    minys.append(miny)
                    maxxs.append(maxx)
                    maxys.append(maxy)
                # choose poly with respect to area
                poly_i = np.random.choice(len(polygons), 1, p=np.array(areas) / sum(areas))[0]
                # minx, miny, maxx, maxy = min(minxs), min(minys), max(maxxs), max(maxys)
                minx, miny, maxx, maxy = minxs[poly_i], minys[poly_i], maxxs[poly_i], maxys[poly_i]
                # generate random points within the bounding box
                in_poly = False
                while not in_poly:
                    x_cord, y_cord = self._random_points_per_cluster(anchor_index, 1)[0]
                    # x_cord = np.random.uniform(low=minx, high=maxx)
                    # y_cord = np.random.uniform(low=miny, high=maxy)
                    for p in polygons:
                        if p.contains(Point(x_cord, y_cord)):
                            low_dim_points[i] = np.array([x_cord, y_cord])
                            in_poly = True
                            break
            else:
                raise Exception(f'Unsupported uniform_points_per: {self.uniform_points_per}')

    def _random_points_per_cluster(self, anchor_index, number_of_random_points=None):
        """
        Generate virtual random points per subcluster
        :param anchor_index: anchor index
        :param number_of_random_points: how many points to generate
        :return: np array of random points
        """
        if number_of_random_points is not None:
            number_of_random_points = self.anchors_density[anchor_index]

        if self.random_points_method == 'voronoi':
            # naive implementation for now and find the correct region by brute force
            low_dim_anchor = Point(self.low_dim_anchors[anchor_index])
            _poly = None
            for i, r in enumerate(self.anchor_voronoi_regions):
                if self.anchor_voronoi_regions_label[i] == self.intra_class_anchors_labels[anchor_index] and \
                        (r.contains(low_dim_anchor) or r.intersects(
                            low_dim_anchor)):  # in case that the anchor intersects the exterior polygon
                    _poly = r
                    break
            if _poly is None:
                # This is probably sub-cluster with one point that is outlier, so it is outside the polygon and it is
                # ok to return the anchor
                self.logger.debug(
                    f'anchor {anchor_index} has no voronoi region. This is sub-cluster with one point that is outlier, so it is outside the polygon and it is returning anchor instead of random point')
                return [self.low_dim_anchors[anchor_index]]
            minx, miny, maxx, maxy = _poly.bounds
            # generate random points within the bounding box
            random_points = []
            while len(random_points) < number_of_random_points:
                x = np.random.uniform(low=minx, high=maxx)
                y = np.random.uniform(low=miny, high=maxy)
                if _poly.contains(Point(x, y)):
                    random_points.append([x, y])
        elif self.random_points_method == 'box':
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
        elif self.random_points_method == 'radius':
            # generate the points
            # TODO ORM not supporting 3d
            radius = self.anchors_radius[anchor_index]
            theta = np.random.rand(number_of_random_points) * (2 * np.pi)
            r = radius * np.sqrt(np.random.rand(number_of_random_points))
            x, y = r * np.cos(theta), r * np.sin(theta)
            x = x + self.low_dim_anchors[anchor_index][0]
            y = y + self.low_dim_anchors[anchor_index][1]
            random_points = np.array(list(zip(x, y)))
        else:
            raise Exception(f'Unsupported random_points_method {self.random_points_method}')
        return random_points

    @staticmethod
    def _l_inf_loss(X, Y):
        return np.absolute(X - Y).max()

    @staticmethod
    def _mse_loss(X, Y):
        return np.square(X - Y).mean()

    def _get_top_anchors_to_relax(self):
        """
        Get top greedy anchors with the highest loss value to perform relaxatio on them
        :return: Tuple (np array of anchors indices to relax, np array of target anchors indices, np array of directions, np array of magnitudes)
        """
        inter_class_relations, inter_class_relations_low_dim = self._mask_inter_class_relations()

        diff_mat = inter_class_relations - inter_class_relations_low_dim
        # diff_mat = self.inter_class_relations - self.inter_class_relations_low_dim
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
        return src_anchor_indices[:self.top_greedy], target_anchor_indices[:self.top_greedy], directions[
                                                                                              :self.top_greedy], magnitudes[
                                                                                                                 :self.top_greedy]

    def _mask_inter_class_relations(self):
        """
        Mask out from overlap matrix relations between sub-clusters in the same label
        :return:
        """
        inter_class_relations = self.inter_class_relations
        inter_class_relations_low_dim = self.inter_class_relations_low_dim
        if self.mask_relation_in_same_label:
            for i in range(len(self.intra_class_anchors_labels)):
                for j in range(len(self.intra_class_anchors_labels)):
                    label_i = self.intra_class_anchors_labels[i]
                    label_j = self.intra_class_anchors_labels[j]
                    if label_i == label_j:
                        inter_class_relations[i][j] = 0
                        inter_class_relations_low_dim[i][j] = 0
        if self.mask_sparse_subcluster is not None:
            mask = np.tile(self.anchors_density < self.mask_sparse_subcluster, (len(self.anchors_density), 1)).T
            inter_class_relations[mask] = 0
            inter_class_relations_low_dim[mask] = 0
        return inter_class_relations, inter_class_relations_low_dim

    def _relax_anchor_cluster(self, src_anchor_index, target_anchor_index, direction, magnitude):
        """
        Perform one step of relaxation on one anchor
        :param src_anchor_index: anchor to relax
        :param target_anchor_index: target anchor to relax in his direction
        :param direction: positive or negative
        :param magnitude: in which magnitude to relax
        :return: None
        """
        if self.mask_sparse_subcluster is not None and self.anchors_density[
            src_anchor_index] < self.mask_sparse_subcluster:
            self.logger.debug(
                f'Skipping relaxation of src {src_anchor_index} ({self.anchors_density[src_anchor_index]}) '
                f'and target {target_anchor_index} ({self.anchors_density[target_anchor_index]})')
            return
        self.logger.debug(
            f'src: {src_anchor_index} target {target_anchor_index} dir {direction} density {self.anchors_density[src_anchor_index]}'
            f'loss: {self.inter_class_relations[src_anchor_index][target_anchor_index] - self.inter_class_relations_low_dim[src_anchor_index][target_anchor_index]}')
        src_anchor, target_anchor = self.low_dim_anchors[src_anchor_index], self.low_dim_anchors[target_anchor_index]
        direction_vec = target_anchor - src_anchor
        # update low dim points of all points in the same label cluster
        label, cluster = self._anchor_to_label_cluster(src_anchor_index)
        label_indices = np.argwhere(self.y_with_centroids == label)
        cluster_indices = np.argwhere(self.clusters == cluster)
        label_cluster_indices = np.intersect1d(label_indices, cluster_indices)
        if not self.magnitude_step:
            magnitude = 1
        if self.learning_rate is not None:
            magnitude = magnitude * self.learning_rate
        self.logger.debug(
            f'update src {src_anchor_index} to target {target_anchor_index} dir {direction} magnitude {magnitude} direction_vec {direction_vec}')
        self.low_dim_points[label_cluster_indices] = self.low_dim_points[
                                                         label_cluster_indices] + direction * magnitude * direction_vec
        # update also the low dim anchors
        self.low_dim_anchors[src_anchor_index] = self.low_dim_anchors[
                                                     src_anchor_index] + direction * magnitude * direction_vec

    def _relaxation(self):
        """
        Perform relaxation/optimization
        :return: None
        """
        saved_iterations_for_gif = []
        for i in range(self.n_iter):
            self._calc_low_dim_inter_class_relations()
            self._calc_proximity_matrix('low')

            inter_class_relations, inter_class_relations_low_dim = self._mask_inter_class_relations()
            loss = self.loss_func(inter_class_relations, inter_class_relations_low_dim)
            if loss < self.stop_criteria:
                self.logger.info(f'loss {loss} < stopping criteria {self.stop_criteria} nothing to do')
                return
            self.losses.append(loss)
            self.logger.info(f'Starting iteration {i + 1} loss = {loss}')
            is_first = True
            for j in range(self.batch_size):
                src_anchor_indices, target_anchor_indices, directions, magnitudes = self._get_top_anchors_to_relax()
                for ra_i in range(len(src_anchor_indices)):
                    self._relax_anchor_cluster(src_anchor_indices[ra_i], target_anchor_indices[ra_i], directions[ra_i],
                                               magnitudes[ra_i])
            if (self.save_fig or self.show_fig) and i % self.save_fig_every == 0:
                saved_iterations_for_gif.append(i)
                self._cluster_plot_set_sns(i,
                                           figsize=self.figsize,
                                           show_blobs=self.show_blobs,
                                           show_points=self.show_points,
                                           show_anchors=self.show_anchors,
                                           show_anchors_annotation=self.show_anchors_annotation,
                                           show_inner_blobs=self.show_inner_blobs,
                                           douglas_peucker_tolerance=self.douglas_peucker_tolerance,
                                           smooth_iter=self.smooth_iter,
                                           show_label_level_plots=self.show_label_level_plots,
                                           show_anchor_level_plot=self.show_anchor_level_plot,
                                           show_loss_plot=self.show_loss_plot,
                                           default_max_val=self.default_max_val,
                                           mat_figsize=self.mat_figsize,
                                           mat_annot_kws_size=self.mat_annot_kws_size,
                                           mat_label_size=self.mat_label_size
                )

        if self.do_animation:
            gif_path = f'{self.output_dir}/animation.gif'
            with imageio.get_writer(gif_path, mode='I') as writer:
                for i in saved_iterations_for_gif:
                    for j in range(5):
                        writer.append_data(
                            imageio.imread(f'{self.output_dir}/iter_{i}_points_anchors_patches_plot.png'))

    def _cluster_plot_set_sns(self, i,
                              figsize: tuple = (21, 15),
                              show_blobs: bool = True,
                              show_points: bool = False,
                              show_anchors: bool = False,
                              show_anchors_annotation: bool = False,
                              show_inner_blobs: bool = False,
                              douglas_peucker_tolerance: float = 0.6,
                              smooth_iter: int = 3,
                              show_label_level_plots: bool = True,
                              show_anchor_level_plot: bool = False,
                              show_loss_plot: bool = True,
                              default_max_val: float = 0.1,
                              mat_figsize: tuple = (14, 10),
                              mat_annot_kws_size: int = 35,
                              mat_label_size: int = 25
                              ):
        sns.set_style("darkgrid")
        self._anchors_plot_sns_separate(i, figsize=figsize,
                                        show_blobs=show_blobs,
                                        show_points=show_points,
                                        show_anchors=show_anchors,
                                        show_anchors_annotation=show_anchors_annotation,
                                        show_inner_blobs=show_inner_blobs,
                                        douglas_peucker_tolerance=douglas_peucker_tolerance,
                                        smooth_iter=smooth_iter,
                                        show_label_level_plots=show_label_level_plots,
                                        show_anchor_level_plot=show_anchor_level_plot,
                                        show_loss_plot=show_loss_plot,
                                        default_max_val=default_max_val,
                                        mat_figsize=mat_figsize,
                                        mat_annot_kws_size=mat_annot_kws_size,
                                        mat_label_size=mat_label_size
                                        )
        sns.reset_defaults()

    def _anchors_plot_sns_separate(self, i, figsize: tuple = (21, 15),
                                   show_blobs: bool = True,
                                   show_points: bool = False,
                                   show_anchors: bool = False,
                                   show_anchors_annotation: bool = False,
                                   show_inner_blobs: bool = False,
                                   douglas_peucker_tolerance: float = 0.6,
                                   smooth_iter: int = 3,
                                   show_label_level_plots: bool = True,
                                   show_anchor_level_plot: bool = False,
                                   show_loss_plot: bool = True,
                                   default_max_val: float = 0.1,
                                   mat_figsize: tuple = (14, 10),
                                   mat_annot_kws_size: int = 35,
                                   mat_label_size: int = 25
                                   ):
        """
        Plot the results
        :param i: Iteration Number
        :param figsize: Figure size of main plot
        :param show_blobs: Show ClusterPlot blobs
        :param show_points: Show points on main plot or not
        :param show_anchors: Show anchors on main plot or not
        :param show_anchors_annotation: Show annotation labels ans subcluster on anchors
        :param show_inner_blobs: Show Voronoi regions
        :param douglas_peucker_tolerance: tolerance of Douglas Peucker smoothing algorithm
        :param smooth_iter: How many smoothing iterations using cutting corner algorithm
        :param show_label_level_plots: Show relations matrices in label level
        :param show_anchor_level_plot: Show relation matrices in anchor level
        :param show_loss_plot: Show loss plot or not
        :param default_max_val: Maximum default value for heatmaps in matrices plot
        :param mat_figsize: Figure size of relation matrices
        :param mat_annot_kws_size: Size of annotations on matrices plot
        :param mat_label_size: Size of label in matrices plot
        :return: None
        """
        df = pd.DataFrame(data=self.low_dim_points, columns=[self.x_col, self.y_col])
        df[self.label_col] = self.y_with_centroids
        df[self.cluster_col] = self.clusters
        df[self.anchor_col] = False
        df.loc[df.index.isin(self.anchors_indices), self.anchor_col] = True

        if self.class_to_label:
            df[self.label_col] = df[self.label_col].transform(lambda x: f'{x}_{self.class_to_label[x]}')
        else:
            df[self.label_col] = df[self.label_col].transform(lambda x: f'label_{x}')

        # main plot
        #   plot main plot anyway
        self._points_anchors_patches_plot(df, i, show_inner_blobs=False, annotate_images=False, figsize=figsize,
                                          show_blobs=show_blobs,
                                          show_points=show_points,
                                          show_anchors=show_anchors,
                                          show_anchors_annotation=show_anchors_annotation,
                                          douglas_peucker_tolerance=douglas_peucker_tolerance,
                                          smooth_iter=smooth_iter)
        #   if blobs main plot with blobs
        if show_inner_blobs:
            self._points_anchors_patches_plot(df, i, show_inner_blobs=True, annotate_images=False, figsize=figsize,
                                              show_blobs=show_blobs,
                                              show_points=show_points,
                                              show_anchors=show_anchors,
                                              show_anchors_annotation=show_anchors_annotation,
                                              douglas_peucker_tolerance=douglas_peucker_tolerance,
                                              smooth_iter=smooth_iter)
        #   if images plot
        if self.annotate_images:
            self._points_anchors_patches_plot(df, i, show_inner_blobs=False, annotate_images=True, figsize=figsize,
                                              show_blobs=show_blobs,
                                              show_points=show_points,
                                              show_anchors=show_anchors,
                                              show_anchors_annotation=show_anchors_annotation,
                                              douglas_peucker_tolerance=douglas_peucker_tolerance,
                                              smooth_iter=smooth_iter)

        # matrices
        #   if label level plot
        if show_label_level_plots:
            self._matrices_plot(label_level=True, iteration=i, default_max_val=default_max_val,
                                mat_figsize=mat_figsize,
                                mat_annot_kws_size=mat_annot_kws_size,
                                mat_label_size=mat_label_size
                                )
        #   if anchors level plot
        if show_anchor_level_plot:
            self._matrices_plot(label_level=False, iteration=i, default_max_val=default_max_val,
                                mat_figsize=mat_figsize,
                                mat_annot_kws_size=mat_annot_kws_size,
                                mat_label_size=mat_label_size
                                )

        # loss
        if show_loss_plot:
            self._loss_plot()

    @staticmethod
    def _expand_matplotlib_bbox(extent, sw, sh):
        """
        Construct a `Bbox` by expanding this one around its center by the
        factors *sw* and *sh*.
        """
        width = extent.width
        height = extent.height
        deltaw = (sw * width - width) / 2.0
        deltah = (sh * height - height) / 2.0
        deltah_top = (1.3 * height - height) / 2.0
        a = np.array([[-deltaw, -deltah], [deltaw, deltah_top]])
        return Bbox(extent._points + a)

    def _matrices_plot(self, label_level, iteration, default_max_val: float = 0.1, mat_figsize: tuple = (14, 10),
                       mat_annot_kws_size: int = 35, mat_label_size: int = 25):
        """
        Plot overlap and proximity matrices
        :param label_level: label level or sub-cluster level plots
        :param iteration: iteration number
        :param default_max_val: maximum value for heatmaps default (0.1)
        :param mat_figsize: Figure size of relation matrices
        :param mat_annot_kws_size: Size of annotations on matrices plot
        :param mat_label_size: Size of label in matrices plot
        :return: None
        """
        a_min = 0
        a_max = default_max_val
        rows = ['Overlap', 'Proximity']
        cols = ['High-Dim', 'Low-Dim', 'Diff']
        label_level_ticks = [self.class_to_label[i] for i in sorted(self.class_to_label.keys())]
        anchor_level_ticks = [str(self._anchor_to_label_cluster(i, visualization=True)) for i in
                              range(self.inter_class_relations.shape[0])]
        if label_level:
            matrices = {
                'Overlap Diff': {'mat': np.clip(
                    np.abs(self.inter_class_relations_label_level - self.inter_class_relations_low_dim_label_level),
                    a_min=a_min,
                    a_max=a_max), 'ticks': label_level_ticks},
                'Overlap High-Dim': {'mat': self.inter_class_relations_label_level, 'ticks': label_level_ticks},
                'Overlap Low-Dim': {'mat': self.inter_class_relations_low_dim_label_level, 'ticks': label_level_ticks},
                'Proximity Diff': {'mat': np.clip(
                    np.square(self.high_dim_proximity_matrix - self.low_dim_proximity_matrix),
                    a_min=a_min,
                    a_max=a_max), 'ticks': label_level_ticks},
                'Proximity High-Dim': {'mat': self.high_dim_proximity_matrix, 'ticks': label_level_ticks},
                'Proximity Low-Dim': {'mat': self.low_dim_proximity_matrix, 'ticks': label_level_ticks},
            }
        else:
            matrices = {
                'Overlap Diff': {'mat': np.clip(
                    np.square(self.inter_class_relations - self.inter_class_relations_low_dim),
                    a_min=a_min,
                    a_max=a_max), 'ticks': anchor_level_ticks},
                'Overlap High-Dim': {'mat': self.inter_class_relations, 'ticks': anchor_level_ticks},
                'Overlap Low-Dim': {'mat': self.inter_class_relations_low_dim, 'ticks': anchor_level_ticks},
                'Proximity Diff': {'mat': np.clip(
                    np.square(self.high_dim_proximity_matrix - self.low_dim_proximity_matrix),
                    a_min=a_min,
                    a_max=a_max), 'ticks': label_level_ticks},
                'Proximity High-Dim': {'mat': self.high_dim_proximity_matrix, 'ticks': label_level_ticks},
                'Proximity Low-Dim': {'mat': self.low_dim_proximity_matrix, 'ticks': label_level_ticks},
            }
        if not self.every_matrix_in_single_plot:
            fig, axs = plt.subplots(len(rows), len(cols),
                                    figsize=(mat_figsize[0] * len(cols), mat_figsize[1] * len(rows)))
        for i, row in enumerate(rows):
            for j, col in enumerate(cols):
                if self.every_matrix_in_single_plot:
                    fig, ax = plt.subplots(1, 1, figsize=mat_figsize)
                else:
                    ax = axs[i][j]
                matrix_key = f'{row} {col}'
                matrix = matrices[matrix_key]['mat']
                if self.mask_relation_in_same_label:
                    np.fill_diagonal(matrix, np.nan)
                ticks = matrices[matrix_key]['ticks']
                if col == 'Diff':
                    vmin = a_min
                    vmax = a_max
                    annot = True
                    cmap = 'OrRd'
                else:
                    vmin = 0
                    vmax = self.vmax_overlap
                    annot = False
                    if row == 'Overlap':
                        cmap = 'YlGnBu'
                    else:  # Proximity
                        cmap = 'BuPu'
                        vmax = self.vmax_proximity
                g = sns.heatmap(matrix, ax=ax, annot=annot, fmt='.3f', square=True, cmap=cmap,
                                vmin=vmin, vmax=vmax, center=((vmax - vmin) / 2),
                                xticklabels=ticks, yticklabels=ticks, annot_kws={"size": mat_annot_kws_size})
                # ax.set_title(matrix_key)   # comment out for paper
                ax.tick_params(axis='both', which='major', labelsize=mat_label_size)
                ax.tick_params(axis='both', which='minor', labelsize=mat_label_size)
                g.set_yticklabels(g.get_yticklabels(), rotation=0)
                g.set_xticklabels(g.get_xticklabels(), rotation=90)
                if self.save_fig and self.every_matrix_in_single_plot:
                    # Save just the portion _inside_ the second axis's boundaries
                    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                    fig.savefig(f'{self.output_dir}/{matrix_key}_iter_{iteration}.png',
                                bbox_inches=ClusterPlot._expand_matplotlib_bbox(extent, 2.5, 2.5))

        if not self.every_matrix_in_single_plot:
            if self.save_fig:
                # Save just the portion _inside_ the second axis's boundaries
                fig.savefig(
                    f'{self.output_dir}/Overlap_and_Proximity_Matrices{"_per_Label" if label_level else ""}.png')
            if self.show_fig:
                plt.show()

    def _loss_plot(self):
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.lineplot(x=list(range(len(self.losses))), y=self.losses, ax=ax)
        ax.set_xlim((0, self.n_iter + 3))
        ax.set_ylim((-0.05, 1.1))
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss')
        ax.set_title('Loss')

        if self.save_fig:
            # Save just the portion _inside_ the second axis's boundaries
            extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            fig.savefig(f'{self.output_dir}/loss_plot.png', bbox_inches=extent.expanded(1.3, 1.3))
        if self.show_fig:
            plt.show()

    def _points_anchors_patches_plot(self, df, i, show_inner_blobs, annotate_images,
                                     figsize: tuple = (21, 15),
                                     show_blobs: bool = True,
                                     show_points: bool = False,
                                     show_anchors: bool = False,
                                     show_anchors_annotation: bool = False,
                                     douglas_peucker_tolerance: float = 0.6,
                                     smooth_iter: int = 3):
        fig, ax = plt.subplots(figsize=figsize)
        hue_order = sorted(df[self.label_col].unique())
        # Plot points
        if show_points:
            sns.scatterplot(data=df[df['anchor'] == False], x=self.x_col, y=self.y_col, hue=self.label_col,
                            style=self.cluster_col if show_anchors else None, ax=ax,
                            alpha=0.2 if show_blobs else 1, legend=False,
                            hue_order=hue_order, s=240)
        # Plot anchors
        if show_anchors:
            sns.scatterplot(data=df[df['anchor'] == True], x=self.x_col, y=self.y_col, hue=self.label_col,
                            ax=ax,
                            alpha=1,
                            hue_order=hue_order,
                            s=240)
        else:
            # it enables the legend
            sns.lineplot(data=df[df['anchor'] == True], x=self.x_col, y=self.y_col, hue=self.label_col,
                         ax=ax,
                         alpha=0,
                         hue_order=hue_order)
        # For each anchor, we add a text above the anchor
        if show_anchors_annotation:
            for j, anchor_i in enumerate(self.anchors_indices):
                ax.text(self.low_dim_points[anchor_i][0], self.low_dim_points[anchor_i][1],
                        f'{self.anchors_density[j]}',
                        horizontalalignment='center', size='medium',
                        color='black', weight='semibold')
        # ax.set_title(f'iter{i}')
        minx, maxx = self.low_dim_points[:, 0].min(), self.low_dim_points[:, 0].max()
        marginx = (maxx - minx) * 0.05
        miny, maxy = self.low_dim_points[:, 1].min(), self.low_dim_points[:, 1].max()
        marginy = (maxy - miny) * 0.05
        xlim = (minx - marginx, maxx + marginx)
        ylim = (miny - marginy, maxy + marginy)

        # Add patches
        if show_blobs:
            contours_df = self._get_contour_df()
            palette = itertools.cycle(sns.color_palette())
            for label in sorted(contours_df[self.label_col].unique()):
                points = contours_df[contours_df[self.label_col] == label][[self.x_col, self.y_col]].values
                concave_hulls = self._get_concave_hull(points, alpha=self.alpha[label],
                                                       remove_outliers_k=self.remove_outliers_k,
                                                       spline=self.use_spline, vis=True,
                                                       douglas_peucker_tolerance=douglas_peucker_tolerance,
                                                       smooth_iter=smooth_iter)
                c = next(palette)
                for concave_hull in concave_hulls:
                    # Skip polygons with very small area that will appear as dots
                    if Polygon(concave_hull).area < self.skip_polygons_with_area:
                        self.logger.debug(f'Skipping polygon of label {label} with area {Polygon(concave_hull).area}')
                        continue
                    coords = concave_hull
                    line_cmde = [Path.MOVETO] + [Path.LINETO] * (len(coords) - 2) + [Path.CLOSEPOLY]
                    path = Path(coords, line_cmde)
                    patch = patches.PathPatch(path, facecolor=c, alpha=0.2, linewidth=None, edgecolor=None)
                    ax.add_patch(patch)
                    patch = patches.PathPatch(path, facecolor=None, linewidth=10, edgecolor=c, fill=False)
                    ax.add_patch(patch)

            if show_inner_blobs:
                # In case of performance issue, calculate the voronoi regions outside
                label_to_contour_df = dict()
                contours_df = self._get_contour_df()
                palette = itertools.cycle(sns.color_palette())
                for label in sorted(contours_df[self.label_col].unique()):
                    c = next(palette)
                    # Skip labels with less than 3 anchors, Voronoi does not support that
                    if self.num_clusters_each_label[label] < 3:
                        continue
                    points = contours_df[contours_df[self.label_col] == label][[self.x_col, self.y_col]].values
                    label_to_contour_df[label] = self._get_concave_hull(points, alpha=self.alpha[label],
                                                                        remove_outliers_k=self.remove_outliers_k,
                                                                        spline=self.use_spline,
                                                                        douglas_peucker_tolerance=douglas_peucker_tolerance,
                                                                        smooth_iter=smooth_iter, vis=True)
                    if len(label_to_contour_df[label]) > 1:
                        polygon = MultiPolygon([Polygon(p) for p in label_to_contour_df[label]])
                    else:
                        polygon = Polygon(label_to_contour_df[label][0])
                    points = []
                    for anchor, anchor_label in zip(self.low_dim_anchors, self.intra_class_anchors_labels):
                        if anchor_label != label:
                            continue
                        p = Point(anchor)
                        if polygon.contains(p) or polygon.intersects(p):
                            points.append(anchor)
                    # handle regions with one or two points
                    if len(points) == 0:
                        self.logger.debug(
                            f'all points of label {label} are outside the concave hull -> no Voroni skipping')
                        continue
                    if len(points) < 3:
                        for i in range(len(points)):
                            point = points[i]
                            points.append([point[0] + 0.01, point[1]])
                            points.append([point[0] - 0.01, point[1]])
                            points.append([point[0], point[1] + 0.01])
                            points.append([point[0], point[1] - 0.01])
                    points = np.array(points)
                    vor = Voronoi(points)

                    regions, vertices = ClusterPlot._voronoi_finite_polygons_2d(vor)

                    mask = polygon
                    for region in regions:
                        polygon = vertices[region]
                        shape = list(polygon.shape)
                        shape[0] += 1
                        p = Polygon(np.append(polygon, polygon[0]).reshape(*shape)).intersection(mask)
                        # intersection can create MultiPolygon
                        if isinstance(p, shapely.geometry.polygon.Polygon):
                            polygons = [p]
                        else:  # Multipolygon
                            polygons = list(p)
                        for p in list(polygons):
                            # skip anchors outside the polygon due to simplification
                            if p.exterior is None:
                                self.logger.debug(f'Skipping voronoi of anchor of label {label} since it is outside '
                                                  f'of polygon after simplification')
                                continue
                            x, y = p.exterior.coords.xy
                            inner_blobs = list(zip(x, y))
                            # plot patch
                            line_cmde = [Path.MOVETO] + [Path.LINETO] * (len(inner_blobs) - 2) + [Path.CLOSEPOLY]
                            path = Path(inner_blobs, line_cmde)
                            patch = patches.PathPatch(path, facecolor=None, linewidth=5, edgecolor=c, fill=False)
                            ax.add_patch(patch)

            if annotate_images:
                palette = itertools.cycle(sns.color_palette())
                for outlier_label, anchor in enumerate(self.overlapped_anchor_per_label):
                    if anchor is None:
                        self.logger.info(f'Missing outlier for label {outlier_label}')
                        continue
                    c = next(palette)
                    outlier = self.cand_samples_to_plot[anchor]['outlier']
                    outlier_is_valid = self.cand_samples_to_plot[anchor]['outlier_is_valid']

                    src_label = self.class_to_label[self.cand_samples_to_plot[anchor]['src_label']]
                    target_label = self.class_to_label[self.cand_samples_to_plot[anchor]['target_label']]
                    if outlier_is_valid:
                        self.logger.info(
                            f'anchor {anchor} outlier {outlier} src_label {src_label} target_label {target_label}, path: '
                            f'\n{self.orig_images[outlier] if self.orig_images else outlier}')
                        sns.scatterplot(x=[self.low_dim_anchors[anchor][0]], y=[self.low_dim_anchors[anchor][1]], c=[c],
                                        s=480)
                        # im = Image.open(self.orig_images[outlier])
                        #
                        # imagebox = offsetbox.AnnotationBbox(
                        #     offsetbox.OffsetImage(im, zoom=0.5),
                        #     self.low_dim_anchors[anchor], bboxprops=dict(edgecolor=c))
                        # ax.add_artist(imagebox)

                    else:
                        self.logger.info(
                            f'anchor {anchor} outlier {outlier} src_label {src_label} target_label {target_label} is not valid')

                # for anchor in self.pure_anchor_per_label:
                #     pure = self.cand_samples_to_plot[anchor]['pure']
                #     im = Image.open(self.orig_images[pure])
                #
                #     imagebox = offsetbox.AnnotationBbox(
                #         offsetbox.OffsetImage(im, zoom=0.3),
                #         self.low_dim_anchors[anchor], bboxprops=dict(edgecolor='green'))
                #     ax.add_artist(imagebox)

        # Modify legend
        num_labels = df[self.label_col].nunique()
        current_handles, current_labels = plt.gca().get_legend_handles_labels()
        if show_blobs:
            if not annotate_images:
                lgd = plt.legend(current_handles[:num_labels + 1], current_labels[:num_labels + 1],
                                 bbox_to_anchor=(1.05, 1), loc=2, fontsize=24)
            else:
                handler_map = dict()
                for pure_label, anchors in enumerate(self.pure_anchor_per_label):
                    if not anchors:
                        self.logger.info(f'Missing pure for label {pure_label}')
                        continue
                    for anchor in anchors:
                        pure = self.cand_samples_to_plot[anchor]['pure']
                        src_label = self.cand_samples_to_plot[anchor]['src_label']
                        if self.orig_images:
                            # probably doesn't work now
                            img_path = self.orig_images[pure]
                            self.logger.info(f'src_label: {src_label} pure path: \n{img_path}')
                            handler_map[current_handles[src_label + 1]] = HandlerLineImage(img_path)
                        else:
                            self.logger.info(f'src_label: {src_label} pure path: \n{pure}')

                if self.orig_images:
                    lgd = plt.legend(current_handles[1:num_labels + 1], current_labels[1:num_labels + 1],
                                     handler_map=handler_map,
                                     handlelength=2, labelspacing=1.4, fontsize=36, borderpad=0.5,
                                     bbox_to_anchor=(1.05, 1), loc=2,
                                     handletextpad=0.5, borderaxespad=0.4)
                else:
                    lgd = plt.legend(current_handles[:num_labels + 1], current_labels[:num_labels + 1],
                                     bbox_to_anchor=(1.05, 1), loc=2, fontsize=24)
        else:
            lgd = plt.legend(current_handles[:num_labels + 1], current_labels[:num_labels + 1],
                             bbox_to_anchor=(1.05, 1), loc=2, fontsize=24)

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        # remove ticks
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.yaxis.set_major_locator(plt.NullLocator())
        # remove labels
        ax.set_xlabel("")
        ax.set_ylabel("")

        if self.save_fig:
            # Save just the portion _inside_ the second axis's boundaries
            fig.savefig(
                f'{self.output_dir}/iter_{i}_points_anchors_patches{"_inner_blobs" if show_inner_blobs else ""}_plot.png',
                bbox_extra_artists=(lgd,), bbox_inches='tight')
            fig.savefig(
                f'{self.output_dir}/iter_{i}_points_anchors_patches{"_inner_blobs" if show_inner_blobs else ""}_plot_no_legend.png')
            # Save just the portion _inside_ the second axis's boundaries
            extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            fig.savefig(
                f'{self.output_dir}/iter_{i}_points_anchors_patches{"_inner_blobs" if show_inner_blobs else ""}_plot_no_legend.png',
                bbox_inches=extent.expanded(0.95, 1))

        if self.show_fig:
            plt.show()

    def _get_contour_df(self):
        """
        Add points with small margin before concave hull calculation to avoid anchors on the edges
        :return: pandas Dataframe with all points
        """
        low_dim_arr = self.low_dim_anchors if not self.reduce_all_points else self.low_dim_points
        x_plus_arr = low_dim_arr.copy()
        x_plus_arr[:, 0] = low_dim_arr[:, 0] + (self.anchors_radius if self.radius_q is not None else 0.01)
        x_minus_arr = low_dim_arr.copy()
        x_minus_arr[:, 0] = low_dim_arr[:, 0] - (self.anchors_radius if self.radius_q is not None else 0.01)
        y_plus_arr = low_dim_arr.copy()
        y_plus_arr[:, 1] = low_dim_arr[:, 1] + (self.anchors_radius if self.radius_q is not None else 0.01)
        y_minus_arr = low_dim_arr.copy()
        y_minus_arr[:, 1] = low_dim_arr[:, 1] - (self.anchors_radius if self.radius_q is not None else 0.01)
        if self.n_components > 2:
            z_plus_arr = low_dim_arr.copy()
            z_plus_arr[:, 2] = low_dim_arr[:, 2] + (self.anchors_radius if self.radius_q is not None else 0.01)
            z_minus_arr = low_dim_arr.copy()
            z_minus_arr[:, 2] = low_dim_arr[:, 2] - (self.anchors_radius if self.radius_q is not None else 0.01)
        anchors_radius = np.concatenate([low_dim_arr, x_plus_arr, x_minus_arr, y_plus_arr, y_minus_arr])
        if self.n_components > 2:
            anchors_radius = np.concatenate([anchors_radius, z_plus_arr, z_minus_arr])
        n_points_per_anchor = 1 + self.n_components * 2
        labels = []
        for i in range(n_points_per_anchor):
            if not self.reduce_all_points:
                labels.extend(self.intra_class_anchors_labels)
            else:
                labels.extend(self.y_with_centroids)
        anchors_df = pd.DataFrame(anchors_radius,
                                  columns=[self.x_col, self.y_col, 'z'] if self.n_components > 2 else [self.x_col,
                                                                                                       self.y_col])
        anchors_df[self.label_col] = labels
        return anchors_df

    @staticmethod
    def _smooth_poly_Douglas_Peucker(poly, douglas_peucker_tolerance):
        _poly = Polygon(poly)
        _poly = _poly.simplify(douglas_peucker_tolerance, preserve_topology=True)
        x, y = _poly.exterior.coords.xy
        return list(zip(x, y))

    @staticmethod
    def _smooth_poly_Chaikins_corner_cutting_iter(poly, iteration=1):
        new_poly = poly[:]
        for i in range(iteration):
            new_poly = ClusterPlot._smooth_poly_Chaikins_corner_cutting(new_poly, True)
        return new_poly

    @staticmethod
    def _smooth_poly_Chaikins_corner_cutting(poly, append_first_point):
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

    def _get_concave_hull(self, points, alpha, remove_outliers_k=None, spline=False, vis=False,
                          douglas_peucker_tolerance=0.6, smooth_iter=13):
        """
        Calculate concave hull of points
        :param alpha: alpha for alphashape algorithm
        :param remove_outliers_k: k for LOF algorithm while removing outliers
        :param spline: use spline - not recommended
        :param vis: if vis is False skipping smothing operations
        :param douglas_peucker_tolerance: tolerance for the douglas_peucker_tolerance smoothing algo
        :param smooth_iter: how many smothing iterations to do with cutting edge corner algorithm
        :return: list of smoothed shapes of concave hulls
        """
        if remove_outliers_k is not None:
            self.logger.debug('removing outliers', remove_outliers_k, alpha)
            clf = LocalOutlierFactor(n_neighbors=remove_outliers_k, contamination='auto')
            is_outlier = clf.fit_predict(points)
            is_outlier = is_outlier == -1
            self.logger.debug('before', points.shape)
            points = points[~is_outlier]
            self.logger.debug('after', points.shape)
        alpha_shape = alphashape.alphashape(points.tolist(), alpha)
        smooth_shapes = []
        if isinstance(alpha_shape, shapely.geometry.polygon.Polygon):
            alpha_shape = [alpha_shape]
        else:  # Multipolygon
            alpha_shape = list(alpha_shape)
        for shape in list(alpha_shape):
            x, y = shape.exterior.coords.xy
            if not spline:
                if vis:
                    smooth_shape = np.array(ClusterPlot._smooth_poly_Chaikins_corner_cutting_iter(
                        ClusterPlot._smooth_poly_Douglas_Peucker(list(zip(x, y)), douglas_peucker_tolerance),
                        iteration=smooth_iter))
                else:
                    smooth_shape = np.array(list(zip(x, y)))
            else:
                tck, u = splprep([np.array(x), np.array(y)], s=3)
                new_points = splev(u, tck)
                x, y = new_points[0], new_points[1]
                smooth_shape = np.array(list(zip(x, y)))
            smooth_shapes.append(smooth_shape)
        return smooth_shapes
