import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from umap import UMAP
from sklearn.manifold import MDS

from numba.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning, NumbaPerformanceWarning
import warnings

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPerformanceWarning)

from DataSetFactory import DataSetFactory
from DataSetFactory import DataSet
from AMAP import AMAP

RANDOM_STATE = 42

# fashion_mnist_dim = 64  # 64 or 784
# df_train = pd.read_csv(f'../data/FashionMNIST/train{fashion_mnist_dim if fashion_mnist_dim == 64 else ""}.csv')
# feature_cols = [f'Pixel{i:03}' for i in range(fashion_mnist_dim)]
# label_col = 'Category'
# ds = DataSet(df=df_train, feature_cols=feature_cols, label_col = label_col)
#
# # Normalize
# if fashion_mnist_dim == 784:
#     ds.df[ds.feature_cols] = ds.df[ds.feature_cols] / 255
#
# # Take only sample for now
# ds.df = ds.df.sample(frac=0.15, random_state=RANDOM_STATE)
#
#
# # Keep Only 3 labels: 'Ankle boot', 'Sneaker', 'Sandal', 'Trouser',
# PART_LABELS = True
# if PART_LABELS:
#     ds.df = ds.df[ds.df[ds.label_col].isin([1, 5, 7, 9])]
#     ds.df[ds.label_col] = ds.df[ds.label_col].replace({1: 0, 5: 1, 7: 2, 9: 3})
#
# show_images = False
# if show_images:
#     for i in range(len(ds.df)):
#         im = ds.df[ds.feature_cols].iloc[i].values.reshape(28,28)
#         plt.imshow(im, cmap='gray')
#         plt.show()
#
# classes = [
#     'T-shirt/top',  # 0
#     'Trouser',      # 1
#     'Pullover',     # 2
#     'Dress',        # 3
#     'Coat',         # 4
#     'Sandal',       # 5
#     'Shirt',        # 6
#     'Sneaker',      # 7
#     'Bag',          # 8
#     'Ankle boot']   # 9
#
# if PART_LABELS:
#     classes = [
#         'Trouser',      # 1
#         'Sandal',       # 5
#         'Sneaker',      # 7
#         'Ankle boot'    # 9
#     ]
#
# class_to_label = {0: 'Trouser', 1: 'Sandal', 2: 'Sneaker', 3: 'Ankle boot'}
#
ds = DataSetFactory.get_dataset('FashionMNIST64', random_state=RANDOM_STATE, sample=0.15, is_subset=True)

X = ds.df[ds.feature_cols].values
y = ds.df[ds.label_col].values


n_iter = 1
batch_size = 0
birch_threshold = 0.8
# birch_branching_factor = 2
uniform_points_per = 'label'

amap = AMAP(learning_rate=0.5, n_iter=n_iter, batch_size=batch_size,
            anchors_method='birch', birch_threshold=birch_threshold, #birch_branching_factor=birch_branching_factor,
            umap_n_neighbors=5,
            dataset='FashionMNIST_64_15_perc',
            # class_to_label=class_to_label,
            class_to_label=ds.class_to_label,
            radius_q=None,
            uniform_points_per=uniform_points_per,
            show_fig=False,
            k=40,
            loss='Linf',
            random_state=RANDOM_STATE,
            do_animation=False,
            save_fig_every=500,
            magnitude_step=True,
            only_inter_relations=True,
            top_greedy=10,
            show_anchors=True,
            supervised=True)
            # reduce_all_points=True,
            # show_points=True,
            # show_polygons=False,
            # supervised=True)
print(amap)
low_dim = amap.fit_transform(X, y)
print(amap.num_clusters_each_label)
#
#
#
#
#
#
#
#








#
#
# import pandas as pd
# import numpy as np
# from AMAP import AMAP
# from DataSetFactory import DataSetFactory
# import plotly.graph_objects as go
# import plotly.express as px
# import seaborn as sns
# import matplotlib.pyplot as plt
# import random
#
# RANDOM_STATE = 47
# if RANDOM_STATE is not None:
#     np.random.seed(RANDOM_STATE)
#     random.seed(RANDOM_STATE)
#
#
# def vis_2d(_df, _x, _y, _color, _algo):
#     fig = px.scatter(_df, x=_x, y=_y, color=_color)
#     fig.update_layout(title=f'{_algo}: 2D Visualization')
#     fig.show()
#
# # ds = DataSetFactory.get_dataset('fists_no_overlap')
# # ds = DataSetFactory.get_dataset('cross')
# # ds = DataSetFactory.get_dataset('simple_overlap')
# # ds = DataSetFactory.get_dataset('dense_in_sparse')
# # ds = DataSetFactory.get_dataset('MNIST')
# # ds = DataSetFactory.get_dataset('MNIST64')
# # ds = DataSetFactory.get_dataset('MNIST64_038')
# ds = DataSetFactory.get_dataset('hourglass')
# # ds.df = ds.df.sample(frac=0.10)
#
# n_iter = 1
# amap = AMAP(is_plotly=False, learning_rate=0.1, n_iter=61, batch_size=1, loss='Linf',magnitude_step=True,
#             only_inter_relations=False, random_state=RANDOM_STATE,save_fig_every=20,do_animation=False,
#             dataset='Hourglass',)
#             # reduce_all_points=True,
#             # show_points=True,
#             # show_polygons=False,
#             # supervised=False)
# # amap = AMAP(learning_rate=0.1, n_iter=n_iter, batch_size=batch_size,
# #             anchors_method='birch', birch_threshold=birch_threshold,
# #             umap_n_neighbors=5,
# #             dataset='FashionMNIST_64_full_only_inter_relations',
# #             class_to_label=class_to_label,
# #             radius_q=None,
# #             uniform_points_per=uniform_points_per,
# #             show_fig=False,
# #             k=40,
# #             loss='Linf',
# #             random_state=RANDOM_STATE,
# #             do_animation=False,
# #             save_fig_every=1,
# #             magnitude_step=True,
# #             only_inter_relations=True)
# #             # reduce_all_points=True,
# #             # show_points=True,
# #             # show_polygons=False,
# #             # supervised=True)
# print(amap)
# low_dim = amap.fit_transform(ds.df[ds.feature_cols].values, ds.df[ds.label_col].values)
