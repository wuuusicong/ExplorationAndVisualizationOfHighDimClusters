import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import joblib
from sklearn.metrics import pairwise_distances
from PIL import Image

from DataSetFactory import DataSetFactory
from ClusterPlot import ClusterPlot

RANDOM_STATE = 42

ds = DataSetFactory.get_dataset('vgg_features_imagenet', RANDOM_STATE ,sample=0.15)

X = ds.df[ds.feature_cols].values
y = ds.df[ds.label_col].values

cp = ClusterPlot(learning_rate=0.5,
                   n_iter=1,
                   batch_size=0,
                   anchors_method='birch',
                   birch_threshold=35,
                   umap_n_neighbors=15,
                   dataset='default',
                   class_to_label=ds.class_to_label,
                   radius_q=None,
                   uniform_points_per='anchor',
                   show_fig=False,
                   save_fig=True,
                   k=20,
                   loss='Linf',
                   random_state=RANDOM_STATE,
                   do_animation=False,
                   save_fig_every=500,
                   magnitude_step=True,
                   only_inter_relations=True,
                   top_greedy=1,
                   show_anchors=False,
                   supervised=False,
                   # alpha=1,
                   alpha=0.5,
                   douglas_peucker_tolerance=1,
                   smooth_iter=3,
                   show_relations=True,
                   show_inner_blobs=True,
                   random_points_method='voronoi')
print(cp)


low_dim = cp.fit_transform(X, y)


