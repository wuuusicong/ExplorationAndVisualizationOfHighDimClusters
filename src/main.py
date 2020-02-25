import pandas as pd
from AMAP import AMAP
from DataSetFactory import DataSetFactory
import imageio
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# ds = DataSetFactory.get_dataset('fists_no_overlap')
# ds = DataSetFactory.get_dataset('cross')
ds = DataSetFactory.get_dataset('simple_overlap')
# ds = DataSetFactory.get_dataset('dense_in_sparse')
# ds = DataSetFactory.get_dataset('MNIST')
# ds = DataSetFactory.get_dataset('MNIST64')
# ds = DataSetFactory.get_dataset('MNIST64_038')
# ds = DataSetFactory.get_dataset('MNIST_038_PCA32')
# ds = DataSetFactory.get_dataset('MNIST_038_PCA16')
# ds.df = ds.df.sample(frac=0.10)
n_iter = 1
amap = AMAP(verbose=True, n_intra_anchors=4, k=20, dim_reduction_algo='umap', supervised=False, anchors_method='kmeans', radius_q=0.5, do_relaxation=True, learning_rate=0.1,
            n_iter=n_iter, batch_size=0, top_greedy=1, self_relation=False, loss='sse', random_state=42,
            class_to_label=[0, 3, 8])
low_dim = amap.fit_transform(ds.df[ds.feature_cols].values, ds.df[ds.label_col].values)

gif_path = './relaxation_images/animation.gif'
frames_path = ""
with imageio.get_writer(gif_path, mode='I') as writer:
    for i in range(n_iter):
        for j in range(5):
            writer.append_data(imageio.imread(f'./relaxation_images/iter{i}.png'))

