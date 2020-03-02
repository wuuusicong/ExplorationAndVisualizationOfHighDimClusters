import pandas as pd
import numpy as np
from AMAP import AMAP
from DataSetFactory import DataSetFactory
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

def vis_2d(_df, _x, _y, _color, _algo):
    fig = px.scatter(_df, x=_x, y=_y, color=_color)
    fig.update_layout(title=f'{_algo}: 2D Visualization')
    fig.show()

# ds = DataSetFactory.get_dataset('fists_no_overlap')
# ds = DataSetFactory.get_dataset('cross')
# ds = DataSetFactory.get_dataset('simple_overlap')
# ds = DataSetFactory.get_dataset('dense_in_sparse')
# ds = DataSetFactory.get_dataset('MNIST')
# ds = DataSetFactory.get_dataset('MNIST64')
# ds = DataSetFactory.get_dataset('MNIST64_038')
ds = DataSetFactory.get_dataset('hourglass')
# ds.df = ds.df.sample(frac=0.10)

n_iter = 1
amap = AMAP(is_plotly=False, learning_rate=0.1, n_iter=2)
print(amap)
low_dim = amap.fit_transform(ds.df[ds.feature_cols].values, ds.df[ds.label_col].values)
