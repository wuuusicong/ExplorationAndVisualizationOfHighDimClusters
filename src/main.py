from DataSetFactory import DataSetFactory
from AMAP import AMAP
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

def vis_2d(_df, _x, _y, _color, _algo):
    fig = px.scatter(_df, x=_x, y=_y, color=_color)
    fig.update_layout(title=f'{_algo}: 2D Visualization')
    fig.show()

fists_ds = DataSetFactory.get_dataset('cross')#('fists_no_overlap')
amap = AMAP()
low_dim = amap.fit_transform(fists_ds.df[fists_ds.feature_cols].values, fists_ds.df[fists_ds.label_col].values)

df = pd.DataFrame(data=low_dim, columns=['x', 'y'])
df['label'] = (list(amap.inter_class_anchors_labels) + list(amap.intra_class_anchors_labels))
vis_2d(df, 'x', 'y', 'label', '')

