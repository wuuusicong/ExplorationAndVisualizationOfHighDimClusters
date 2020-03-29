import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from umap import UMAP
from sklearn.manifold import MDS
from skimage.transform import resize

from numba.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning, NumbaPerformanceWarning
import warnings

from tqdm import tqdm, tqdm_pandas
tqdm.pandas()

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPerformanceWarning)

from DataSetFactory import DataSet
from AMAP import AMAP

df = pd.read_csv('../../data/FashionMNIST/train.csv')
feature_cols = [f'Pixel{i:03}' for i in range(784)]
label_col = 'Category'

# images = []
# for i in tqdm(range(len(df))):
#         im = resize(df[feature_cols].iloc[i].values.reshape(28,28), output_shape=(8,8)).flatten()
#         images.append(im)
# #         plt.imshow(im, cmap='gray')
# #         plt.show()
# #         if i == 10:
# #             break
# images_arr = np.array(images)
# resize_df = pd.DataFrame(images_arr)
# resize_df[label_col] = df[label_col]

images = df[feature_cols].progress_apply(lambda x: resize(x.values.reshape(28,28), output_shape=(8,8)).flatten(), axis=1)