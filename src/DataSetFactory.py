import pandas as pd
import PolygonsFactory as pf
from sklearn import datasets
from sklearn.decomposition import PCA
import numpy as np
import random
import matplotlib.pyplot as plt

class DataSet:
    """
    Wrapper class for pandas DataFrame in order to keep common interface
    of dataframe feature columns and label column for all the datasets
    """
    def __init__(self, df, feature_cols, label_col, class_to_label: dict = None):
        self.df = df
        self.feature_cols = feature_cols
        self.label_col = label_col
        self.class_to_label = class_to_label


class DataSetFactory:
    """
    Factory class for datasets
    """
    @staticmethod
    def mnist():
        """
        Get MNIST dataset
        :return: DataSet Object which holds MNIST data
        """
        X, y = datasets.fetch_openml('mnist_784', version=1, return_X_y=True)
        X = X / 255.
        feature_cols = ['pixel' + str(i) for i in range(X.shape[1])]
        df = pd.DataFrame(X, columns=feature_cols)
        df['y'] = y
        df['label'] = df['y'].apply(lambda i: str(i))
        return DataSet(df, feature_cols, 'y')

    @staticmethod
    def mnist64():
        """
        Get MNIST dataset
        :return: DataSet Object which holds MNIST data
        """
        X, y = datasets.load_digits(return_X_y=True)
        X = X / 255.
        feature_cols = ['pixel' + str(i) for i in range(X.shape[1])]
        df = pd.DataFrame(X, columns=feature_cols)
        df['y'] = y
        df['label'] = df['y'].apply(lambda i: str(i))
        return DataSet(df, feature_cols, 'y')

    @staticmethod
    def mnist64_038():
        """
        Get MNIST dataset of digits 0, 3 and 8
        :return: DataSet Object which holds MNIST data
        """
        ds = DataSetFactory.mnist64()
        ds.df = ds.df[ds.df['y'].isin([0,3,8])]
        ds.df['y'] = ds.df['y'].replace({0: 0, 3: 1, 8: 2})
        return ds

    @staticmethod
    def mnist_038_pca_d(d=32):
        """
        Get MNIST dataset of digits 0, 3 and 8
        :return: DataSet Object which holds MNIST data
        """
        ds = DataSetFactory.mnist()
        ds.df = ds.df[ds.df['y'].isin(['0', '3', '8'])]
        ds.df['y'] = ds.df['y'].replace({'0': 0, '3': 1, '8': 2})
        pca = PCA(n_components=d)
        low_dim = pca.fit_transform(ds.df[ds.feature_cols])
        new_features = [f'{i}' for i in range(d)]
        new_df = pd.DataFrame(low_dim, columns=new_features)
        new_df['y'] = ds.df['y'].values
        return DataSet(new_df, new_features, 'y')

    @staticmethod
    def fashion_mnist(dataset_name, random_state, sample, is_subset, show_images=False):
        if dataset_name == 'FashionMNIST64':
            fashion_mnist_dim = 64  # 64 or 784
        elif dataset_name == 'FashionMNIST':
            fashion_mnist_dim = 784
        else:
            raise Exception(f'Unsupported Fasion MNIST Dataset {dataset_name}')

        df_train = pd.read_csv(f'../data/FashionMNIST/train{fashion_mnist_dim if fashion_mnist_dim == 64 else ""}.csv')
        feature_cols = [f'Pixel{i:03}' for i in range(fashion_mnist_dim)]
        label_col = 'Category'
        ds = DataSet(df=df_train, feature_cols=feature_cols, label_col=label_col)

        # Normalize
        if fashion_mnist_dim == 784:
            ds.df[ds.feature_cols] = ds.df[ds.feature_cols] / 255

        # Take only sample for now
        if sample is not None:
            print(f'Taking sample of {sample} from the data')
            ds.df = ds.df.sample(frac=sample, random_state=random_state)

        if is_subset:
            # Keep Only 3 labels: 'Ankle boot', 'Sneaker', 'Sandal', 'Trouser',
            ds.df = ds.df[ds.df[ds.label_col].isin([1, 5, 7, 9])]
            ds.df[ds.label_col] = ds.df[ds.label_col].replace({1: 0, 5: 1, 7: 2, 9: 3})

        if show_images:
            for i in range(len(ds.df)):
                im = ds.df[ds.feature_cols].iloc[i].values.reshape(28, 28)
                plt.imshow(im, cmap='gray')
                plt.show()

        class_names=[
                'T-shirt/top',  # 0
                'Trouser',  # 1
                'Pullover',  # 2
                'Dress',  # 3
                'Coat',  # 4
                'Sandal',  # 5
                'Shirt',  # 6
                'Sneaker',  # 7
                'Bag',  # 8
                'Ankle boot']  # 9

        if is_subset:
            class_names = [
                'Trouser',  # 1
                'Sandal',  # 5
                'Sneaker',  # 7
                'Ankle boot'  # 9
            ]

        ds.class_to_label = {i: class_names[i] for i in range(len(class_names))}

        return ds

    @staticmethod
    def get_dataset(dataset_name, random_state=None, sample=None, is_subset=False):
        if random_state is not None:
            np.random.seed(random_state)
            random.seed(random_state)

        if dataset_name == 'MNIST':
            return DataSetFactory.mnist()
        if dataset_name == 'MNIST64':
            return DataSetFactory.mnist64()
        if dataset_name == 'MNIST64_038':
            return DataSetFactory.mnist64_038()
        if dataset_name == 'MNIST_038_PCA32':
            return DataSetFactory.mnist_038_pca_d()
        if dataset_name == 'MNIST_038_PCA16':
            return DataSetFactory.mnist_038_pca_d(16)
        if dataset_name in ['fists_no_overlap', 'cross', 'simple_overlap', 'dense_in_sparse', 'hourglass',
                              'hourglass-spike']:
            df, feature_cols, label_col = pf.PolygonsFactory.get_polygons(dataset_name)
            return DataSet(df, feature_cols, label_col)
        if dataset_name in ['FashionMNIST', 'FashionMNIST64']:
            return DataSetFactory.fashion_mnist(dataset_name, random_state, sample, is_subset)
        else:
            raise Exception(f'Unsupported dataset {dataset_name}')
