import pandas as pd
import PolygonsFactory as pf
from sklearn import datasets
from sklearn.decomposition import PCA

class DataSet:
    """
    Wrapper class for pandas DataFrame in order to keep common interface
    of dataframe feature columns and label column for all the datasets
    """
    def __init__(self, df, feature_cols, label_col):
        self.df = df
        self.feature_cols = feature_cols
        self.label_col = label_col


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
    def get_dataset(dataset_name):
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
        elif dataset_name in ['fists_no_overlap', 'cross', 'simple_overlap', 'dense_in_sparse', 'hourglass']:
            df, feature_cols, label_col = pf.PolygonsFactory.get_polygons(dataset_name)
            return DataSet(df, feature_cols, label_col)
        else:
            raise Exception(f'Unsupported dataset {dataset_name}')
