import pandas as pd
import PolygonsFactory as pf
from sklearn import datasets

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
    def get_dataset(dataset_name):
        if dataset_name == 'MNIST':
            return DataSetFactory.mnist()
        if dataset_name == 'MNIST64':
            return DataSetFactory.mnist64()
        elif dataset_name == 'fists_no_overlap':
            df, feature_cols, label_col = pf.PolygonsFactory.get_polygons('fists_no_overlap')
            return DataSet(df, feature_cols, label_col)
        elif dataset_name == 'cross':
            df, feature_cols, label_col = pf.PolygonsFactory.get_polygons('cross')
            return DataSet(df, feature_cols, label_col)
        else:
            raise Exception(f'Unsupported dataset {dataset_name}')
