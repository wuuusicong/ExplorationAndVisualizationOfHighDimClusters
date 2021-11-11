import numpy as np
import pandas as pd
import random
import PolygonsFactory as pf

class DataSet:
    """
    Wrapper class for pandas DataFrame in order to keep common interface
    of dataframe feature columns and label column for all the datasets
    """
    def __init__(self, df, feature_cols, label_col, class_to_label: dict = None, orig_images: list = None):
        self.df = df
        self.feature_cols = feature_cols
        self.label_col = label_col
        self.class_to_label = class_to_label
        self.orig_images = orig_images


class DataSetFactory:
    """
    Factory class for datasets
    """

    @staticmethod
    def mnist_ae(dim=None, sample=None, is_subset=False, random_state=None):
        if dim is not None:
            encoded_features = np.load(f'../data/MNIST_AE/mnist_encoded_to_{dim}_dim.npy')
        else:
            encoded_features = np.load(f'../data/MNIST_AE/mnist_orig_dim.npy')
        labels = np.load(f'../data/MNIST_AE/mnist_labels_ae.npy')

        feature_cols = ['pixel' + str(i) for i in range(encoded_features.shape[1])]
        df = pd.DataFrame(encoded_features, columns=feature_cols)
        df['label'] = labels
        df['label'] = df['label'].apply(lambda i: int(i))
        digit_to_label = {i: i for i in df['label'].unique()}
        if is_subset:
            # set_label = [0,3,8,9]
            # df, digit_to_label = DataSetFactory.setMnistTestLabel(df, set_label)
            df = df[df['label'].isin([0, 3, 8, 9])]
            df['label'] = df['label'].replace({0: 0, 3: 1, 8: 2, 9: 3})
            digit_to_label = {0: 0, 1: 3, 2: 8, 3: 9}

        if sample is not None:
            df = df.sample(frac=sample, random_state=random_state)

        class_to_label = {i: f'Digit_{digit_to_label[i]}' for i in df['label'].unique()}

        return DataSet(df, feature_cols, 'label', class_to_label)

    #选定mnist中的labels进行测试,返回label过滤后的数据   暂时没用到
    @staticmethod
    def setMnistTestLabel(df, set_label):
        order_label = [i for i in range(len(set_label))]
        test_label = set_label
        digit_to_label = dict(zip(order_label, test_label))    #{num1:0,num2:1,num3:2...}
        df = df[df['label'].isin(test_label)]
        df['label'] = df['label'].replace(digit_to_label)
        return df, digit_to_label

    @staticmethod
    def get_dataset(dataset_name, random_state=None, sample=None, is_subset=False):
        if random_state is not None:
            np.random.seed(random_state)
            random.seed(random_state)

        if dataset_name == 'MNIST_AE_ORIG':
            return DataSetFactory.mnist_ae(dim=None, sample=sample, is_subset=True, random_state=random_state)
        if dataset_name == 'MNIST_AE32':
            return DataSetFactory.mnist_ae(dim=32, sample=sample, is_subset=True, random_state=random_state)
        if dataset_name == 'MNIST_AE4':
            return DataSetFactory.mnist_ae(dim=4, sample=sample, is_subset=True, random_state=random_state)

        if dataset_name in ['fists_no_overlap', 'cross', 'cross-denses', 'cross7', 'simple_overlap', 'dense_in_sparse', 'hourglass',
                              'hourglass-spike', 'hourglass2']:
            df, feature_cols, label_col = pf.PolygonsFactory.get_polygons(dataset_name)
            return DataSet(df, feature_cols, label_col, class_to_label={label: f'Poly{label}' for label in df[label_col].unique()})
        else:
            raise Exception(f'Unsupported dataset {dataset_name}')
