import pandas as pd
import PolygonsFactory as pf
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
import random
import matplotlib.pyplot as plt
import joblib

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
    def mnist(sample, is_subset=False, random_state=None):
        """
        Get MNIST dataset
        :return: DataSet Object which holds MNIST data
        """
        X, y = datasets.fetch_openml('mnist_784', version=1, return_X_y=True,
                                     data_home='C:\\Users\\omalkai\\Google Drive\\Thesis\\Dani Cohen Or\\Code\ExplorationAndVisualizationOfHighDimClusters\\data')
        X = X / 255.
        feature_cols = ['pixel' + str(i) for i in range(X.shape[1])]
        df = pd.DataFrame(X, columns=feature_cols)
        df['y'] = y
        df['label'] = df['y'].apply(lambda i: int(i))
        digit_to_label = {i: i for i in df['label'].unique()}
        if is_subset:
            df = df[df['label'].isin([0, 3, 8, 9])]
            df['label'] = df['label'].replace({0: 0, 3: 1, 8: 2, 9:3})
            digit_to_label = {0:0, 1:3, 2:8, 3:9}

        if sample is not None:
            df = df.sample(frac=sample, random_state=random_state)

        class_to_label = {i: f'Digit_{digit_to_label[i]}' for i in df['label'].unique()}

        return DataSet(df, feature_cols, 'label', class_to_label)

    @staticmethod
    def cover_type(sample, random_state=None):
        """
        Get Covtype dataset
        :return: DataSet Object which holds MNIST data
        """
        X, y = datasets.fetch_covtype(return_X_y=True, data_home='C:\\Users\\omalkai\\Google Drive\\Thesis\\Dani Cohen Or\\Code\ExplorationAndVisualizationOfHighDimClusters\\data')
        feature_cols = [
    'Elevation',
    'Aspect',
    'Slope',
    'Horizontal_Distance_To_Hydrology',
    'Vertical_Distance_To_Hydrology',
    'Horizontal_Distance_To_Roadways',
    'Hillshade_9am',
    'Hillshade_Noon',
    'Hillshade_3pm',
    'Horizontal_Distance_To_Fire_Points',
    'Wilderness_Area1',
    'Wilderness_Area2',
    'Wilderness_Area3',
    'Wilderness_Area4',
    'Soil_Type1',
    'Soil_Type2',
    'Soil_Type3',
    'Soil_Type4',
    'Soil_Type5',
    'Soil_Type6',
    'Soil_Type7',
    'Soil_Type8',
    'Soil_Type9',
    'Soil_Type10',
    'Soil_Type11',
    'Soil_Type12',
    'Soil_Type13',
    'Soil_Type14',
    'Soil_Type15',
    'Soil_Type16',
    'Soil_Type17',
    'Soil_Type18',
    'Soil_Type19',
    'Soil_Type20',
    'Soil_Type21',
    'Soil_Type22',
    'Soil_Type23',
    'Soil_Type24',
    'Soil_Type25',
    'Soil_Type26',
    'Soil_Type27',
    'Soil_Type28',
    'Soil_Type29',
    'Soil_Type30',
    'Soil_Type31',
    'Soil_Type32',
    'Soil_Type33',
    'Soil_Type34',
    'Soil_Type35',
    'Soil_Type36',
    'Soil_Type37',
    'Soil_Type38',
    'Soil_Type39',
    'Soil_Type40',
]
        # Normalizing Features using min max scaler
        print('Normalizing features using min max scaler')
        X = MinMaxScaler().fit_transform(X)
        df = pd.DataFrame(X, columns=feature_cols)
        df['y'] = y
        df['label'] = df['y'].apply(lambda i: i-1)  # labels start at 1, we need 0

        if sample is not None:
            df = df.sample(frac=sample, random_state=random_state)

        class_to_label = [
            'Spruce / Fir',
            'Lodgepole Pine',
            'Ponderosa Pine',
            'Cottonwood / Willow',
            'Aspen',
            'Douglas - fir',
            'Krummholz',
        ]

        return DataSet(df, feature_cols, 'label', class_to_label={i:class_to_label[i] for i in range(len(class_to_label))})

    @staticmethod
    def wine(sample, random_state=None):
        """
        Get wine dataset
        :return: DataSet Object which holds MNIST data
        """

        df_red = pd.read_csv('../data/wine_quality/winequality-red.csv', sep=';')
        df_red['type'] = 'red'
        df_white = pd.read_csv('../data/wine_quality/winequality-white.csv', sep=';')
        df_white['type'] = 'white'
        df = pd.concat([df_red, df_white])
        quality_col = 'quality'
        label_col = 'label'
        feature_cols = [col for col in df.columns if col != quality_col and col != 'type']
        print('Normalizing features using standard scaler')
        df[feature_cols] = StandardScaler().fit_transform(df[feature_cols].values)
        df[label_col] = df[quality_col].apply(lambda x: x-3)  # quality is from 3 to 8

        if sample is not None:
            df = df.sample(frac=sample, random_state=random_state)

        class_to_label = {i: f'Quality_{i+3}' for i in df[label_col].unique()}

        return DataSet(df, feature_cols, label_col,
                       class_to_label=class_to_label)

    @staticmethod
    def wine_sklearn(sample, scale, random_state=None):
        features, target = datasets.load_wine(return_X_y=True)
        if scale:
            print('using min max scaler')
            df = pd.DataFrame(MinMaxScaler().fit_transform(features))
        else:
            print('skip scaling')
            df = pd.DataFrame(features)
        feature_cols = df.columns
        label_col = 'label'
        df[label_col] = target
        class_to_label = {0: 'class_0', 1: 'class_1', 2: 'class_2'}

        return DataSet(df, feature_cols, label_col, class_to_label)

    @staticmethod
    def shuttle(sample, random_state=None):
        """
        Get shuttle dataset
        :return: DataSet Object which holds MNIST data
        """
        df = pd.read_csv('../data/shuttle/shuttle.trn', sep=' ', header=None)
        num_columns = df.shape[1]

        label_col = num_columns-1
        feature_cols = [col for col in range(num_columns-1)]
        print('Normalizing features using min max scaler')
        df[feature_cols] = MinMaxScaler().fit_transform(df[feature_cols].values)
        df[label_col] = df[label_col].apply(lambda x: x - 1)  # classes are from 1 to 6

        if sample is not None:
            df = df.sample(frac=sample, random_state=random_state)

        class_to_label = {i: f'Class_{i + 1}' for i in df[label_col].unique()}

        return DataSet(df, feature_cols, label_col,
                       class_to_label=class_to_label)

    @staticmethod
    def hapt(sample, is_train, random_state=None):
        """
        Get hapt dataset
        :return: DataSet Object which holds MNIST data
        """
        if is_train:
            X = pd.read_csv('../data/HAPT/Train/X_train.txt', sep=' ', header=None)
            y = pd.read_csv('../data/HAPT/Train/y_train.txt', sep=' ', header=None)
        else:
            X = pd.read_csv('../data/HAPT/Test/X_test.txt', sep=' ', header=None)
            y = pd.read_csv('../data/HAPT/Test/y_test.txt', sep=' ', header=None)
        y.columns = [X.shape[1]]
        df = pd.concat([X,y], axis=1)
        num_columns = df.shape[1]

        label_col = num_columns - 1
        feature_cols = [col for col in range(num_columns - 1)]
        # print('Normalizing features using min max scaler')
        # df[feature_cols] = MinMaxScaler().fit_transform(df[feature_cols].values)
        df[label_col] = df[label_col].apply(lambda x: x - 1)  # classes are from 1 to 6

        print('removing label 10 and 11 to keep only 10 classes to keep sns default pallet')
        df = df[df[label_col] < 10]

        if sample is not None:
            df = df.sample(frac=sample, random_state=random_state)

        class_names = [
            'WALKING',
            'WALKING_UPSTAIRS',
            'WALKING_DOWNSTAIRS',
            'SITTING',
            'STANDING',
            'LAYING',
            'STAND_TO_SIT',
            'SIT_TO_STAND',
            'SIT_TO_LIE',
            'LIE_TO_SIT',
            'STAND_TO_LIE',  # Removed
            'LIE_TO_STAND',  # Removed
        ]
        class_to_label = {i: f'{i}_{class_names[i]}' for i in df[label_col].unique()}

        return DataSet(df, feature_cols, label_col,
                       class_to_label=class_to_label)

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
            df = df[df['label'].isin([0, 3, 8, 9])]
            df['label'] = df['label'].replace({0: 0, 3: 1, 8: 2, 9: 3})
            digit_to_label = {0: 0, 1: 3, 2: 8, 3: 9}

        if sample is not None:
            df = df.sample(frac=sample, random_state=random_state)

        class_to_label = {i: f'Digit_{digit_to_label[i]}' for i in df['label'].unique()}

        return DataSet(df, feature_cols, 'label', class_to_label)

    @staticmethod
    def mnist_deep(dim=None, epoch_acc=None, sample=None, is_subset=False, random_state=None):
        if dim is not None:
            encoded_features = np.load(f'../data/DeepFeaturesMnist/mnist_conv{dim[0]}_conv{dim[1]}.npy')
            labels = np.load(f'../data/DeepFeaturesMnist/mnist_conv{dim[0]}_conv{dim[1]}_label.npy')
            labels = np.argmax(labels, axis=1)  # it is one-hot vectors
        elif epoch_acc is not None:
            encoded_features = np.load(f'../data/DeepFeaturesMnist/mnist_conv32_conv64_{epoch_acc[0]}_epochs_acc_{epoch_acc[1]}.npy')
            labels = np.load(f'../data/DeepFeaturesMnist/mnist_conv32_conv64_label_{epoch_acc[0]}_epochs_{epoch_acc[1]}.npy')
            labels = np.argmax(labels, axis=1)  # it is one-hot vectors
        else:
            encoded_features = np.load(f'../data/MNIST_AE/mnist_orig_dim.npy')
            labels = np.load(f'../data/MNIST_AE/mnist_labels_ae.npy')

        feature_cols = ['pixel' + str(i) for i in range(encoded_features.shape[1])]
        df = pd.DataFrame(encoded_features, columns=feature_cols)
        df['label'] = labels
        df['label'] = df['label'].apply(lambda i: int(i))

        digit_to_label = {i: i for i in df['label'].unique()}
        if is_subset:
            df = df[df['label'].isin([0, 3, 8, 9])]
            df['label'] = df['label'].replace({0: 0, 3: 1, 8: 2, 9: 3})
            digit_to_label = {0: 0, 1: 3, 2: 8, 3: 9}

        if sample is not None:
            df = df.sample(frac=sample, random_state=random_state)

        class_to_label = {i: f'Digit_{digit_to_label[i]}' for i in df['label'].unique()}

        return DataSet(df, feature_cols, 'label', class_to_label)

    @staticmethod
    def fashion_mnist_ae(dim=None, sample=None, is_subset=False, random_state=None):
        if dim is not None:
            encoded_features = np.load(f'../data/FASHION_MNIST_AE/fashion_mnist_encoded_to_{dim}_dim.npy')
        else:
            encoded_features = np.load(f'../data/FASHION_MNIST_AE/fashion_mnist_orig_dim.npy')
        labels = np.load(f'../data/FASHION_MNIST_AE/fashion_mnist_labels_ae.npy')

        feature_cols = ['pixel' + str(i) for i in range(encoded_features.shape[1])]
        df = pd.DataFrame(encoded_features, columns=feature_cols)
        df['label'] = labels
        df['label'] = df['label'].apply(lambda i: int(i))

        label_to_class = {i: i for i in df['label'].unique()}
        if is_subset:
            df = df[df['label'].isin([0, 3, 8, 9])]
            df['label'] = df['label'].replace({0: 0, 3: 1, 8: 2, 9: 3})
            label_to_class = {0: 0, 1: 3, 2: 8, 3: 9}

        if sample is not None:
            df = df.sample(frac=sample, random_state=random_state)

        class_names = [
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

        class_to_label = {i: f'{i}_{class_names[i]}' for i in df['label'].unique()}

        return DataSet(df, feature_cols, 'label', class_to_label)

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

        # backward compatability
        if is_subset == True:
            is_subset = [1,5,7,9]

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
            # ds.df = ds.df[ds.df[ds.label_col].isin([1, 5, 7, 9])]
            # ds.df[ds.label_col] = ds.df[ds.label_col].replace({1: 0, 5: 1, 7: 2, 9: 3})
            ds.df = ds.df[ds.df[ds.label_col].isin(is_subset)]
            ds.df[ds.label_col] = ds.df[ds.label_col].replace({orig_label: new_label for new_label, orig_label in
                                                               enumerate(is_subset)})

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
            # class_names = [
            #     'Trouser',  # 1
            #     'Sandal',  # 5
            #     'Sneaker',  # 7
            #     'Ankle boot'  # 9
            # ]
            class_names = [class_names[i] for i in range(len(class_names)) if i in is_subset]

        ds.class_to_label = {i: class_names[i] for i in range(len(class_names))}

        return ds

    @staticmethod
    def get_deep_features_imagenet(net, random_state, sample, is_subset):
        X = joblib.load(f'../data/ImageNetCropped/{net}_animals_embeddings.pickle')
        y = joblib.load(f'../data/ImageNetCropped/{net}_animals_labels.pickle')
        df = pd.DataFrame(X)
        feature_cols = [f'Feat{i:03}' for i in range(X.shape[1])]
        df.columns = feature_cols
        label_col = 'Label'
        orig_image_col = 'orig_image'
        df[label_col] = y

        class_names = [
            'Norwegian_elkhound',   # n02091467
            'golden_retriever',     # n02099601
            'Maltese_dog',          # n02085936
            'Scottish_deerhound',   # n02092002
            'Bedlington_terrier',   # n02093647
            'Blenheim_spaniel',     # n02086646
            'Boston_bull',          # n02096585
            'toy_terrier',          # n02087046
            'Rhodesian_ridgeback',  # n02087394
            'Afghan_hound'          # n02088094
        ]
        class_to_label = {i: class_names[i] for i in range(len(class_names))}

        with open(f'../data/ImageNetCropped/{net}_data_new_format.txt', 'r') as f:
            lines = f.readlines()
            images_path = []
            for line in lines:
                # remove prefix of class
                # line = line.split('/')[1]
                # # Split to class dir and image path
                # class_dir, image_path = line.split('_', maxsplit=1)
                # remove new line char
                # image_path = image_path[:-1]
                # new_path = f'../data/ImageNetCropped/animals_faces_dataset/{class_dir}/{class_dir}_{image_path}'
                # images_path.append(new_path)
                images_path.append(line[:-1])  # remove new line character

        # add to columns in the dataframe to take the correct list after sampling
        print(df.shape, len(images_path))
        df[orig_image_col] = images_path

        ds = DataSet(df=df, feature_cols=feature_cols, label_col=label_col, class_to_label=class_to_label,
                     orig_images=images_path)

        if is_subset:
            # Keep Only 3 labels: 'Ankle boot', 'Sneaker', 'Sandal', 'Trouser',
            # ds.df = ds.df[ds.df[ds.label_col].isin([1, 5, 7, 9])]
            # ds.df[ds.label_col] = ds.df[ds.label_col].replace({1: 0, 5: 1, 7: 2, 9: 3})
            ds.df = ds.df[ds.df[ds.label_col].isin(is_subset)]
            ds.df[ds.label_col] = ds.df[ds.label_col].replace({orig_label: new_label for new_label, orig_label in
                                                               enumerate(is_subset)})
            ds.class_to_label = {new_label: ds.class_to_label[old_label] for new_label, old_label in enumerate(is_subset)}

        # Take only sample for now
        if sample is not None:
            print(f'Taking sample of {sample} from the data')
            ds.df = ds.df.sample(frac=sample, random_state=random_state)
            ds.orig_images = ds.df[orig_image_col].tolist()

        return ds

    @staticmethod
    def get_deep_features_imagenet_cats(net, random_state, sample, is_subset):
        X = joblib.load(f'../data/ImageNetCropped/{net}_cats_embeddings.pickle')
        y = joblib.load(f'../data/ImageNetCropped/{net}_cats_labels.pickle')
        df = pd.DataFrame(X)
        feature_cols = [f'Feat{i:03}' for i in range(X.shape[1])]
        df.columns = feature_cols
        label_col = 'Label'
        orig_image_col = 'orig_image'
        df[label_col] = y

        class_names = [
            'tabby',  # n02123045
            'Persian_cat',  # n02123394
            'Siamese_cat',  # n02123597
            'leopard',  # n02128385
            'snow_leopard',  # n02128757
            'tiger',  # n02129604
            'cheetah',  # n02130308
            'cougar',  # n02125311
            'lynx',  # n02127052
            'jaguar',  # n02128925
        ]  #
        class_to_label = {i: class_names[i] for i in range(len(class_names))}

        with open(f'../data/ImageNetCropped/{net}_data_cats.txt', 'r') as f:
            lines = f.readlines()
            images_path = []
            for line in lines:
                images_path.append(line[:-1])  # remove new line character

        # add to columns in the dataframe to take the correct list after sampling
        print(df.shape, len(images_path))
        df[orig_image_col] = images_path

        ds = DataSet(df=df, feature_cols=feature_cols, label_col=label_col, class_to_label=class_to_label,
                     orig_images=images_path)

        if is_subset:
            # Keep Only 3 labels: 'Ankle boot', 'Sneaker', 'Sandal', 'Trouser',
            # ds.df = ds.df[ds.df[ds.label_col].isin([1, 5, 7, 9])]
            # ds.df[ds.label_col] = ds.df[ds.label_col].replace({1: 0, 5: 1, 7: 2, 9: 3})
            ds.df = ds.df[ds.df[ds.label_col].isin(is_subset)]
            ds.df[ds.label_col] = ds.df[ds.label_col].replace({orig_label: new_label for new_label, orig_label in
                                                               enumerate(is_subset)})
            ds.class_to_label = {new_label: ds.class_to_label[old_label] for new_label, old_label in
                                 enumerate(is_subset)}

        # Take only sample for now
        if sample is not None:
            print(f'Taking sample of {sample} from the data')
            ds.df = ds.df.sample(frac=sample, random_state=random_state)
            ds.orig_images = ds.df[orig_image_col].tolist()

        return ds

    @staticmethod
    def get_dataset(dataset_name, random_state=None, sample=None, is_subset=False):
        if random_state is not None:
            np.random.seed(random_state)
            random.seed(random_state)

        if dataset_name == 'MNIST':
            return DataSetFactory.mnist(sample, random_state=random_state)
        if dataset_name == 'MNIST_AE_ORIG':
            return DataSetFactory.mnist_ae(dim=None, sample=sample, is_subset=True, random_state=random_state)
        if dataset_name == 'MNIST_AE32':
            return DataSetFactory.mnist_ae(dim=32, sample=sample, is_subset=True, random_state=random_state)
        if dataset_name == 'MNIST_AE4':
            return DataSetFactory.mnist_ae(dim=4, sample=sample, is_subset=True, random_state=random_state)
        if dataset_name == 'FASHION_MNIST_AE_ORIG':
            return DataSetFactory.fashion_mnist_ae(dim=None, sample=sample, is_subset=is_subset, random_state=random_state)
        if dataset_name == 'FASHION_MNIST_AE512':
            return DataSetFactory.fashion_mnist_ae(dim=512, sample=sample, is_subset=is_subset, random_state=random_state)
        if dataset_name == 'FASHION_MNIST_AE128':
            return DataSetFactory.fashion_mnist_ae(dim=128, sample=sample, is_subset=is_subset, random_state=random_state)
        if dataset_name == 'FASHION_MNIST_AE32':
            return DataSetFactory.fashion_mnist_ae(dim=32, sample=sample, is_subset=is_subset, random_state=random_state)
        if dataset_name == 'MNIST_479':
            return DataSetFactory.mnist(sample, is_subset=True, random_state=random_state)
        if dataset_name == 'MNIST64':
            return DataSetFactory.mnist64()
        if dataset_name == 'MNIST64_038':
            return DataSetFactory.mnist64_038()
        if dataset_name == 'MNIST_038_PCA32':
            return DataSetFactory.mnist_038_pca_d()
        if dataset_name == 'MNIST_038_PCA16':
            return DataSetFactory.mnist_038_pca_d(16)
        if dataset_name == 'MNIST-DEEP-2-4':
            return DataSetFactory.mnist_deep(dim=(2,4), sample=sample, is_subset=is_subset, random_state=random_state)
        if dataset_name == 'MNIST-DEEP-32-64':
            return DataSetFactory.mnist_deep(dim=(32,64), sample=sample, is_subset=is_subset, random_state=random_state)
        if dataset_name == 'MNIST-DEEP-32-64-1000-985':
            return DataSetFactory.mnist_deep(dim=None, epoch_acc=(1000,'98_5'), sample=sample, is_subset=is_subset, random_state=random_state)
        if dataset_name == 'MNIST-DEEP-32-64-3-64':
            return DataSetFactory.mnist_deep(dim=None, epoch_acc=(3,'64'), sample=sample, is_subset=is_subset, random_state=random_state)
        if dataset_name == 'MNIST-DEEP-32-64-9-80':
            return DataSetFactory.mnist_deep(dim=None, epoch_acc=(9,'80'), sample=sample, is_subset=is_subset, random_state=random_state)
        if dataset_name == 'MNIST-DEEP-32-64-200-96':
            return DataSetFactory.mnist_deep(dim=None, epoch_acc=(200,'96'), sample=sample, is_subset=is_subset, random_state=random_state)
        if dataset_name == 'MNIST-DEEP-32-64-750-984':
            return DataSetFactory.mnist_deep(dim=None, epoch_acc=(750,'98_4'), sample=sample, is_subset=is_subset, random_state=random_state)
        if dataset_name in ['fists_no_overlap', 'cross', 'cross-denses', 'cross7', 'simple_overlap', 'dense_in_sparse', 'hourglass',
                            'hourglass-spike', 'hourglass2',
                            '4clusters_1', '4clusters_2', '4clusters_3', '4clusters_1_v2', '4clusters_3_v2',
                            '6clusters_1', '6clusters_2', '6clusters_3', '6clusters_2_v2', '6clusters_3_v2',
                            '6clusters_2_v3',
                            '8clusters_1', '8clusters_2', '8clusters_3', '8clusters_1_v2', '8clusters_2_v2']:
            df, feature_cols, label_col = pf.PolygonsFactory.get_polygons(dataset_name)
            return DataSet(df, feature_cols, label_col, class_to_label={label: f'Poly{label}' for label in df[label_col].unique()})
        if dataset_name in ['FashionMNIST', 'FashionMNIST64']:
            return DataSetFactory.fashion_mnist(dataset_name, random_state, sample, is_subset)
        if dataset_name == 'vgg_features_imagenet':
            return DataSetFactory.get_deep_features_imagenet('vgg', random_state, sample, is_subset)
        if dataset_name == 'vgg_features_cats':
            return DataSetFactory.get_deep_features_imagenet_cats('vgg', random_state, sample, is_subset)
        if dataset_name == 'densenet_features_imagenet':
            return DataSetFactory.get_deep_features_imagenet('densenet', random_state, sample, is_subset)
        if dataset_name == 'mobilenet_features_imagenet':
            return DataSetFactory.get_deep_features_imagenet('mobilenet', random_state, sample, is_subset)
        if dataset_name == 'CoverType':
            return DataSetFactory.cover_type(sample, random_state=random_state)
        if dataset_name == 'wine':
            return DataSetFactory.wine(sample, random_state=random_state)
        if dataset_name == 'wine-sklearn-scaled':
            return DataSetFactory.wine_sklearn(sample, scale=True, random_state=random_state)
        if dataset_name == 'wine-sklearn-not-scaled':
            return DataSetFactory.wine_sklearn(sample, scale=False, random_state=random_state)
        if dataset_name == 'shuttle':
            return DataSetFactory.shuttle(sample, random_state=random_state)
        if dataset_name == 'hapt-train':
            return DataSetFactory.hapt(sample, is_train=True, random_state=random_state)
        if dataset_name == 'hapt-test':
            return DataSetFactory.hapt(sample, is_train=False, random_state=random_state)
        else:
            raise Exception(f'Unsupported dataset {dataset_name}')
