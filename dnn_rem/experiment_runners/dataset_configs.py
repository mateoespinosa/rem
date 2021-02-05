"""
Configurations of the different supported datasets for training.
"""
from collections import namedtuple
from sklearn.preprocessing import MinMaxScaler
import logging
import numpy as np

################################################################################
## Global Variables
################################################################################


# The names of all the datasets whose configurations we currently support
AVAILABLE_DATASETS = [
    'Artif-1',
    'Artif-2',
    'MB-GE-ER',
    'BreastCancer',
    'Iris',
    'LetterRecognition',
    'MNIST',
    'TCGA-PANCAN',
    'MB-GE-DR',
    'MB-Clin-DR',
    'MB-Clin-ER',
    'MB-ClinP-ER',
    'MB-GE-Clin-ER',
    'MB-GE-ClinP-ER',
    'MB-GE-2Hist',
    'MB_GE_CDH1_2Hist',
    'MB-1004-GE-2Hist',
    'MB-ImageVec5-6Hist',
    'MB-ImageVec50-6Hist',
    'mb_imagevec50_2Hist',
    'MB-GE-6Hist',
    'mb_imagevec50_ER',
    'mb_imagevec50_DR',
]

################################################################################
## Helper Classes
################################################################################


class OutputClass(object):
    """
    Represents the conclusion of a given rule. Immutable and Hashable.

    Each output class has a name and its relevant encoding in the network
    i.e. which output neuron it corresponds to
    """

    def __init__(self, name: str, encoding: int):
        self.name = name
        self.encoding = encoding

    def __str__(self):
        return f'OUTPUT_CLASS={self.name} (neuron name {self.encoding})'

    def __eq__(self, other):
        return (
            isinstance(other, OutputClass) and
            (self.name == other.name) and
            (self.encoding == other.encoding)
        )

    def __hash__(self):
        return hash((self.name, self.encoding))


# Define a class that can be used to encapsulate all the information we will
# store from a given dataset.
class DatasetMetaData(object):
    def __init__(
        self,
        name,
        output_classes,
        n_features=None,
        target_col=None,
        feature_names=None,
        preprocessing=None,
    ):
        self.name = name
        self.n_features = n_features
        self.output_classes = output_classes
        self.target_col = target_col
        self.feature_names = feature_names
        self.preprocessing = preprocessing


################################################################################
## Helper Methods
################################################################################

def unit_scale_preprocess(X_train, y_train, X_test=None, y_test=None):
    """
    Simple scaling preprocessing function to scale the X matrix so that all of
    it's features are in [0, 1]

    :param np.array X_train: 2D matrix of data points to be used for training.
    :param np.array y_train: 1D matrix of labels for the given training data
        points.
    :param np.array X_test: optional 2D matrix of data points to be used for
        testing.
    :param np.array y_test: optional 1D matrix of labels for the given testing
        data points.
    :returns Tuple[np.array, np.array]: The new processed (X_train, y_train)
        data if not test data was provided. Otherwise it returns the processed
        (X_train, y_train, X_test, y_test)
    """
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    if X_test is not None:
        X_test = scaler.transform(X_test)
        return X_train, y_train, X_test, y_test
    return X_train, y_train


def replace_categorical_outputs(
    X_train,
    y_train,
    output_classes,
    X_test=None,
    y_test=None,
):
    """
    Simple scaling preprocessing function to replace categorical values in the
    given vector y_train with their numerical encodings.

    :param np.array X_train: 2D matrix of data points to be used for training.
    :param np.array y_train: 1D matrix of labels for the given training data
        points.
    :param List[OutputClass]: list of possible output categorical classes in
        vector y_train.
    :param np.array X_test: optional 2D matrix of data points to be used for
        testing.
    :param np.array y_test: optional 1D matrix of labels for the given testing
        data points.
    :returns Tuple[np.array, np.array]: The new processed (X_train, y_train)
        data if not test data was provided. Otherwise it returns the processed
        (X_train, y_train, X_test, y_test)
    """
    out_map = {
        c.name: c.encoding
        for c in output_classes
    }
    for i, val in enumerate(y_train):
        y_train[i] = out_map[val]
    if y_test is not None:
        for i, val in enumerate(y_test):
            y_test[i] = out_map[val]
    if X_test is not None:
        return (
            X_train,
            y_train.astype(np.int32),
            X_test,
            y_test.astype(np.int32),
        )

    return X_train, y_train.astype(np.int32)


################################################################################
## Exposed Methods
################################################################################


def get_data_configuration(dataset_name):
    """
    Gets the configuration of dataset with name `dataset_name` if it is a
    supported dataset (case insensitive). Otherwise a ValueError is thrown.

    :param str dataset_name:  The name of the dataset we want to fetch.
    :return DatasetMetaData:  The configuration corresponding to the given
                              dataset.
    """
    dataset_name = dataset_name.lower()
    if dataset_name == 'artif-1':
        output_classes = (
            OutputClass(name='y0', encoding=0),
            OutputClass(name='y1', encoding=1),
        )
        return DatasetMetaData(
            n_features=5,
            name=dataset_name,
            output_classes=output_classes,
            target_col='y',
        )
    if dataset_name == 'artif-2':
        output_classes = (
            OutputClass(name='y0', encoding=0),
            OutputClass(name='y1', encoding=1),
        )
        return DatasetMetaData(
            n_features=5,
            name=dataset_name,
            output_classes=output_classes,
            target_col='y',
        )
    if dataset_name == 'mb-ge-er':
        output_classes = (
            OutputClass(name='-', encoding=0),
            OutputClass(name='+', encoding=1),
        )
        return DatasetMetaData(
            n_features=1000,
            name=dataset_name,
            output_classes=output_classes,
            preprocessing=unit_scale_preprocess,
            target_col='ER_Expr',
        )
    if dataset_name == 'breastcancer':
        output_classes = (
            OutputClass(name='M', encoding=0),
            OutputClass(name='B', encoding=1),
        )
        return DatasetMetaData(
            n_features=30,
            name=dataset_name,
            output_classes=output_classes,
            target_col='diagnosis',
        )
    if dataset_name == 'iris':
        output_classes = (
            OutputClass(name='Setosa', encoding=0),
            OutputClass(name='Versicolor', encoding=1),
            OutputClass(name='Virginica', encoding=2),
        )
        # Helper method for preprocessing our data
        def preprocess_fun(X_train, y_train, X_test=None, y_test=None):
            return replace_categorical_outputs(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                output_classes=output_classes,
            )
        return DatasetMetaData(
            n_features=4,
            name=dataset_name,
            output_classes=output_classes,
            preprocessing=preprocess_fun,
            target_col='variety',
        )
    if dataset_name == 'letterrecognition':
        output_classes = (
            OutputClass(name='A', encoding=0),
            OutputClass(name='B-Z', encoding=1),
        )
        return DatasetMetaData(
            n_features=16,
            name=dataset_name,
            output_classes=output_classes,
            target_col='letter',
        )
    if dataset_name == 'mnist':
        output_classes = (
            OutputClass(name='0', encoding=0),
            OutputClass(name='1-9', encoding=1),
        )
        return DatasetMetaData(
            n_features=784,
            name=dataset_name,
            output_classes=output_classes,
            target_col='digit',
        )
    if dataset_name == 'tcga-pancan':
        output_classes = (
            OutputClass(name='BRCA', encoding=0),
            OutputClass(name='KIRC', encoding=1),
            OutputClass(name='LAUD', encoding=2),
            OutputClass(name='PRAD', encoding=3),
            OutputClass(name='COAD', encoding=4),
        )
        return DatasetMetaData(
            n_features=20502,
            name=dataset_name,
            output_classes=output_classes,
            target_col='TCGA',
        )
    if dataset_name == 'mb-ge-dr':
        output_classes = (
            OutputClass(name='NDR', encoding=0),
            OutputClass(name='DR', encoding=1),
        )
        return DatasetMetaData(
            n_features=1000,
            name=dataset_name,
            output_classes=output_classes,
            target_col='DR',
        )

    if dataset_name == 'mb-clin-dr':
        output_classes = (
            OutputClass(name='NDR', encoding=0),
            OutputClass(name='DR', encoding=1),
        )
        return DatasetMetaData(
            n_features=350,
            name=dataset_name,
            output_classes=output_classes,
            target_col='DR',
        )

    if dataset_name == 'mb-clin-er':
        output_classes = (
            OutputClass(name='-', encoding=0),
            OutputClass(name='+', encoding=1),
        )
        return DatasetMetaData(
            n_features=350,
            name=dataset_name,
            output_classes=output_classes,
            target_col='ER_Expr',
        )

    if dataset_name == 'mb-clinp-er':
        output_classes = (
            OutputClass(name='-', encoding=0),
            OutputClass(name='+', encoding=1),
        )
        return DatasetMetaData(
            n_features=13,
            name=dataset_name,
            output_classes=output_classes,
            target_col='ER_Expr',
        )

    if dataset_name == 'mb-ge-clin-er':
        output_classes = (
            OutputClass(name='-', encoding=0),
            OutputClass(name='+', encoding=1),
        )
        return DatasetMetaData(
            n_features=1350,
            name=dataset_name,
            output_classes=output_classes,
            target_col='ER_Expr',
        )

    if dataset_name == 'mb-ge-clinp-er':
        output_classes = (
            OutputClass(name='-', encoding=0),
            OutputClass(name='+', encoding=1),
        )
        return DatasetMetaData(
            n_features=1013,
            name=dataset_name,
            output_classes=output_classes,
            target_col='ER_Expr',
        )

    if dataset_name == 'mb-ge-2hist':
        output_classes = (
            OutputClass(name='IDC', encoding=0),
            OutputClass(name='ILC', encoding=1),
        )
        return DatasetMetaData(
            n_features=1000,
            name=dataset_name,
            output_classes=output_classes,
            preprocessing=unit_scale_preprocess,
            target_col='Histological_Type',
        )

    if dataset_name == 'mb_ge_cdh1_2hist':
        output_classes = (
            OutputClass(name='IDC', encoding=0),
            OutputClass(name='ILC', encoding=1),
        )
        return DatasetMetaData(
            n_features=1001,
            name=dataset_name,
            output_classes=output_classes,
            preprocessing=unit_scale_preprocess,
            target_col='Histological_Type',
        )

    if dataset_name == 'mb-1004-ge-2hist':
        output_classes = (
            OutputClass(name='IDC', encoding=0),
            OutputClass(name='ILC', encoding=1),
        )
        return DatasetMetaData(
            n_features=1004,
            name=dataset_name,
            output_classes=output_classes,
            preprocessing=unit_scale_preprocess,
            target_col='Histological_Type',
        )

    if dataset_name == 'mb-imagevec5-6hist':
        output_classes = (
            OutputClass(name='IDC', encoding=0),
            OutputClass(name='ILC', encoding=1),
            OutputClass(name='IDC+ILC', encoding=2),
            OutputClass(name='IDC-MUC', encoding=3),
            OutputClass(name='IDC-TUB', encoding=4),
            OutputClass(name='IDC-MED', encoding=5),
        )
        return DatasetMetaData(
            n_features=368,
            name=dataset_name,
            output_classes=output_classes,
            target_col='Histological_Type',
        )

    if dataset_name == 'mb-imagevec50-6hist':
        output_classes = (
            OutputClass(name='IDC', encoding=0),
            OutputClass(name='ILC', encoding=1),
            OutputClass(name='IDC+ILC', encoding=2),
            OutputClass(name='IDC-MUC', encoding=3),
            OutputClass(name='IDC-TUB', encoding=4),
            OutputClass(name='IDC-MED', encoding=5),
        )
        return DatasetMetaData(
            n_features=368,
            name=dataset_name,
            output_classes=output_classes,
            target_col='Histological_Type',
        )

    if dataset_name == 'mb_imagevec50_2hist':
        output_classes = (
            OutputClass(name='IDC', encoding=0),
            OutputClass(name='ILC', encoding=1),
        )
        return DatasetMetaData(
            n_features=368,
            name=dataset_name,
            output_classes=output_classes,
            target_col='Histological_Type',
        )

    if dataset_name == 'mb-ge-6hist':
        output_classes = (
            OutputClass(name='IDC', encoding=0),
            OutputClass(name='ILC', encoding=1),
            OutputClass(name='IDC+ILC', encoding=2),
            OutputClass(name='IDC-MUC', encoding=3),
            OutputClass(name='IDC-TUB', encoding=4),
            OutputClass(name='IDC-MED', encoding=5),
        )
        return DatasetMetaData(
            n_features=1000,
            name=dataset_name,
            output_classes=output_classes,
            preprocessing=unit_scale_preprocess,
            target_col='Histological_Type',
        )

    if dataset_name == 'mb_imagevec50_er':
        output_classes = (
            OutputClass(name='-', encoding=0),
            OutputClass(name='+', encoding=1),
        )
        return DatasetMetaData(
            n_features=368,
            name=dataset_name,
            output_classes=output_classes,
            target_col='ER_Expr',
        )

    if dataset_name == 'mb_imagevec50_dr':
        output_classes = (
            OutputClass(name='NDR', encoding=0),
            OutputClass(name='DR', encoding=1),
        )
        return DatasetMetaData(
            n_features=368,
            name=dataset_name,
            output_classes=output_classes,
            target_col='DR',
        )

    # Else this is a big no-no
    raise ValueError(f'Invalid dataset name "{dataset_name}"')


def is_valid_dataset(dataset_name):
    """
    Determines whether the specified dataset name is valid dataset name within
    the supported datasets for experimentation.

    :param str dataset_name:  The name of the dataset we want to check
    :return bool: whether or not this is a valid dataset.
    """
    return dataset_name.lower() in list(map(
        lambda x: x.lower(),
        AVAILABLE_DATASETS,
    ))


