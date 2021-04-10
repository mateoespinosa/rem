"""
Configurations of the different supported datasets for training.
"""
from collections import namedtuple
from sklearn.preprocessing import MinMaxScaler
import logging
import numpy as np
import pandas as pd

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
    'PARTNER-Genomic',
    'PARTNER-Clinical',
]

################################################################################
## Feature Descriptor Classes
################################################################################


class FeatureDescriptor(object):
    def __init__(self, units=None):
        self.units = units

    def is_normalized(self):
        return False

    def transform_to_numeric(self, x):
        # By default this is the identity value
        return x

    def transform_from_numeric(self, x):
        # By default this is the identity value
        return x

    def is_discrete(self):
        raise UnimplementedError("is_discrete(self)")

    def default_value(self):
        raise UnimplementedError("default_value(self)")

    def numeric_bounds(self):
        return (-float("inf"), float("inf"))

    def is_categorical(self):
        raise UnimplementedError("is_categorical(self)")


class RealDescriptor(FeatureDescriptor):
    def __init__(
        self,
        max_val=float("inf"),
        min_val=-float("inf"),
        normalized=False,
        units=None,
    ):
        super(RealDescriptor, self).__init__(units=units)
        self.normalized = normalized
        self.max_val = max_val
        self.min_val = min_val

    def default_value(self):
        if self.max_val not in [float("inf"), -float("inf")] and (
            self.min_val not in [float("inf"), -float("inf")]
        ):
            return (self.max_val + self.min_val)/2
        elif self.max_val not in [float("inf"), -float("inf")]:
            return self.max_val
        elif self.min_val not in [float("inf"), -float("inf")]:
            return self.min_val
        return 0

    def numeric_bounds(self):
        return (self.min_val, self.max_val)

    def is_discrete(self):
        return False

    def is_categorical(self):
        return False

    def is_normalized(self):
        return self.normalized


class DiscreteNumericDescriptor(RealDescriptor):
    def __init__(self, values, units=None):
        super(DiscreteNumericDescriptor, self).__init__(units=units)
        self.values = sorted(values)
        if values:
            self.max_val = values[-1]
        else:
            self.max_val = float("inf")
        if values:
            self.min_val = values[0]
        else:
            self.min_val = -float("inf")

    def default_value(self):
        if self.values:
            return self.values[len(self.values)//2]
        return 0

    def is_discrete(self):
        return True


class DiscreteEncodingDescriptor(DiscreteNumericDescriptor):
    def __init__(self, encoding_map, units=None):
        self.encoding_map = encoding_map
        values = []
        self.inverse_map = {}
        self._default_value = None
        for datum_name, numeric_val in self.encoding_map.items():
            self._default_value = self._default_value or datum_name
            values.append(numeric_val)
            self.inverse_map[numeric_val] = datum_name

        super(DiscreteEncodingDescriptor, self).__init__(
            values=values,
            units=units,
        )

    def default_value(self):
        return self._default_value

    def is_categorical(self):
        return True

    def transform_to_numeric(self, x):
        if isinstance(x, (int)):
            # Then this is already the numeric encoding
            return x
        return self.encoding_map[x]

    def transform_from_numeric(self, x):
        if not isinstance(x, int):
            # Then this is already not a numeric encoding
            return x
        return self.inverse_map[x]


class TrivialCatDescriptor(DiscreteEncodingDescriptor):
    def __init__(self, vals, units=None):
        super(TrivialCatDescriptor, self).__init__(
            encoding_map=dict(
                zip(vals, range(len(vals)))
            ),
            units=units,
        )


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
class DatasetDescriptor(object):
    def __init__(
        self,
        name="dataset",
        output_classes=None,
        n_features=None,
        target_col=None,
        feature_names=None,
        preprocessing=None,
        feature_descriptors=None,
    ):
        self.name = name
        self.n_features = n_features
        self.output_classes = output_classes
        self.target_col = target_col
        self.feature_names = feature_names
        self.preprocessing = preprocessing
        self.feature_descriptors = feature_descriptors
        self.data = None
        self.X = None
        self.y = None

    def process_dataframe(self, df):
        self.data = df
        # Set the target column, number of inputs, and feature names of our
        # dataset accordingly from the opened file if they were not provided
        self.target_col = self.target_col or (
            self.data.columns[-1]
        )
        self.n_features = self.n_features or (
            len(self.data.columns) - 1
        )
        self.feature_names = self.feature_names or (
            self.data.columns[:self.n_features]
        )
        if self.feature_descriptors is None:
            # Then we will assume they are arbitrary real numbers anyone
            # can set
            self.feature_descriptors = {
                None: RealDescriptor(
                    max_val=float("inf"),
                    min_val=-float("inf"),
                )
            }

        self.X = self.data.drop([self.target_col], axis=1).values
        self.y = self.data[self.target_col].values
        if self.output_classes is None:
            out_classes = sorted(list(set(self.y)))
            self.output_classes = []
            for out_class in out_classes:
                self.output_classes.append(
                    OutputClass(name='{out_class}', encoding=out_class)
                )
        return self.X, self.y, self.data

    def read_data(self, data_path):
        # Read our dataset. This will be the first thing we will do:
        return self.process_dataframe(pd.read_csv(data_path, sep=','))

    def get_feature_ranges(self, feature_name):
        if feature_name in self.feature_descriptors:
            return self.feature_descriptors[feature_name].numeric_bounds()
        return self.feature_descriptors.get(
            None,
            RealDescriptor(),
        ).numeric_bounds()

    def get_allowed_values(self, feature_name):
        if feature_name not in self.feature_descriptors and (
            None in self.feature_descriptors
        ):
            feature_name = None
        if feature_name in self.feature_descriptors:
            descriptor = self.feature_descriptors[feature_name]
            if descriptor.is_discrete():
                return descriptor.values
            return None
        return None

    def is_categorical(self, feature_name):
        if feature_name not in self.feature_descriptors and (
            None in self.feature_descriptors
        ):
            feature_name = None

        if feature_name in self.feature_descriptors:
            return self.feature_descriptors[feature_name].is_categorical()
        return False

    def is_discrete(self, feature_name):
        if feature_name not in self.feature_descriptors and (
            None in self.feature_descriptors
        ):
            feature_name = None

        if feature_name in self.feature_descriptors:
            return self.feature_descriptors[feature_name].is_discrete()
        return False

    def get_units(self, feature_name):
        if feature_name not in self.feature_descriptors and (
            None in self.feature_descriptors
        ):
            feature_name = None

        if feature_name in self.feature_descriptors:
            return self.feature_descriptors[feature_name].units
        return None

    def get_default_value(self, feature_name):
        if feature_name in self.feature_descriptors:
            return self.feature_descriptors[feature_name].default_value()
        return self.feature_descriptors.get(
            None,
            RealDescriptor(),
        ).default_value()

    def transform_to_numeric(self, feature_name, x):
        if feature_name in self.feature_descriptors:
            return self.feature_descriptors[feature_name].transform_to_numeric(
                x
            )
        return self.feature_descriptors.get(
            None,
            RealDescriptor(),
        ).transform_to_numeric(x)

    def transform_from_numeric(self, feature_name, x):
        if feature_name in self.feature_descriptors:
            return self.feature_descriptors[feature_name].transform_from_numeric(
                x
            )
        return self.feature_descriptors.get(
            None,
            RealDescriptor(),
        ).transform_from_numeric(x)

    def is_normalized(self, feature_name):
        if feature_name in self.feature_descriptors:
            return self.feature_descriptors[feature_name].is_normalized()
        return self.feature_descriptors.get(
            None,
            RealDescriptor(),
        ).is_normalized()

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
    :return DatasetDescriptor:  The configuration corresponding to the given
                              dataset.
    """
    dataset_name = dataset_name.lower()
    if dataset_name == 'artif-1':
        output_classes = (
            OutputClass(name='y0', encoding=0),
            OutputClass(name='y1', encoding=1),
        )
        return DatasetDescriptor(
            name=dataset_name,
            output_classes=output_classes,
            target_col='y',
        )
    if dataset_name == 'artif-2':
        output_classes = (
            OutputClass(name='y0', encoding=0),
            OutputClass(name='y1', encoding=1),
        )
        return DatasetDescriptor(
            name=dataset_name,
            output_classes=output_classes,
            target_col='y',
        )
    if dataset_name == 'mb-ge-er':
        output_classes = (
            OutputClass(name='negative', encoding=0),
            OutputClass(name='positive', encoding=1),
        )
        return DatasetDescriptor(
            name=dataset_name,
            output_classes=output_classes,
            preprocessing=unit_scale_preprocess,
            target_col='ER_Expr',
            feature_descriptors={
                None: RealDescriptor(min_val=0, max_val=1, normalized=True),
            },
        )
    if dataset_name == 'breastcancer':
        output_classes = (
            OutputClass(name='M', encoding=0),
            OutputClass(name='B', encoding=1),
        )
        return DatasetDescriptor(
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
        return DatasetDescriptor(
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
        return DatasetDescriptor(
            name=dataset_name,
            output_classes=output_classes,
            target_col='letter',
        )
    if dataset_name == 'mnist':
        output_classes = (
            OutputClass(name='0', encoding=0),
            OutputClass(name='1-9', encoding=1),
        )
        return DatasetDescriptor(
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
        return DatasetDescriptor(
            name=dataset_name,
            output_classes=output_classes,
            target_col='TCGA',
        )
    if dataset_name == 'mb-ge-dr':
        output_classes = (
            OutputClass(name='NDR', encoding=0),
            OutputClass(name='DR', encoding=1),
        )
        return DatasetDescriptor(
            name=dataset_name,
            output_classes=output_classes,
            target_col='DR',
        )

    if dataset_name == 'mb-clin-dr':
        output_classes = (
            OutputClass(name='NDR', encoding=0),
            OutputClass(name='DR', encoding=1),
        )
        return DatasetDescriptor(
            name=dataset_name,
            output_classes=output_classes,
            target_col='DR',
        )

    if dataset_name == 'mb-clin-er':
        output_classes = (
            OutputClass(name='Negative', encoding=0),
            OutputClass(name='Positive', encoding=1),
        )
        return DatasetDescriptor(
            name=dataset_name,
            output_classes=output_classes,
            target_col='ER_Expr',
        )

    if dataset_name == 'mb-clinp-er':
        output_classes = (
            OutputClass(name='Negative', encoding=0),
            OutputClass(name='Positive', encoding=1),
        )
        return DatasetDescriptor(
            name=dataset_name,
            output_classes=output_classes,
            target_col='ER_Expr',
        )

    if dataset_name == 'mb-ge-clin-er':
        output_classes = (
            OutputClass(name='Negative', encoding=0),
            OutputClass(name='Positive', encoding=1),
        )
        return DatasetDescriptor(
            name=dataset_name,
            output_classes=output_classes,
            target_col='ER_Expr',
        )

    if dataset_name == 'mb-ge-clinp-er':
        output_classes = (
            OutputClass(name='Negative', encoding=0),
            OutputClass(name='Positive', encoding=1),
        )
        return DatasetDescriptor(
            name=dataset_name,
            output_classes=output_classes,
            target_col='ER_Expr',
        )

    if dataset_name == 'mb-ge-2hist':
        output_classes = (
            OutputClass(name='IDC', encoding=0),
            OutputClass(name='ILC', encoding=1),
        )
        return DatasetDescriptor(
            name=dataset_name,
            output_classes=output_classes,
            preprocessing=unit_scale_preprocess,
            target_col='Histological_Type',
            feature_descriptors={
                None: RealDescriptor(min_val=0, max_val=1, normalized=True),
            },
        )

    if dataset_name == 'mb_ge_cdh1_2hist':
        output_classes = (
            OutputClass(name='IDC', encoding=0),
            OutputClass(name='ILC', encoding=1),
        )
        return DatasetDescriptor(
            name=dataset_name,
            output_classes=output_classes,
            preprocessing=unit_scale_preprocess,
            target_col='Histological_Type',
            feature_descriptors={
                None: RealDescriptor(min_val=0, max_val=1, normalized=True),
            },
        )

    if dataset_name == 'mb-1004-ge-2hist':
        output_classes = (
            OutputClass(name='IDC', encoding=0),
            OutputClass(name='ILC', encoding=1),
        )
        return DatasetDescriptor(
            name=dataset_name,
            output_classes=output_classes,
            preprocessing=unit_scale_preprocess,
            target_col='Histological_Type',
            feature_descriptors={
                None: RealDescriptor(min_val=0, max_val=1, normalized=True),
            },
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
        return DatasetDescriptor(
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
        return DatasetDescriptor(
            name=dataset_name,
            output_classes=output_classes,
            target_col='Histological_Type',
        )

    if dataset_name == 'mb_imagevec50_2hist':
        output_classes = (
            OutputClass(name='IDC', encoding=0),
            OutputClass(name='ILC', encoding=1),
        )
        return DatasetDescriptor(
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
        return DatasetDescriptor(
            name=dataset_name,
            output_classes=output_classes,
            preprocessing=unit_scale_preprocess,
            target_col='Histological_Type',
            feature_descriptors={
                None: RealDescriptor(min_val=0, max_val=1, normalized=True),
            },
        )

    if dataset_name == 'mb_imagevec50_er':
        output_classes = (
            OutputClass(name='ER Negative', encoding=0),
            OutputClass(name='ER Positive', encoding=1),
        )
        return DatasetDescriptor(
            name=dataset_name,
            output_classes=output_classes,
            target_col='ER_Expr',
        )

    if dataset_name == 'mb_imagevec50_dr':
        output_classes = (
            OutputClass(name='NDR', encoding=0),
            OutputClass(name='DR', encoding=1),
        )
        return DatasetDescriptor(
            name=dataset_name,
            output_classes=output_classes,
            target_col='DR',
        )
    if dataset_name == "partner-clinical":
        output_classes = (
            OutputClass(name='non-pCR', encoding=0),
            OutputClass(name='pCR', encoding=1),
        )
        return DatasetDescriptor(
            name=dataset_name,
            output_classes=output_classes,
            target_col='pCR',
            feature_descriptors={
                None: RealDescriptor(),
                "Age": DiscreteNumericDescriptor(
                    list(range(1, 100)),
                    units="yrs",
                ),
                "Receptor_Status": TrivialCatDescriptor(
                    ["Negative", "Positive"]
                ),
                "Tumour_subtype": TrivialCatDescriptor([
                    "Dictal(NST)",
                    "Metaplastic",
                    "Medualiary",
                    "Apocrine",
                    "Mixed",
                ]),
                "Ductal_subtype": TrivialCatDescriptor(["No Ductal", "Ductal"]),
                "Grade": DiscreteNumericDescriptor([1, 2, 3]),
                "Largest_Clinical_Size": RealDescriptor(min_val=0, units="mm"),
                "T4_or_Inflammatory": TrivialCatDescriptor(["No", "Yes"]),
                "Clinical_Nodal_Involvement": TrivialCatDescriptor(
                    ["No", "Yes"]
                ),
                "Clinical_Stage": TrivialCatDescriptor([
                    "IA",
                    "IIA",
                    "IIIA",
                    "IIB",
                    "IIIB",
                    "IIIC",
                ]),
                "Smoking_category": TrivialCatDescriptor([
                    "Never",
                    "Current",
                    "Former",
                    "Unknown",
                ]),
                "BMI": RealDescriptor(min_val=0, units="kg/m^2"),
                "anthracycline": TrivialCatDescriptor(["No", "Yes"]),
                "PARTNER": TrivialCatDescriptor(["No", "Yes"]),
                "treatment_group": TrivialCatDescriptor(["Control", "Olaparib"]),
                "TILs": RealDescriptor(min_val=0, max_val=1),
                "EGFR": TrivialCatDescriptor(["Negative", "Positive"]),
                "CK5_6": TrivialCatDescriptor(["Negative", "Positive"]),
                "ARIHC": RealDescriptor(min_val=0, max_val=100, units="%"),
                "pCR": RealDescriptor(["non-pCR", "pCR"]),
            }
        )

    if dataset_name == "partner-genomic":
        output_classes = (
            OutputClass(name='non-pCR', encoding=0),
            OutputClass(name='pCR', encoding=1),
        )
        return DatasetDescriptor(
            name=dataset_name,
            output_classes=output_classes,
            target_col='pCR',
            preprocessing=unit_scale_preprocess,
            feature_descriptors={
                None: RealDescriptor(min_val=0, max_val=1, normalized=True),
                "pCR": RealDescriptor(["non-pCR", "pCR"]),
            }
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


