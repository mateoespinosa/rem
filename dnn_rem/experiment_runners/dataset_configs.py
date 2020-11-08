"""
Configurations of the different supported datasets for training.
"""

from collections import namedtuple
import logging

from dnn_rem.rules.rule import OutputClass

################################################################################
## Global Variables
################################################################################


# The names of all the datasets whose configurations we currently support
AVAILABLE_DATASETS = [
    'Artif-1',
    'Artif-2',
    'MB-GE-ER',
    'BreastCancer',
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

# Define a class that can be used to encapsulate all the information we will
# store from a given dataset.
DatasetMetaData = namedtuple(
    'DatasetMetaData',
    [
        'name',
        'target_col',
        'output_classes',
        'n_inputs',
        'n_outputs'
    ],
)

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
            name=dataset_name,
            target_col='y',
            output_classes=output_classes,
            n_inputs=5,
            n_outputs=2,
        )
    if dataset_name == 'artif-2':
        output_classes = (
            OutputClass(name='y0', encoding=0),
            OutputClass(name='y1', encoding=1),
        )
        return DatasetMetaData(
            name=dataset_name,
            target_col='y',
            output_classes=output_classes,
            n_inputs=5,
            n_outputs=2,
        )
    if dataset_name == 'mb-ge-er':
        output_classes = (
            OutputClass(name='-', encoding=0),
            OutputClass(name='+', encoding=1),
        )
        return DatasetMetaData(
            name=dataset_name,
            target_col='ER_Expr',
            output_classes=output_classes,
            n_inputs=1000,
            n_outputs=2,
        )
    if dataset_name == 'breastcancer':
        output_classes = (
            OutputClass(name='M', encoding=0),
            OutputClass(name='B', encoding=1),
        )
        return DatasetMetaData(
            name=dataset_name,
            target_col='diagnosis',
            output_classes=output_classes,
            n_inputs=30,
            n_outputs=2,
        )
    if dataset_name == 'letterrecognition':
        output_classes = (
            OutputClass(name='A', encoding=0),
            OutputClass(name='B-Z', encoding=1),
        )
        return DatasetMetaData(
            name=dataset_name,
            target_col='letter',
            output_classes=output_classes,
            n_inputs=16,
            n_outputs=2,
        )
    if dataset_name == 'mnist':
        output_classes = (
            OutputClass(name='0', encoding=0),
            OutputClass(name='1-9', encoding=1),
        )
        return DatasetMetaData(
            name=dataset_name,
            target_col='digit',
            output_classes=output_classes,
            n_inputs=784,
            n_outputs=2,
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
            name=dataset_name,
            target_col='TCGA',
            output_classes=output_classes,
            n_inputs=20502,
            n_outputs=5,
        )
    if dataset_name == 'mb-ge-dr':
        output_classes = (
            OutputClass(name='NDR', encoding=0),
            OutputClass(name='DR', encoding=1),
        )
        return DatasetMetaData(
            name=dataset_name,
            target_col='DR',
            output_classes=output_classes,
            n_inputs=1000,
            n_outputs=2,
        )

    if dataset_name == 'mb-clin-dr':
        output_classes = (
            OutputClass(name='NDR', encoding=0),
            OutputClass(name='DR', encoding=1),
        )
        return DatasetMetaData(
            name=dataset_name,
            target_col='DR',
            output_classes=output_classes,
            n_inputs=350,
            n_outputs=2,
        )

    if dataset_name == 'mb-clin-er':
        output_classes = (
            OutputClass(name='-', encoding=0),
            OutputClass(name='+', encoding=1),
        )
        return DatasetMetaData(
            name=dataset_name,
            target_col='ER_Expr',
            output_classes=output_classes,
            n_inputs=350,
            n_outputs=2,
        )

    if dataset_name == 'mb-clinp-er':
        output_classes = (
            OutputClass(name='-', encoding=0),
            OutputClass(name='+', encoding=1),
        )
        return DatasetMetaData(
            name=dataset_name,
            target_col='ER_Expr',
            output_classes=output_classes,
            n_inputs=13,
            n_outputs=2,
        )

    if dataset_name == 'mb-ge-clin-er':
        output_classes = (
            OutputClass(name='-', encoding=0),
            OutputClass(name='+', encoding=1),
        )
        return DatasetMetaData(
            name=dataset_name,
            target_col='ER_Expr',
            output_classes=output_classes,
            n_inputs=1350,
            n_outputs=2,
        )

    if dataset_name == 'mb-ge-clinp-er':
        output_classes = (
            OutputClass(name='-', encoding=0),
            OutputClass(name='+', encoding=1),
        )
        return DatasetMetaData(
            name=dataset_name,
            target_col='ER_Expr',
            output_classes=output_classes,
            n_inputs=1013,
            n_outputs=2,
        )

    if dataset_name == 'mb-ge-2hist':
        output_classes = (
            OutputClass(name='IDC', encoding=0),
            OutputClass(name='ILC', encoding=1),
        )
        return DatasetMetaData(
            name=dataset_name,
            target_col='Histological_Type',
            output_classes=output_classes,
            n_inputs=1000,
            n_outputs=2,
        )

    if dataset_name == 'mb_ge_cdh1_2hist':
        output_classes = (
            OutputClass(name='IDC', encoding=0),
            OutputClass(name='ILC', encoding=1),
        )
        return DatasetMetaData(
            name=dataset_name,
            target_col='Histological_Type',
            output_classes=output_classes,
            n_inputs=1001,
            n_outputs=2,
        )

    if dataset_name == 'mb-1004-ge-2hist':
        output_classes = (
            OutputClass(name='IDC', encoding=0),
            OutputClass(name='ILC', encoding=1),
        )
        return DatasetMetaData(
            name=dataset_name,
            target_col='Histological_Type',
            output_classes=output_classes,
            n_inputs=1004,
            n_outputs=2,
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
            name=dataset_name,
            target_col='Histological_Type',
            output_classes=output_classes,
            n_inputs=368,
            n_outputs=6,
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
            name=dataset_name,
            target_col='Histological_Type',
            output_classes=output_classes,
            n_inputs=368,
            n_outputs=6,
        )

    if dataset_name == 'mb_imagevec50_2hist':
        output_classes = (
            OutputClass(name='IDC', encoding=0),
            OutputClass(name='ILC', encoding=1),
        )
        return DatasetMetaData(
            name=dataset_name,
            target_col='Histological_Type',
            output_classes=output_classes,
            n_inputs=368,
            n_outputs=2,
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
            name=dataset_name,
            target_col='Histological_Type',
            output_classes=output_classes,
            n_inputs=1000,
            n_outputs=6,
        )

    if dataset_name == 'mb_imagevec50_er':
        output_classes = (
            OutputClass(name='-', encoding=0),
            OutputClass(name='+', encoding=1),
        )
        return DatasetMetaData(
            name=dataset_name,
            target_col='ER_Expr',
            output_classes=output_classes,
            n_inputs=368,
            n_outputs=2,
        )

    if dataset_name == 'mb_imagevec50_dr':
        output_classes = (
            OutputClass(name='NDR', encoding=0),
            OutputClass(name='DR', encoding=1),
        )
        return DatasetMetaData(
            name=dataset_name,
            target_col='DR',
            output_classes=output_classes,
            n_inputs=368,
            n_outputs=2,
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


