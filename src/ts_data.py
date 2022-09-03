# pylint: disable=C0103,C0115,C0116,R0913,E1121,C0301
"""Functions for extracting dataset features and labels"""
import h5py
import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer

from src.settings import DATA_ROOT


def load_ABIDE1(
    dataset_path: str = DATA_ROOT.joinpath("abide/ABIDE1_AllData.h5"),
    indices_path: str = DATA_ROOT.joinpath("abide/correct_indices_GSP.csv"),
    labels_path: str = DATA_ROOT.joinpath("abide/labels_ABIDE1.csv"),
):
    """
    Return ABIDE1 data

    Input:
    dataset_path: str = DATA_ROOT.joinpath("abide/ABIDE1_AllData.h5")
    - path to the dataset
    indices_path: str = DATA_ROOT.joinpath("abide/correct_indices_GSP.csv")
    - path to correct indices/components
    labels_path: str = DATA_ROOT.joinpath("abide/labels_ABIDE1.csv")
    - path to labels

    Output:
    features, labels
    """

    # get data
    hf = h5py.File(dataset_path, "r")
    data = hf.get("ABIDE1_dataset")
    data = np.array(data)
    # print(data.shape)
    # >>> (569, 14000)

    # reshape data
    num_subjects = data.shape[0]
    num_components = 100
    data = data.reshape(num_subjects, num_components, -1)
    # 569 - sessions - data.shape[0]
    # 100 - components - data.shape[1]
    # 140 - time points - data.shape[2]

    # get correct indices/components
    indices = pd.read_csv(indices_path, header=None)
    idx = indices[0].values - 1
    # filter the data: leave only correct components
    data = data[:, idx, :]
    # print(data.shape)
    # 53 - components - data.shape[1]

    # get labels
    labels = pd.read_csv(labels_path, header=None)
    labels = labels.values.flatten().astype("int") - 1

    return data, labels


def load_ABIDE1_869(
    dataset_path: str = DATA_ROOT.joinpath(
        "abide869/ABIDE1_AllData_869Subjects_ICA.npz"
    ),
    indices_path: str = DATA_ROOT.joinpath("abide869/correct_indices_GSP.csv"),
    labels_path: str = DATA_ROOT.joinpath("abide869/labels_ABIDE1_869Subjects.csv"),
):
    """
    Return ABIDE1 data

    Input:
    dataset_path: str = DATA_ROOT.joinpath("abide/ABIDE1_AllData.h5")
    - path to the dataset
    indices_path: str = DATA_ROOT.joinpath("abide/correct_indices_GSP.csv")
    - path to correct indices/components
    labels_path: str = DATA_ROOT.joinpath("abide/labels_ABIDE1.csv")
    - path to labels

    Output:
    features, labels
    """

    data = np.load(dataset_path)
    # 869 - sessions - data.shape[0]
    # 100 - components - data.shape[1]
    # 295 - time points - data.shape[2]

    # get correct indices/components
    indices = pd.read_csv(indices_path, header=None)
    idx = indices[0].values - 1

    # filter the data: leave only correct components and the first 156 time points
    # (not all subjects have all 160 time points)
    data = data[:, idx, :]
    # print(data.shape)
    # 53 - components - data.shape[1]
    # 156 - time points - data.shape[2]

    # get labels
    labels = pd.read_csv(labels_path, header=None)
    labels = labels.values.flatten().astype("int") - 1

    return data, labels


def load_COBRE(
    dataset_path: str = DATA_ROOT.joinpath("cobre/COBRE_AllData.h5"),
    indices_path: str = DATA_ROOT.joinpath("cobre/correct_indices_GSP.csv"),
    labels_path: str = DATA_ROOT.joinpath("cobre/labels_COBRE.csv"),
):
    """
    Return COBRE data

    Input:
    dataset_path: str = DATA_ROOT.joinpath("cobre/COBRE_AllData.h5")
    - path to the dataset
    indices_path: str = DATA_ROOT.joinpath("cobre/correct_indices_GSP.csv")
    - path to correct indices/components
    labels_path: str = DATA_ROOT.joinpath("cobre/labels_COBRE.csv")
    - path to labels

    Output:
    features, labels
    """

    # get data
    hf = h5py.File(dataset_path, "r")
    data = hf.get("COBRE_dataset")
    data = np.array(data)
    # print(data.shape)
    # >>> (157, 14000)

    # reshape data
    num_subjects = data.shape[0]
    num_components = 100
    data = data.reshape(num_subjects, num_components, -1)
    # 157 - sessions - data.shape[0]
    # 100 - components - data.shape[1]
    # 140 - time points - data.shape[2]

    # get correct indices/components
    indices = pd.read_csv(indices_path, header=None)
    idx = indices[0].values - 1
    # filter the data: leave only correct components
    data = data[:, idx, :]
    # print(data.shape)
    # 53 - components - data.shape[1]

    # get labels
    labels = pd.read_csv(labels_path, header=None)
    labels = labels.values.flatten().astype("int") - 1

    return data, labels


def load_FBIRN(
    dataset_path: str = DATA_ROOT.joinpath("fbirn/FBIRN_AllData.h5"),
    indices_path: str = DATA_ROOT.joinpath("fbirn/correct_indices_GSP.csv"),
    labels_path: str = DATA_ROOT.joinpath("fbirn/labels_FBIRN_new.csv"),
):
    """
    Return FBIRN data

    Input:
    dataset_path: str = DATA_ROOT.joinpath("fbirn/FBIRN_AllData.h5")
    - path to the dataset
    indices_path: str = DATA_ROOT.joinpath("fbirn/correct_indices_GSP.csv")
    - path to correct indices/components
    labels_path: str = DATA_ROOT.joinpath("fbirn/labels_FBIRN_new.csv")
    - path to labels

    Output:
    features, labels
    """

    # get data
    hf = h5py.File(dataset_path, "r")
    data = hf.get("FBIRN_dataset")
    data = np.array(data)
    # print(data.shape)
    # >>> (311, 14000)

    # reshape data
    num_subjects = data.shape[0]
    num_components = 100
    data = data.reshape(num_subjects, num_components, -1)
    # 311 - sessions - data.shape[0]
    # 100 - components - data.shape[1]
    # 140 - time points - data.shape[2]

    # get correct indices/components
    indices = pd.read_csv(indices_path, header=None)
    idx = indices[0].values - 1
    # filter the data: leave only correct components
    data = data[:, idx, :]
    # print(data.shape)
    # 53 - components - data.shape[1]

    # get labels
    labels = pd.read_csv(labels_path, header=None)
    labels = labels.values.flatten().astype("int") - 1

    return data, labels


def load_OASIS(
    only_first_sessions: bool = True,
    only_two_classes: bool = True,
    dataset_path: str = DATA_ROOT.joinpath("oasis/OASIS3_AllData_allsessions.npz"),
    indices_path: str = DATA_ROOT.joinpath("oasis/correct_indices_GSP.csv"),
    labels_path: str = DATA_ROOT.joinpath("oasis/labels_OASIS_6_classes.csv"),
    sessions_path: str = DATA_ROOT.joinpath("oasis/oasis_first_sessions_index.csv"),
):
    """
    Return OASIS data

    Input:
    only_first_sessions: bool = True
    - load only first sessions of each subject
    only_two_classes: bool = True
    - filter all classes except for 0 and 1 (HC and AZ)
    dataset_path: str = DATA_ROOT.joinpath("oasis/OASIS3_AllData_allsessions.npz")
    - path to the dataset
    indices_path: str = DATA_ROOT.joinpath("oasis/correct_indices_GSP.csv")
    - path to correct indices/components
    labels_path: str = DATA_ROOT.joinpath("oasis/labels_OASIS_6_classes.csv")
    - path to labels
    sessions_path: str = DATA_ROOT.joinpath("oasis/oasis_first_sessions_index.csv")
    - path to indices of the first sessions

    Output:
    features, labels
    """

    data = np.load(dataset_path)
    # 2826 - sessions - data.shape[0]
    # 100 - components - data.shape[1]
    # 160 - time points - data.shape[2]

    # get correct indices/components
    indices = pd.read_csv(indices_path, header=None)
    idx = indices[0].values - 1

    # filter the data: leave only correct components and the first 156 time points
    # (not all subjects have all 160 time points)
    data = data[:, idx, :156]
    # print(data.shape)
    # 53 - components - data.shape[1]
    # 156 - time points - data.shape[2]

    # get labels
    labels = pd.read_csv(labels_path, header=None)
    labels = labels.values.flatten().astype("int") - 1

    if only_first_sessions:
        # leave only first sessions
        sessions = pd.read_csv(sessions_path, header=None)
        first_session = sessions[0].values - 1

        data = data[first_session, :, :]
        # 912 - sessions - data.shape[0] - only first session
        labels = labels[first_session]

    if only_two_classes:
        # leave subjects of class 0 and 1 only
        filter_array = []
        for label in labels:
            if label in (0, 1):
                filter_array.append(True)
            else:
                filter_array.append(False)

        data = data[filter_array, :, :]
        # 2559 - sessions - data.shape[0] - subjects of class 0 and 1
        # 823 - sessions - data.shape[0] - if only first sessions are considered
        labels = labels[filter_array]

    return data, labels


def load_balanced_OASIS():
    """
    Return 320 balanced OASIS subjects (classes 0 and 1 only)

    Output:
    features, labels
    """

    features, labels = load_OASIS(only_first_sessions=True, only_two_classes=True)

    # for 651 subjects with label 0
    filter_array_0 = []
    # for 172 subjects with label 1
    filter_array_1 = []

    for label in labels:
        if label == 0:
            filter_array_0.append(True)
            filter_array_1.append(False)
        else:
            filter_array_0.append(False)
            filter_array_1.append(True)

    # copy subjects to separate arrays
    features_0 = features[filter_array_0]
    labels_0 = labels[filter_array_0]
    features_1 = features[filter_array_1]
    labels_1 = labels[filter_array_1]

    # balance the arrays
    features_0 = features_0[:160]
    labels_0 = labels_0[:160]
    features_1 = features_1[:160]
    labels_1 = labels_1[:160]

    features = np.concatenate((features_0, features_1), axis=0)
    labels = np.concatenate((labels_0, labels_1), axis=0)

    return features, labels


def load_UKB(
    dataset_path: str = "/data/users2/ppopov1/UKB_data/UKB_sex_data.npz",
    indices_path: str = "/data/users2/ppopov1/UKB_data/correct_indices_GSP.csv",
):
    """
    Return UKB data

    Input:
    dataset_path: str = "/data/users2/ppopov1/UKB_data/UKB_sex_data.npz"
    - path to the dataset with lablels
    indices_path: str = "/data/users2/ppopov1/UKB_data/correct_indices_GSP.csv"
    - path to correct indices/components


    Output:
    features, labels
    """

    features = None
    labels = None
    with np.load(dataset_path) as npzfile:
        features = npzfile["features"]
        labels = npzfile["labels"]

    # get correct indices/components
    indices = pd.read_csv(indices_path, header=None)
    idx = indices[0].values - 1
    features = features[:, idx, :]

    return features, labels


def load_BSNIP(
    only_two_classes: bool = True,
    invert_classes: bool = True,
    dataset_path: str = DATA_ROOT.joinpath("bsnip/BSNIP_data.npz"),
    indices_path: str = DATA_ROOT.joinpath("bsnip/correct_indices_GSP.csv"),
):
    """
    Return BSNIP data

    Input:
    dataset_path: str = DATA_ROOT.joinpath("bsnip/BSNIP_data.npz")
    - path to the dataset with lablels
    indices_path: str = DATA_ROOT.joinpath("bsnip/correct_indices_GSP.csv")
    - path to correct indices/components


    Output:
    features, labels
    """

    features = None
    labels = None
    with np.load(dataset_path) as npzfile:
        features = npzfile["features"]
        labels = npzfile["labels"]

    # get correct indices/components
    indices = pd.read_csv(indices_path, header=None)
    idx = indices[0].values - 1
    features = features[:, idx, :]

    if only_two_classes:
        # leave subjects of class 0 and 1 only
        # {"NC": 0, "SZ": 1, "SAD": 2, "BP": 3, "BPnon": 4, "OTH": 5}
        filter_array = []
        for label in labels:
            if label in (0, 1):
                filter_array.append(True)
            else:
                filter_array.append(False)

        features = features[filter_array, :, :]
        labels = labels[filter_array]

    if only_two_classes and invert_classes:
        new_labels = []
        for label in labels:
            if label == 0:
                new_labels.append(1)
            else:
                new_labels.append(0)

        labels = np.array(new_labels)

    return features, labels


class TSQuantileTransformer:
    def __init__(self, *args, n_quantiles: int, **kwargs):
        self.n_quantiles = n_quantiles
        self._args = args
        self._kwargs = kwargs
        self.transforms = {}

    def fit(self, features: np.ndarray):
        for i in range(features.shape[1]):
            self.transforms[i] = QuantileTransformer(
                *self._args, n_quantiles=self.n_quantiles, **self._kwargs
            ).fit(features[:, i, :])
        return self

    def transform(self, features: np.ndarray):
        result = np.empty_like(features, dtype=np.int32)
        for i in range(features.shape[1]):
            result[:, i, :] = (
                self.transforms[i].transform(features[:, i, :]) * self.n_quantiles
            ).astype(np.int32)
        return result
