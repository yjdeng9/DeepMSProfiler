
import pandas as pd
import copy
import sys
import pandas as pd
import numpy as np
from pyteomics import mzml, auxiliary

from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers
from tensorflow.keras.models import Model

import os

def get_bin_peak(start, width, end, raw_mzml, tt):
    mz = raw_mzml.time[tt]["m/z array"]
    it = raw_mzml.time[tt]["intensity array"]

    label_idx = list(range(start, end, width))
    bins = copy.deepcopy(label_idx)
    bins.append(end)

    mz_bins = pd.cut(mz, bins=bins, labels=label_idx)

    xValue = copy.deepcopy(label_idx)
    yValue = []
    for vv in xValue:
        value_idx = np.where(mz_bins == vv)
        it_values = it[value_idx]
        if len(it_values) != 0:
            it_max = np.max(it_values)
        else:
            it_max = 0
        yValue.append(it_max)
    return yValue


def mzML2npy(raw_mzml):
    itmx = mzml2itmx(raw_mzml)
    np.save(os.path.join(out_dir, 'itmx.npy'), itmx)
    return itmx

def mzml2itmx(raw_mzml):
    rt_start = 0.5
    rt_gap = 0.016
    rt_end = rt_start+1024*rt_gap
    mz_start = 50
    mz_gap = 1
    mz_end = mz_start+1024*mz_gap

    rt_idx = np.arange(rt_start, rt_end, rt_gap)
    # print(len(rt_idx))
    intensities = []
    for rt_time in rt_idx:
        mz_array = get_bin_peak(mz_start, mz_gap, mz_end, raw_mzml, rt_time)
        intensities.append(mz_array)
    return np.array(intensities)


def pool_model(shape=(1024, 1024, 3), pool_size=3):
    input_tensor = layers.Input(shape=shape)
    x = layers.MaxPooling2D(pool_size=(pool_size, pool_size))(input_tensor)
    model = Model(input_tensor, x)
    return model


def pre_pool(matrix):
    pre_model = pool_model()
    matrix = matrix[:, :, :,np.newaxis]
    matrix = np.tile(matrix, [1,1,1,3])
    pool_matrix = pre_model(matrix)
    return pool_matrix


def get_x_samples(datalist, index):
    x_datas = []
    samples = []
    for i in datalist.index:
        data_path = datalist.loc[i, index]
        sample = data_path.split('/')[-1].split('.')[0]

        data_format = data_path.split('.')[-1].lower()

        if 'mzml' in data_format:
            mzml_data = mzml.MzML(data_path)
            intensity_matrix = mzml2itmx(mzml_data)
            x_datas.append(intensity_matrix)
            samples.append(sample)

        elif 'npy' in data_format:
            intensity_matrix = np.load(data_path)
            x_datas.append(intensity_matrix)
            samples.append(sample)

    x_datas = np.array(x_datas)
    samples = np.array(samples)
    return x_datas, samples


def get_labels(datalist, index, num_classes=None):
    labels = []
    for i in datalist.index:
        label = datalist.loc[i, index]
        labels.append(label)
    labels = np.array(labels)
    if num_classes is not None:
        num_classes = np.max(labels)+1
    labels = to_categorical(labels, num_classes)
    return labels


def load_data_pn(datalist_path, num_classes=3, pool=True):
    datalist = pd.read_csv(datalist_path, sep=' ', header=None)

    p_x_datas, p_samples = get_x_samples(datalist, 0)
    n_x_datas, n_samples = get_x_samples(datalist, 1)
    labels = get_labels(datalist, 2, num_classes)

    if pool==True:
        print(p_x_datas.shape)
        print(n_x_datas.shape)
        p_x_datas = pre_pool(p_x_datas)
        n_x_datas = pre_pool(n_x_datas)
    y_data = to_categorical(labels, num_classes)

    return p_x_datas, n_x_datas, y_data, p_samples, n_samples




def load_data(datalist_path, num_classes = 3, pool=True):
    datalist = pd.read_csv(datalist_path, sep=' ', header=None)

    x_datas = []
    labels = []
    samples = []
    for i in datalist.index:
        data_path = datalist.loc[i, 0]
        sample = data_path.split('/')[-1].split('.')[0]

        if len(datalist.columns) == 2:
            label = datalist.loc[i, 1]
        else:
            label = 0

        data_format = data_path.split('.')[-1].lower()

        if 'mzml' in data_format:
            mzml_data = mzml.MzML(data_path)
            intensity_matrix = mzml2itmx(mzml_data)
            x_datas.append(intensity_matrix)
            labels.append(label)
            samples.append(sample)

        elif 'npy' in data_format:
            intensity_matrix = np.load(data_path)
            x_datas.append(intensity_matrix)
            labels.append(label)
            samples.append(sample)

    x_datas = np.array(x_datas)
    labels = np.array(labels)
    samples = np.array(samples)

    if pool==True:
        print(x_datas.shape)
        x_datas = pre_pool(x_datas)
    y_data = to_categorical(labels, num_classes)

    return x_datas, y_data,samples
