
import sys
import platform
import os
import time
import copy
import argparse
import numpy as np
import pandas as pd
from featureExtraction.single_RISE import RISE
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, recall_score, precision_score

from utils import dataTools
from models.basemodel import SingleModel, EMDAM


def get_feature_with_args(model_dir,mode,x_data,y_data,out_path):
    if mode == 'single':
        model_path = os.path.join(model_dir, 'model.h5')
        if not os.path.exists(model_path):
            raise ValueError("model_path not exist")
        model = SingleModel(model_path)

    elif mode == 'ensemble':
        model_list = []
        for model_name in os.listdir(model_dir):
            model_path = os.path.join(model_dir, model_name, 'model.h5')
            if os.path.exists(model_path):
                model_list.append(model_path)
        model = EMDAM(model_list)
    else:
        raise ValueError("mode must be ensemble or single")

    rise = RISE(model, N=1500)
    heatmaps = rise.get_heatmaps(x_data, y_data, N=1000)

    print("[INFO] Heatmap Shape: ",heatmaps.shape)
    print("[INFO] Save Heatmap to: ",out_path)
    np.save(out_path, heatmaps)


def main():
    args = args_setting()
    mode = args.mode
    datalist_path = args.datalist_path
    model_dir = args.model_dir

    out_path = args.out_path
    # load data
    # x_data, y_data, samples = dataTools.load_data(datalist_path)

    # model_dir = '/public6/lilab/student/yjdeng/metaPro/metaTensor/_in_github/experiments/DenseNet121_save/DenseNet121_adam_lr_1.0e-04_7_168621963486'
    data_dir = '/public6/lilab/student/yjdeng/metaPro/metaTensor/_in_github/experiments/DenseNet121_save'
    x_data = np.load(os.path.join(data_dir, 'x_data.npy'))
    y_data = np.load(os.path.join(data_dir, 'y_data.npy'))
    samples = list(np.load(os.path.join(data_dir, 'samples.npy')))

    # load model
    get_feature_with_args(model_dir, mode, x_data, y_data, out_path)


def args_setting():
    parser = argparse.ArgumentParser(description='ArgUtils')
    parser.add_argument('-data', type=str, dest='datalist_path', default='../data/trainlist.txt')
    parser.add_argument('-model', type=str, dest='model_dir', default='../experiments/DenseNet121_save/')
    parser.add_argument('-out', type=str, dest='out_path', default='../experiments/DenseNet121_heatmap.csv')
    parser.add_argument('-mode', type=str, dest='mode', default='ensemble')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    print("RUNing in %s" % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    main()
    print("DONE in %s" % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
