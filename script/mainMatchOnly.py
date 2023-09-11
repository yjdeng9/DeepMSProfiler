
import sys
import platform
if platform.system().lower() == 'linux':
    print("[INFO] run in linux platform!")
import os
import time
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from pyteomics import mzml
from modelTrain import run_model_with_args
from modelPred import pred_by_args
from modelFeature import get_feature_with_args

import shutil

from utils import dataTools
from utils import callback
from utils import loadKerasModel


def add_args_df(args):
    args.script_path = sys.argv[0]
    args.platform = platform.platform()

    # save args to csv
    args_dict = vars(args)
    args_df = pd.DataFrame()
    args_df['Parameter'] = args_dict.keys()
    args_df['Value'] = args_dict.values()
    args.args_df = args_df

    return args



def step1(args, job_dir):
    # Step1 : load data
    # 在第一个步骤中，用户会上传一系列文件与对应的标签。如果选择match按钮，那么我们需要对数据进行匹配，即将数据与标签进行匹配。

    data_dir = args.data_dir  # 用户上传文件文件夹
    label_path = args.label_path  # 用户上传标签文件
    split_path = args.split_path  # 用户上传数据集划分文件
    random_split = args.random_split  # 是否随机划分数据集

    # 读取标签文件
    if label_path is None:
        labels = [0 for i in range(len(os.listdir(data_dir)))]
    else:
        labels = []
        with open(label_path, 'r') as f:
            for line in f.readlines():
                labels.append(line.strip())

    # 读取数据集划分文件
    if random_split:
        train_idx, test_idx = train_test_split(range(len(os.listdir(data_dir))), test_size=0.2, random_state=42)
        splits = ['train' if i in train_idx else 'test' for i in range(len(os.listdir(data_dir)))]
    elif split_path is not None:
        splits = []
        with open(split_path, 'r') as f:
            for line in f.readlines():
                splits.append(line.strip())
    else:
        splits = [None for i in range(len(os.listdir(data_dir)))]

    # 匹配数据、标签与数据集划分
    # 【Note】需要在网页版本中注意一下数据的排序问题
    match_df = []
    for file_name, label, split in zip(os.listdir(data_dir), labels, splits):
        match_df.append([file_name, label, split])

    match_df = pd.DataFrame(match_df, columns=['FilePath', 'Label', 'Dataset'])
    print(match_df)
    match_df.to_csv(os.path.join(job_dir, 'match_data_label.csv'), index=False)

    # 检查数据格式，如果数据格式不为npy或者mzML，那么需要进行转换

    job_data_dir = os.path.join(job_dir, 'data')
    if not os.path.exists(job_data_dir):
        os.makedirs(job_data_dir)

    for file_name in match_df['FilePath']:
        if file_name.endswith('.npy'):
            shutil.copy(os.path.join(data_dir, file_name), os.path.join(job_data_dir, file_name))
        elif file_name.endswith('.mzML'):
            raw_mzml = mzml.MzML(os.path.join(data_dir, file_name))
            intensity_matrix = dataTools.mzml2itmx(raw_mzml)
            np.save(os.path.join(job_data_dir, file_name), intensity_matrix)
        else:
            print(
                "Error: %s is not a .mzML file! (change data to .mzML format by MSCovert or other softwares.)" % file_name)
            continue


def load_jobs_data(job_dir, num_classes=3):
    match_df = pd.read_csv(os.path.join(job_dir, 'match_data_label.csv'))
    label2num = {'health': 0, 'nodule': 1, 'cancer': 2}
    match_df['label_num'] = [label2num[i] for i in match_df['Label']]

    train_data = []
    train_label = []
    train_samples = []
    test_data = []
    test_label = []
    test_samples = []
    for i in range(len(match_df)):
        file_name = match_df.loc[i, 'FilePath']
        file_path = os.path.join(job_dir, 'data', file_name)
        x_data = np.load(file_path)
        label = match_df.loc[i, 'label_num']
        dataset = match_df.loc[i, 'Dataset']
        sample_name = file_name.split('.')[0]

        if dataset == 'train':
            train_data.append(x_data)
            train_label.append(label)
            train_samples.append(sample_name)
        elif dataset == 'test':
            test_data.append(x_data)
            test_label.append(label)
            test_samples.append(sample_name)
        else:
            train_data.append(x_data)
            train_label.append(label)
            train_samples.append(sample_name)
            test_data.append(x_data)
            test_label.append(label)
            test_samples.append(sample_name)

    train_data,train_label,train_samples = np.array(train_data),np.array(train_label),np.array(train_samples)
    test_data,test_label,test_samples = np.array(test_data),np.array(test_label),np.array(test_samples)

    train_label = dataTools.to_categorical(train_label, num_classes)
    test_label = dataTools.to_categorical(test_label, num_classes)

    return train_data, train_label, train_samples, test_data, test_label , test_samples


def step2_train(args, job_dir,train_data, train_label, train_samples):
    run_time = args.run_time
    model_name = args.architecture
    pretrain_path = args.pretrain_path
    optimizer = args.optimizer
    lr = args.lr
    batch_size = args.batch_size
    input_shape = (train_data.shape[1], train_data.shape[2], train_data.shape[3])

    save_dir = os.path.join(job_dir, 'models_run')

    args = add_args_df(args)
    for rt in range(run_time):
        model = loadKerasModel.load_imagenet_model(model_name, input_shape=input_shape)
        try:
            if pretrain_path is not None:
                model.load_weights(pretrain_path)
        except:
            print('[WORRY] Invalid pretrain model path!')
        model = loadKerasModel.compile_model(model, optimizer=optimizer, lr=lr)

        run_args = '%s_%s_lr_%.1e_%d' % (model_name, optimizer, lr, rt)
        run_model_with_args(model, train_data, train_label, train_samples, args, run_args, save_dir=save_dir,
                            batch_size=batch_size)


def step2_pred(args, job_dir, test_data, test_label, test_samples):
    model_dir = args.models_for_pred
    if model_dir == 'use_old':
        model_dir = os.path.join(job_dir, 'models_run')
    mode = args.mode
    boost = args.boost
    plot_auc = args.plot_auc
    plot_cm = args.plot_cm
    save_dir = os.path.join(job_dir, 'pred_results')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    pred_by_args(test_data, test_label, model_dir, mode, boost, plot_cm, plot_auc, save_dir)

def step2_feature(args, job_dir, test_data, test_label, test_samples):
    model_dir = args.models_for_pred
    if model_dir == 'use_old':
        model_dir = os.path.join(job_dir, 'models_run')
    mode = args.mode
    x_data = test_data
    y_data = test_label
    out_path = os.path.join(job_dir, 'feature_results', '%s_RISE.npy' % mode)

    if not os.path.exists(os.path.join(job_dir, 'feature_results')):
        os.makedirs(os.path.join(job_dir, 'feature_results'))

    print(model_dir)

    get_feature_with_args(model_dir, mode, x_data, y_data, out_path)


def main():
    args = args_setting()
    # args.data_dir = 'C:/Users/yong-/Desktop/metaTest/metaTensor/_code_ocean/data/LCMS/denoise/batch1/cancer'
    args.data_dir = '../../data/P/META/batch1/cancer/'

    # 创建一个文件夹用于工作目录
    job_name = 'jobs007'
    job_dir = os.path.join('../jobs', job_name)
    if not os.path.exists(job_dir):
        os.makedirs(job_dir)

    print('[INFO] Start in %s!' % job_dir)
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    # Step1 : load data
    step1(args, job_dir)

    print('[INFO] Step1 Done!')
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))


def args_setting():
    parser = argparse.ArgumentParser(description='ArgUtils')

    parser.add_argument('-data', type=str, dest='data_dir', default='../example/data/')
    parser.add_argument('-label', type=str, dest='label_path', default='../example/label.txt')
    parser.add_argument('-split', type=str, dest='split_path', default=None)
    parser.add_argument('-random_split', type=bool, dest='random_split', default=False)

    parser.add_argument('-run_train', type=bool, dest='run_train', default=False)
    parser.add_argument('-run_pred', type=bool, dest='run_pred', default=False)
    parser.add_argument('-run_feature', type=bool, dest='run_feature', default=True)

    parser.add_argument('-arch', type=str, dest='architecture', default='DenseNet121')
    parser.add_argument('-pretrain',dest='pretrain_path', default=None)
    parser.add_argument('-lr', type=float, dest='lr', default=1e-4)
    parser.add_argument('-opt', type=str, dest='optimizer', default='adam')
    parser.add_argument('-batch', type=int, dest='batch_size', default=8)
    parser.add_argument('-epoch', type=int, dest='epoch', default=2)
    parser.add_argument('-run', type=int, dest='run_time', default=10)

    parser.add_argument('-models', type=str, dest='models_for_pred', default='use_old')
    parser.add_argument('-mode', type=str, dest='mode', default='ensemble')
    parser.add_argument('-boost', dest='boost', action='store_true', help='boosting mode')
    parser.add_argument('-plot_auc', dest='plot_auc', action='store_true', help='plot auc curve')
    parser.add_argument('-plot_cm', dest='plot_cm', action='store_true', help='plot confusion matrix')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()
