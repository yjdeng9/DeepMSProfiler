
import sys
import platform
import os
import time
import copy
import argparse
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, recall_score, precision_score

from utils import dataTools
from models.basemodel import SingleModel, EMDAM
from plots.plot_metrics import plot_confusion_matrix, plot_auc_curve


def spe(con_mat, n=3):
    spec = []
    for i in range(n):
        number = np.sum(con_mat[:, :])
        tp = con_mat[i][i]
        fn = np.sum(con_mat[i, :]) - tp
        fp = np.sum(con_mat[:, i]) - tp
        tn = number - tp - fn - fp
        spe1 = tn / (tn + fp)
        spec.append(spe1)

    return np.mean(spec)

def sen(con_mat, n=3):  # n为分类数
    sent = []
    for i in range(n):
        tp = con_mat[i][i]
        fn = np.sum(con_mat[i, :]) - tp
        sen1 = tp / (tp + fn)
        sent.append(sen1)

    return np.mean(sent)

def count_metrics(model, x_data, y_data):
    y_probs = model.predict(x_data)
    y_pred = np.argmax(y_probs, axis=1)
    y_true = np.argmax(y_data, axis=1)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

    acc_score = accuracy_score(y_true, y_pred)
    auc_score = roc_auc_score(y_data, y_probs)

    recall = recall_score(y_true, y_pred, average='macro')
    precision = precision_score(y_true, y_pred, average='macro')

    sensitivity = sen(cm)
    specificity = spe(cm)

    return acc_score, auc_score, precision, recall, sensitivity, specificity


def pred_by_args(x_data, y_data, model_dir, mode='ensmeble', boost=False,plot_cm=False, plot_auc=False,save_dir = None):

    # load model
    if mode == 'single':
        model_path = os.path.join(model_dir, 'model.h5')
        if not os.path.exists(model_path):
            raise ValueError("model_path not exist")
        model = SingleModel(model_path)

    elif mode == 'ensemble':
        model_list = []

        # print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>debuging>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        # print(model_dir)
        # print('')

        for model_name in os.listdir(model_dir):
            model_path = os.path.join(model_dir, model_name, 'model.h5')
            if os.path.exists(model_path):
                model_list.append(model_path)
        model = EMDAM(model_list)
    else:
        raise ValueError("mode must be ensemble or single")

    if boost:
        print("[INFO] boosting...")
        n_iter = 10
        rate = 0.8
        N = x_data.shape[0]
        metric_df = []
        for i in range(n_iter):
            sub_index = np.random.choice(list(range(N)),int(rate*N), replace=False)
            acc_score, auc_score, precision, recall, sensitivity, specificity = count_metrics(model, x_data[sub_index], y_data[sub_index])
            metric_df.append([acc_score, auc_score, precision, recall, sensitivity, specificity])
        metric_df = pd.DataFrame(np.array(metric_df),columns=['acc_score', 'auc_score', 'precision', 'recall', 'sensitivity', 'specificity'])
        print('[INFO] boosting result:')
        print('acc_score: %.4f[%.4f,%.4f]' % (metric_df['acc_score'].mean(), metric_df['acc_score'].min(), metric_df['acc_score'].max()))
        print('auc_score: %.4f[%.4f,%.4f]' % (metric_df['auc_score'].mean(), metric_df['auc_score'].min(), metric_df['auc_score'].max()))
        print('precision: %.4f[%.4f,%.4f]' % (metric_df['precision'].mean(), metric_df['precision'].min(), metric_df['precision'].max()))
        print('recall: %.4f[%.4f,%.4f]' % (metric_df['recall'].mean(), metric_df['recall'].min(), metric_df['recall'].max()))
        print('sensitivity: %.4f[%.4f,%.4f]' % (metric_df['sensitivity'].mean(), metric_df['sensitivity'].min(), metric_df['sensitivity'].max()))
        print('specificity: %.4f[%.4f,%.4f]' % (metric_df['specificity'].mean(), metric_df['specificity'].min(), metric_df['specificity'].max()))
    else:
        acc_score, auc_score, precision, recall, sensitivity, specificity = count_metrics(model, x_data, y_data)
        metric_df = [[acc_score, auc_score, precision, recall, sensitivity, specificity]]
        metric_df = pd.DataFrame(np.array(metric_df), columns=['acc_score', 'auc_score', 'precision', 'recall', 'sensitivity','specificity'])
        print("[INFO] result:")
        print("acc_score: %.4f" % acc_score)
        print("auc_score: %.4f" % auc_score)
        print("precision: %.4f" % precision)
        print("recall: %.4f" % recall)
        print("sensitivity: %.4f" % sensitivity)
        print("specificity: %.4f" % specificity)

    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, 'metric.csv')
        metric_df.to_csv(save_path, index=False)

    if plot_auc:
        print("[INFO] plot auc curve...")
        y_pred = model.predict(x_data)
        save_path = os.path.join(save_dir, 'auc_curve.pdf') if save_dir is not None else None
        plot_auc_curve(y_data, y_pred,save_path)

    if plot_cm:
        print("[INFO] plot confusion matrix...")
        y_pred = model.predict(x_data)
        y_true = np.argmax(y_data, axis=1)
        y_pred_class = np.argmax(y_pred, axis=1)
        save_path = os.path.join(save_dir, 'cm.pdf') if save_dir is not None else None
        plot_confusion_matrix(y_true, y_pred_class, save_path)


def main():
    args = args_setting()
    mode = args.mode
    datalist_path = args.datalist_path
    model_dir = args.model_dir
    save_path = args.save_path
    boost = args.boost
    plot_auc = args.plot_auc
    plot_cm = args.plot_cm

    # load data
    # x_data, y_data, samples = dataTools.load_data(datalist_path)
    data_dir = '/public6/lilab/student/yjdeng/metaPro/metaTensor/_in_github/experiments/DenseNet121_save'
    data_dir = '../experiments/DenseNet121_save'
    x_data = np.load(os.path.join(data_dir, 'x_data.npy'))
    y_data = np.load(os.path.join(data_dir, 'y_data.npy'))
    samples = list(np.load(os.path.join(data_dir, 'samples.npy')))

    model_dir = '/public6/lilab/student/yjdeng/metaPro/metaTensor/_in_github/experiments/DenseNet121_save/DenseNet121_adam_lr_1.0e-04_7_168621963486'
    model_dir = '../experiments/DenseNet121_save/DenseNet121_adam_lr_1.0e-04_7_168621963486'
    mode = 'single'

    # model_dir = '../experiments/DenseNet121_save'
    # mode = 'ensemble'

    pred_by_args(x_data, y_data, model_dir, mode, boost, plot_cm, plot_auc)





def args_setting():
    parser = argparse.ArgumentParser(description='ArgUtils')
    parser.add_argument('-data', type=str, dest='datalist_path', default='../data/trainlist.txt')
    parser.add_argument('-models', type=str, dest='model_dir', default='../experiments/DenseNet121_save/')
    parser.add_argument('-save', type=str, dest='save_path', default='../experiments/DenseNet121_eval.csv')
    parser.add_argument('-mode', type=str, dest='mode', default='ensemble')
    parser.add_argument('-boost', dest='boost', action='store_true', help='boosting mode')
    parser.add_argument('plot_auc', dest='plot_auc', action='store_true', help='plot auc curve')
    parser.add_argument('plot_cm', dest='plot_cm', action='store_true', help='plot confusion matrix')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    print("RUNing in %s" % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    main()
    print("DONE in %s" % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
