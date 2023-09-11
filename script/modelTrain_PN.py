

import sys
import platform
if platform.system().lower() == 'linux':
    print("run in linux platform!")
import os
import time
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from utils import dataTools
from utils import callback
from utils import loadKerasModel
from utils import loadPNModel


def split_with_samples(x_data, y_data, samples, test_size=0.2, seed=None):
    samples_index = np.array(list(range(0, y_data.shape[0])))

    print(y_data.shape)
    print(samples_index.shape)

    index_train, index_test, y_train, y_test = train_test_split(samples_index, y_data, stratify=y_data,test_size=test_size, random_state=seed)
    x_train = x_data[index_train]
    x_test = x_data[index_test]
    samples_train = [samples[i] for i in index_train]
    samples_test = [samples[i] for i in index_test]

    return x_train,x_test, y_train, y_test, samples_train, samples_test


def save_split_data(samples_train, samples_val,save_dir):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    with open(os.path.join(save_dir, 'samples_train.txt'), 'w') as fw:
        fw.write('\n'.join(samples_train))
    with open(os.path.join(save_dir, 'samples_val.txt'), 'w') as fw:
        fw.write('\n'.join(samples_val))


def get_run_log(history):
    run_log = pd.DataFrame()
    run_log['epoch'] = list(range(1, 1 + len(history.history['loss'])))
    run_log['loss'] = history.history['loss']
    run_log['accuracy'] = history.history['accuracy']
    run_log['val_loss'] = history.history['val_loss']
    run_log['val_accuracy'] = history.history['val_accuracy']
    return run_log


def save_args(args):
    args_dict = vars(args)
    args_dict['script'] = sys.argv[0]
    args_dict['time'] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    args_dict['platform'] = platform.platform()
    args_dict['python'] = platform.python_version()
    print(args_dict)

    # save args to csv
    args_df = pd.DataFrame()
    args_df['key'] = args_dict.keys()
    args_df['value'] = args_dict.values()

    # args_df.to_csv(os.path.join(args.save_dir, 'args.csv'), index=False, sep='\t')


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


def run_model_with_args(model, x_data, y_data, samples, args, run_args, save_dir=None, batch_size=8,
                        threshold=0.99, verbose=0):
    print('[INFO] run model with args: %s' % (run_args.replace('_',' ')))
    epoch = args.epoch
    args_df = args.args_df

    callbacks = [callback.EarlyStoppingWithThreshold(monitor='val_accuracy', threshold=threshold, patience=10)]

    x_train, x_val, y_train, y_val, samples_train, samples_val = split_with_samples(x_data, y_data, samples, test_size=0.25)
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epoch, validation_data=(x_val, y_val),
                        shuffle=True, callbacks=callbacks, verbose=verbose)
    run_log = get_run_log(history)
    print('[INFO] run log:')
    print(run_log)

    model_id = str(int(time.time()))+str(np.random.randint(0,1000))
    if save_dir is not None:
        model_savedir = os.path.join(save_dir, '%s_%s' % (run_args, model_id))
        if not os.path.exists(model_savedir):
            os.mkdir(model_savedir)

        args_path = os.path.join(model_savedir, 'args.txt')
        args_df.to_csv(args_path, index=False, sep='\t')

        log_path = os.path.join(model_savedir, 'log.txt')
        run_log.to_csv(log_path, index=False, sep='\t')

        save_threshold = 0.1
        last_acc = history.history['accuracy'][-1]
        last_val_acc = history.history['val_accuracy'][-1]
        if last_acc > save_threshold and last_val_acc > save_threshold:
            model_path = os.path.join(model_savedir, 'model.h5')
            model.save(model_path)
            save_split_data(samples_train, samples_val, model_path.replace('.h5', ''))


def main():
    args = args_setting()
    datalist_path = args.datalist_path

    args = add_args_df(args)

    print('[INFO] args is:')
    print(args.args_df)

    model_name = args.model_name
    pretrain_path = args.pretrain_path

    run_time = args.runtime
    optimizer = args.optimizer
    lr = args.lr
    epoch = args.epoch
    batch_size = args.batch_size

    save_dir = os.path.join(args.save_dir, '%s_PN_save' % (args.model_name))

    if save_dir is not None and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    datalist_path = '../example/data/trainlist_pn.txt'

    # load data
    # .raw data > matrix data


    # p_x_datas, n_x_datas, y_data, p_samples, n_samples = dataTools.load_data_pn(datalist_path)
    # print('[INFO] data shape:')
    # print('P_X_Data ', p_x_datas.shape)
    # print('N_X_Data ', n_x_datas.shape)
    # print('Y_Data ', y_data.shape)
    # print('P_Samples ', len(p_samples))
    # print('N_Samples ', len(n_samples))
    #
    # np.save(os.path.join(save_dir, 'p_x_datas.npy'), p_x_datas)
    # np.save(os.path.join(save_dir, 'n_x_datas.npy'), n_x_datas)
    # np.save(os.path.join(save_dir, 'y_data.npy'), y_data)
    # np.save(os.path.join(save_dir, 'p_samples.npy'), np.array(p_samples))
    # np.save(os.path.join(save_dir, 'n_samples.npy'), np.array(n_samples))

    p_x_datas = np.load(os.path.join(save_dir, 'p_x_datas.npy'))
    n_x_datas = np.load(os.path.join(save_dir, 'n_x_datas.npy'))
    y_data = np.load(os.path.join(save_dir, 'y_data.npy'))
    p_samples = np.load(os.path.join(save_dir, 'p_samples.npy'))
    n_samples = np.load(os.path.join(save_dir, 'n_samples.npy'))

    samples = ['%s_%s' % (ps,ns) for ps,ns in zip(p_samples, n_samples)]

    input_shape = p_x_datas.shape[1:]

    for rt in range(run_time):
        model = loadPNModel.load_model(model_name, input_shape=input_shape)
        try:
            if pretrain_path is not None:
                model.load_weights(pretrain_path)
        except:
            print('[WORRY] Invalid pretrain model path!')
        model = loadKerasModel.compile_model(model, optimizer=optimizer, lr=lr)

        run_args = '%s_%s_lr_%.1e_%d' % (model_name, optimizer, lr, rt)
        run_model_with_args(model, [p_x_datas,n_x_datas], y_data, samples, args, run_args, save_dir=save_dir,
                            batch_size=batch_size)


def args_setting():
    parser = argparse.ArgumentParser(description='ArgUtils')
    parser.add_argument('-data', type=str, dest='datalist_path', default='../data/trainlist_pn.txt')
    parser.add_argument('-model', type=str, dest='model_name', default='DenseNet121')
    parser.add_argument('-pretrain',dest='pretrain_path', default=None)

    parser.add_argument('-lr', type=float, dest='lr', default=1e-4)
    parser.add_argument('-opt', type=str, dest='optimizer', default='adam')
    parser.add_argument('-batch', type=int, dest='batch_size', default=8)
    parser.add_argument('-epoch', type=int, dest='epoch', default=200)
    parser.add_argument('-run', type=int, dest='runtime', default=10)

    parser.add_argument('-save', type=str, dest='save_dir', default='../experiments/PN')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    print("RUNing in %s" % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    main()
    print("END in %s" % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
