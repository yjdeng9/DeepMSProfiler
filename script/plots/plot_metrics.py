
import platform
if platform.system().lower() == 'linux':
    print('Using Linux')
    import matplotlib
    matplotlib.use('Agg')

import matplotlib.pyplot as plt

import os
import numpy as np
import itertools

from sklearn.metrics import confusion_matrix, roc_curve, auc
from tensorflow.keras.models import load_model


def plot_confusion_matrix(y_true, y_pred_calss, save=None):

    cm = confusion_matrix(y_true,y_pred_calss, labels=[0,1,2])
    cm_sum = np.sum(cm, axis=1)
    cm_sum = np.tile(cm_sum, (3, 1)).T
    cm_rate = cm / cm_sum * 100

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    # plt.imshow(cm, interpolation='nearest', cmap='Reds')

    classes = ["Health","Lung Nodule","Lung Cancer"]
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)
    plt.tick_params(labelsize=15)

    plt.ylabel('True label',size=12, weight = 'bold',fontsize=20)
    plt.xlabel('Predicted label', weight = 'bold',fontsize=20)

    index_collect = np.array([[None, None, None], [None, None, None], [None, None, None]])
    thresh = 50
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, "%d(%.1f%%)" % (cm[i, j], cm_rate[i, j]),
                 horizontalalignment="center",fontsize=15,
                 color="black" if cm_rate[i, j] < thresh else "white")
        for k in range(len(y_true)):
            if y_true[k] == i and y_pred_calss[k] == j:
                index_collect[i, j] = k
                break
    index_collect = index_collect.reshape((-1))

    plt.tight_layout()
    if save is not None:
        plt.savefig(save,dpi=300)
        plt.close()
    else:
        plt.show()


def plot_auc_curve(y_true, y_score, save=None):
    fpr0, tpr0, thresholds0 = roc_curve(y_true[:, 0], y_score[:, 0])
    fpr1, tpr1, thresholds1 = roc_curve(y_true[:, 1], y_score[:, 1])
    fpr2, tpr2, thresholds2 = roc_curve(y_true[:, 2], y_score[:, 2])

    plt.plot(fpr0, tpr0, lw=2, label='Healthy(AUC = %0.2f)' % (auc(fpr0,tpr0)))
    plt.plot(fpr1, tpr1, lw=2, label='Benign (AUC = %0.2f)' % (auc(fpr1, tpr1)))
    plt.plot(fpr2, tpr2, lw=2, label='Malignant (AUC = %0.2f)' % (auc(fpr2, tpr2)))

    plt.plot([0, 1], [0, 1], linestyle='--', lw=1.5, color='grey', label='Random', alpha=.5)
    plt.xlabel('False Positive Rate',fontsize=20)
    plt.ylabel('True Positive Rate',fontsize=20)
    plt.legend(loc="lower right",prop = {'size':15})
    plt.tick_params(labelsize=20)
    plt.tight_layout()

    if save is not None:
        plt.savefig(save, dpi=300)
        plt.close()
    else:
        plt.show()


def main():
    data_dir = '../../experiments/DenseNet121_save'
    x_data = np.load(os.path.join(data_dir, 'x_data.npy'))
    y_data = np.load(os.path.join(data_dir, 'y_data.npy'))
    samples = list(np.load(os.path.join(data_dir, 'samples.npy')))

    model_dir = '../../experiments/DenseNet121_save/DenseNet121_adam_lr_1.0e-04_7_168621963486'
    model = load_model(os.path.join(model_dir, 'model.h5'))

    y_true = np.argmax(y_data, axis=1)
    y_pred = model.predict(x_data)
    y_pred_class = np.argmax(y_pred, axis=1)

    plot_auc_curve(y_data, y_pred)
    # plot_confusion_matrix(y_true, y_pred_class)


if __name__ == '__main__':
    main()