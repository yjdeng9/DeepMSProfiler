
import os
import numpy as np
import matplotlib.pyplot as plt


def main():
    job_name = 'jobs007'
    out_dir = '../jobs'
    mode = 'ensemble'

    job_dir = os.path.join(out_dir, job_name)
    feature_path = os.path.join(job_dir, 'feature_results', '%s_RISE.npy' % mode)
    labels_path = '../example/label.txt'

    # 1. 加载数据
    feature = np.load(feature_path)
    with open(labels_path, 'r') as f:
        labels = f.readlines()
    labels = [label.strip() for label in labels]
    labels = np.array(labels)
    print('Labels >>>')
    print(labels)
    print('Shape of Feature: %s' % str(feature.shape))

    unique_labels = np.unique(labels)
    print(unique_labels)

    i=1
    for label in unique_labels:
        plt.subplot(1, len(unique_labels), i)
        label_idx = np.where(labels==label)
        print(label_idx)
        label_feature = feature[label_idx]
        label_feature_mean = np.mean(label_feature, axis=0)
        label_feature_mean = np.flip(label_feature_mean, axis=0)

        plt.imshow(label_feature_mean, cmap='jet', interpolation='nearest')
        plt.title(label)
        plt.xticks([])
        plt.yticks([])
        plt.xlabel('m/z')
        plt.ylabel('RT')
        i=i+1
    print(feature.shape)
    plt.savefig(os.path.join(job_dir, 'feature_results', '%s_RISE.png' % mode))
    plt.show()


if __name__ == '__main__':
    main()