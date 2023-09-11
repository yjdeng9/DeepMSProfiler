
import platform
if platform.system().lower() == 'linux':
    print("run in linux platform!")
    import matplotlib
    matplotlib.use('agg')
import matplotlib.pyplot as plt
import os
import time
import sys
import argparse
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

from pyteomics import mzml

from utils import dataTools


class metaData():
    def __init__(self, mzml_path, point=None, windows=None):
        self.raw_mzml = mzml.MzML(mzml_path)
        self.sample_name = os.path.basename(mzml_path).replace('.mzML', '')
        self.point = point
        self.windows = windows

        rt_gap = self.windows['RT']
        mz_gap = self.windows['MZ']
        self.rt_range = [np.max([self.point['RT'] - rt_gap, 0]), self.point['RT'] + rt_gap]
        self.mz_range = [self.point['MZ'] - mz_gap, self.point['MZ'] + mz_gap]

        self.data_array = self.gen_3D_points()

    def update(self, point, windows):
        self.point = point
        self.windows = windows

    def gen_3D_points(self):
        if self.point is None:
            return None
        self.rt_gap = self.windows['RT']
        self.mz_gap = self.windows['MZ']

        index_range = [self.raw_mzml.time[self.rt_range[0]]['index'], self.raw_mzml.time[self.rt_range[1]]['index']]

        data_array = []
        for index in range(index_range[0], index_range[1]):
            start_time = self.raw_mzml.get_by_index(index)['scanList']['scan'][0]['scan start time']
            mz = self.raw_mzml.get_by_index(index)["m/z array"]
            it = self.raw_mzml.get_by_index(index)["intensity array"]

            if len(mz) != len(it):
                print("Wory: different Length for m/z and intensity!")
            for i in range(len(mz)):
                if mz[i] < self.mz_range[0] or mz[i] > self.mz_range[1]:
                    continue
                if it[i] == 0:
                    continue
                data_array.append([start_time, mz[i], it[i]])

        data_array = np.array(data_array)
        return data_array

    def gen_2D_heatmap(self):
        if self.data_array is None:
            return None

        n = 200
        wing = 1.5
        rt_gap = (self.rt_range[1] - self.rt_range[0]) / n
        mz_gap = (self.mz_range[1] - self.mz_range[0]) / n

        intensities_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                rt = self.rt_range[0] + i * rt_gap
                mz = self.mz_range[0] + j * mz_gap
                rt_true = np.where(
                    np.logical_and(rt - wing * rt_gap <= self.data_array[:, 0], self.data_array[:, 0] <= rt + wing * rt_gap),
                    1,
                    0)
                mz_true = np.where(
                    np.logical_and(mz - wing * mz_gap <= self.data_array[:, 1], self.data_array[:, 1] <= mz + wing * mz_gap),
                    1,
                    0)
                point_true = rt_true + mz_true
                sub_points = self.data_array[point_true == 2, 2]
                if sub_points.shape[0] != 0:
                    intensities_matrix[i, j] = np.max(sub_points)
                    # intensities_matrix[i, j] = np.median(sub_points)

        return intensities_matrix, rt_gap, mz_gap


    def show_3D_points(self,color_fw='black',color_bg=None,save=None):
        forword_array = []
        background_array = []

        # threshold = np.median(self.data_array[:, 2])
        threshold = 0
        data_array = self.data_array.copy()
        data_array[:, 2] = np.log2(data_array[:, 2] + 1)

        for idx in range(data_array.shape[0]):
            if data_array[idx, 2] > threshold:
                forword_array.append(data_array[idx])
            else:
                background_array.append(data_array[idx])

        forword_array = np.array(forword_array)
        background_array = np.array(background_array)

        fig = plt.figure()
        ax = Axes3D(fig)
        if color_bg is not None:
            ax.scatter(background_array[:, 1], background_array[:, 0], background_array[:, 2], s=5, c=color_bg)
        ax.scatter(forword_array[:, 1], forword_array[:, 0], forword_array[:, 2], s=5, c=color_fw)

        ax.set_zlabel('Intensity', fontdict={'size': 15, 'color': 'black'})
        ax.set_xlabel('m/z', fontdict={'size': 15, 'color': 'black'})
        ax.set_ylabel('Retention time', fontdict={'size': 15, 'color': 'black'})

        if save is None:
            plt.show()
        else:
            plt.savefig(save, dpi=300)
            plt.close()

    def show_2d_heatmap(self, save=None, show_labels=True):
        intensity_matrix, rt_gap, mz_gap = self.gen_2D_heatmap()
        intensity_matrix = np.log2(intensity_matrix + 1)

        # plt.figure(figsize=(1.5, 1.5))
        plt.contour(intensity_matrix, cmap='gray')
        plt.imshow(intensity_matrix, cmap='Blues')

        if show_labels:
            plt.xlabel('m/z')
            x_labels = [format(self.mz_range[0] + x * 20 * mz_gap, '.2f') for x in range(10)]
            plt.xticks(np.arange(0, intensity_matrix.shape[1], 20), x_labels, rotation=90)

            plt.ylabel('Retention time')
            y_labels = [format(self.rt_range[0] + y * 20 * rt_gap, '.2f') for y in range(10)]
            plt.yticks(np.arange(0, intensity_matrix.shape[0], 20), np.asarray(y_labels))

            plt.colorbar(label='Intensity')
        else:
            plt.xticks([])
            plt.yticks([])

        plt.tight_layout()

        if save is None:
            plt.show()
        else:
            plt.savefig(save, dpi=300)
            plt.close()

    def get_1D_plot(self, rt_gap=None, save=None):
        if rt_gap is None:
            rt_gap = self.windows['RT']/100

        lines = []
        rt_loc = []

        for i in np.arange(self.rt_range[0], self.rt_range[1], rt_gap):
            rt_loc.append(i)
            rt_true = np.where(
                np.logical_and(i - rt_gap <= self.data_array[:, 0], self.data_array[:, 0] <= i + rt_gap),
                1,
                0)
            sub_points = self.data_array[rt_true == 1, 2]
            if sub_points.shape[0] != 0:
                lines.append(np.max(sub_points))
            else:
                lines.append(0)

        if save is None:
            plt.plot(rt_loc, lines)
            plt.xlabel('Retention time')
            plt.show()
        else:
            line_df = pd.DataFrame({'RT': rt_loc, 'Intensity': lines})
            line_df.to_csv(save, index=False)



def run_single_file():
    ref06_path = 'C:/Users/yong-/Desktop/metaTest/metaTensor/dataVision/Ref06_p.mzML'
    ref23_path = 'C:/Users/yong-/Desktop/metaTest/metaTensor/dataVision/Ref_23_p.mzML'

    refb6_path = 'C:/Users/yong-/Desktop/metaTest/metaTensor/_in_github/data/Ref_03_P.mzML'
    refb7_path = 'C:/Users/yong-/Desktop/metaTest/metaTensor/_in_github/data/Ref_01_P.mzML'


    # point = {'RT': 9, 'MZ': 151.06168}
    point = {'RT': 9, 'MZ': 138.12203}
    windows = {'RT': 1.0, 'MZ': 0.05}

    # load data
    # .raw data > matrix data

    # get 3D point array
    # mzData = metaData(refb6_path, point, windows)
    mzData = metaData(refb7_path, point, windows)
    # mzData.get_1D_plot()
    # mzData.show_2d_heatmap()
    # mzData.show_3D_points()

    mzData.get_1D_plot(save='1D_plot_refb7.csv')
    mzData.show_2d_heatmap(save='heatmap_refb7.svg')
    mzData.show_3D_points(save='heatmap_3D_refb7.svg')


def main():
    point = {'RT': 9, 'MZ': 138.12203}
    # point = {'RT': 9, 'MZ': 151.06168}
    windows = {'RT': 1.0, 'MZ': 0.1}

    mzml_dir = '/public6/lilab/student/triangle/tidy_meta_data/MZML/P'
    out_dir = '/public6/lilab/student/yjdeng/metaPro/metaTensor/dataVision/ref_plot_%.2f_%.2f' % (
    point['RT'], point['MZ'])
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for batch in os.listdir(mzml_dir):
        for file_name in os.listdir(os.path.join(mzml_dir, batch)):
            if 'Ref' not in file_name:
                continue
            mzml_path = os.path.join(mzml_dir, batch, file_name)
            sample = batch + '_' + file_name.replace('.mzML', '')

            mzData = metaData(mzml_path, point, windows)
            mzData.get_1D_plot(save=os.path.join(out_dir, '1D_%s.csv' % sample))
            # mzData.show_3D_points(save=os.path.join(out_dir, '3D_%s.png' % sample))
            # mzData.show_2d_heatmap(save=os.path.join(out_dir, 'heatmap_%s.png' % sample))
            mzData.show_2d_heatmap(save=os.path.join(out_dir, 'heatmap_%s.svg' % sample))


def args_setting():
    parser = argparse.ArgumentParser(description='ArgUtils')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    print("RUNing in %s" % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    args = args_setting()
    main()
    # run_single_file()
    print("DONE in %s" % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))