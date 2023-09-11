
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.transform import resize
import keras
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from keras import backend as K
from keras.preprocessing import image
from keras.applications.imagenet_utils import decode_predictions
from tensorflow.keras.models import load_model


class RISE:
    def __init__(self, model, N=1500, s=8, p1=0.1, batch_size=128):
        self.model = model
        self.input_size = model.input_size
        self.N = N
        self.s = s
        self.p1 = p1
        self.batch_size = batch_size

        self.masks = self.generate_masks()


    def generate_masks(self, N=None):
        if N is None:
            N = self.N
        cell_size = np.ceil(np.array(self.input_size) / self.s)
        up_size = (self.s + 1) * cell_size

        grid = np.random.rand(N, self.s, self.s) < self.p1
        grid = grid.astype('float32')

        masks = np.empty((N, *self.input_size))
        for i in range(N):
            # Random shifts
            x = np.random.randint(0, cell_size[0])
            y = np.random.randint(0, cell_size[1])
            # Linear upsampling and cropping
            masks[i, :, :] = resize(grid[i], up_size, order=1, mode='reflect',
                                    anti_aliasing=False)[x:x + self.input_size[0], y:y + self.input_size[1]]
        masks = masks.reshape(-1, *self.input_size, 1)
        return masks

    def _get_sub_masks(self, N, resample=False):
        if resample:
            sub_idx = np.random.choice(list(range(self.masks.shape[0])), N, replace=True)
        else:
            sub_idx = random.sample(list(range(self.masks.shape[0])), N)
        sub_masks = self.masks[sub_idx]
        return sub_masks

    def _explain(self, x, masks):
        preds = []
        masked = x * masks

        N = masked.shape[0]

        for i in range(0, N, self.batch_size):
            preds.append(self.model.predict(masked[i:min(i + self.batch_size, N)]))
        preds = np.concatenate(preds)

        sal = preds.T.dot(masks.reshape(N, -1)).reshape(-1, *self.input_size)
        sal = sal / N /self.p1
        return sal

    def get_heatmaps(self, x_data,y_data=None,N=1000,resample=False):
        if y_data is not None:
            labels = np.argmax(y_data, axis=1)
        else:
            y_data = self.model.predict(x_data)
            labels = np.argmax(y_data, axis=1)

        if N > self.masks.shape[0]:
            self.masks = self.generate_masks(N)

        heatmaps = []
        for idx in range(x_data.shape[0]):
            x = x_data[idx]
            class_idx = labels[idx]
            sub_masks = self._get_sub_masks(N, resample=resample)
            sal = self._explain(x, sub_masks)
            heatmap = sal[class_idx]
            heatmaps.append(heatmap)

        # 3. heatmaps结果保存
        heatmaps = np.array(heatmaps)
        return heatmaps






