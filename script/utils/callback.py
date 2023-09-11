import numpy as np
import logging
from tensorflow.keras.callbacks import Callback

class EarlyStoppingWithThreshold(Callback):
    def __init__(self, monitor='val_accuracy', monitor2='accuracy', threshold=0.99, mode='auto', min_delta=0,
                 patience=0, baseline=None, restore_best_weights=False, verbose=0):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.monitor2 = monitor2
        self.threshold = threshold
        self.verbose = verbose

        self.patience = patience
        self.baseline = baseline
        self.min_delta = abs(min_delta)
        self.wait = 0
        self.stopped_epoch = 0
        self.restore_best_weights = restore_best_weights
        self.best_weights = None

        if mode not in ['auto', 'min', 'max']:
            logging.warning('EarlyStopping mode %s is unknown, '
                            'fallback to auto mode.', mode)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
        elif mode == 'max':
            self.monitor_op = np.greater
        else:
            if 'acc' in self.monitor:
                self.monitor_op = np.greater
            else:
                self.monitor_op = np.less

        if self.monitor_op == np.greater:
            self.min_delta *= 1
        else:
            self.min_delta *= -1

    def on_train_begin(self, logs=None):
        # Allow instances to be re-used
        self.wait = 0
        self.stopped_epoch = 0
        if self.baseline is not None:
            self.best = self.baseline
        else:
            self.best = np.Inf if self.monitor_op == np.less else -np.Inf
        self.best_weights = None

    def on_epoch_end(self, epoch, logs=None):
        current = self.get_monitor_value(logs)
        if current is None:
            return

        current2 = self.get_monitor2_value(logs)
        stop_basement = False
        if self.monitor_op == np.less:
            if current < self.threshold and current2 < self.threshold:
                stop_basement = True
        else:
            if current > self.threshold and current2 > self.threshold:
                stop_basement = True

        if self.monitor_op(current - self.min_delta, self.best):
            self.best = current
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience and stop_basement:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                if self.restore_best_weights:
                    if self.verbose > 0:
                        print('Restoring model weights from the end of the best epoch.')
                    self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0 and self.verbose > 0:
            print('Epoch %05d: early stopping' % (self.stopped_epoch + 1))

    def get_monitor_value(self, logs):
        logs = logs or {}
        monitor_value = logs.get(self.monitor)
        if monitor_value is None:
            logging.warning('Early stopping conditioned on metric `%s` '
                            'which is not available. Available metrics are: %s',
                            self.monitor, ','.join(list(logs.keys())))
        return monitor_value

    def get_monitor2_value(self, logs):
        logs = logs or {}
        monitor_value = logs.get(self.monitor2)
        if monitor_value is None:
            logging.warning('Early stopping conditioned on metric `%s` '
                            'which is not available. Available metrics are: %s',
                            self.monitor2, ','.join(list(logs.keys())))
        return monitor_value