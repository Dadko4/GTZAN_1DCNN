import numpy as np
from scipy.io import wavfile
from glob import glob
import os
from tqdm import tqdm
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import KFold
np.random.seed(1)


class DataLoader:

    def __init__(self, window_size=512, overleap=256,
                 regex_path=r'./genres_original/**/*.wav'):
        self.window_size = window_size
        self.overleap = overleap
        self.regex_path = regex_path
        self._data = None
        self.test_y = None
        self._test_size = 0.1

    def _init_data(self, shuffle=True):
        data_list = []
        genres = []
        for fname in glob(self.regex_path):
            try:
                _, data = wavfile.read(fname)
                data_list.append(data)
                dir_ = os.path.dirname(fname)
                genres.append(os.path.basename(dir_))
            except ValueError:
                continue
        self.label2idx = {l: idx for idx, l in enumerate(np.unique(genres))}
        idx = np.random.randint(0, len(data_list), len(data_list))
        self._data = np.array(data_list)[idx]
        self._genres = np.array(genres)[idx]

    def _make_windows(self, data_list, genres, shuffle=True):
        all_windows = []
        all_labels = []
        for datapoint, label in tqdm(zip(data_list, genres)):
            windows = self._window(datapoint,
                                   window_size=self.window_size,
                                   overleap=self.overleap)
            all_windows.append(windows)
            all_labels.extend([label] * len(windows))

        data = np.expand_dims(np.vstack(all_windows), axis=-1)
        labels = np.array([self.label2idx[label] for label in all_labels])
        if shuffle:
            idx = np.random.randint(0, len(data), len(data))
            data = data[idx]
            labels = labels[idx]
        return data, to_categorical(labels)

    def _scale(self, X, fit=False):
        if fit:
            self._mean = np.mean(X)
            self._std = np.std(X)
        X = (X - self._mean) / self._std
        return X

    def _window(self, a, window_size=512, overleap=256):
        shape = (a.size - window_size + 1, window_size)
        strides = a.strides * 2
        view = np.lib.stride_tricks.as_strided(a, strides=strides,
                                               shape=shape)[0::overleap]
        return view

    def _train_test_split(self, test_size, preprocess_test=True):
        print("train test split...", end="")
        self._test_size = test_size
        ts = 1 - test_size
        self.train_X, self.test_X = np.split(self._data, [int(ts*len(self._data))])
        self.train_y, self.test_y = np.split(self._genres, [int(ts*len(self._data))])
        if preprocess_test:
            train_X, _ = self._make_windows(self.train_X, self.train_y)
            _ = self._scale(train_X, fit=True)
            test_X, self.test_y = self._make_windows(self.test_X, self.test_y)
            self.test_X = self._scale(test_X)

    def get_folds(self, test_size=0.1, cv=5, scale=True):
        if self._data is None:
            self._init_data()
            print("init data done")

        if self.test_y is None or self._test_size != test_size:
            self._train_test_split(test_size, preprocess_test=True)
            print("done")

        kf = KFold(n_splits=cv)
        for train_index, val_index in kf.split(self.train_X):
            train_X = self.train_X[train_index]
            train_y = self.train_y[train_index]
            train_X, train_y = self._make_windows(train_X, train_y)
            train_X = self._scale(train_X, fit=True)

            val_X = self.train_X[val_index]
            val_y = self.train_y[val_index]
            val_X, val_y = self._make_windows(val_X, val_y)
            val_X = self._scale(val_X)

            yield train_X, val_X, train_y, val_y

    def get_test_data(self, test_size=0.1):
        if self.test_y is None or self._test_size != test_size:
            self._train_test_split(test_size, preprocess_test=True)
        return self.test_X, self.test_y
