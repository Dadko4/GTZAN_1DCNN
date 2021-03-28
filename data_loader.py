import numpy as np
from scipy.io import wavfile
from glob import glob
import os
from tqdm import tqdm
from tensorflow.keras.utils import to_categorical
np.random.seed(1)


class DataLoader:
    
    def __init__(self, window_size=512, overleap=256,
                 regex_path=r'./genres_original/**/*.wav'):
        self.window_size = window_size
        self.overleap = overleap
        self.regex_path = regex_path
        self.data = None

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
        self._data_list = np.array(data_list)[idx]
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
        print("Scaling: ", end="")
        if fit:
            self._mean = np.mean(X)
            self._std = np.std(X)
        print("mean and std computing done, scaling arrs: ", end="")
        X = (X - self._mean) / self._std
        print("Done")
        return X

    def _window(self, a, window_size=512, overleap=256):
        shape = (a.size - window_size + 1, window_size)
        strides = a.strides * 2
        view = np.lib.stride_tricks.as_strided(a, strides=strides,
                                               shape=shape)[0::overleap]
        return view

    def get_data(self, val_size=0.1, test_size=0.1, scale=True):
        if self.data is None:
            self._init_data()
        vs = 1 - (test_size + val_size)
        ts = 1 - test_size
        print("Splitting: ", end="")
        train_X, val_X, test_X = np.split(self._data_list,
                                          [int(vs*len(self._data_list)),
                                           int(ts*len(self._data_list))])
        train_y, val_y, test_y = np.split(self._genres, [int(vs*len(self._genres)),
                                                         int(ts*len(self._genres))])
        print("Done")
        train_X, train_y = self._make_windows(train_X, train_y)
        train_X = self._scale(train_X, fit=True)
        val_X, val_y = self._make_windows(val_X, val_y)
        val_X = self._scale(val_X)
        test_X, test_y = self._make_windows(test_X, test_y)
        test_X = self._scale(test_X)

        return train_X, val_X, test_X, train_y, val_y, test_y
