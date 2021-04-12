import numpy as np
from scipy.io import wavfile
from glob import glob
import os
from tqdm import tqdm
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
np.random.seed(1)


class DataLoader:
    
    def __init__(self, window_size=512, overleap=256,
                 regex_path=r'./genres_original/**/*.wav'):
        self.window_size = window_size
        self.overleap = overleap
        self.regex_path = regex_path
        self.test_song_mapper = []
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
        idx = np.arange(0, len(data_list))
        np.random.shuffle(idx)
        self._data_list = np.array(data_list)[idx]
        self._genres = np.array(genres)[idx]

    def _make_windows(self, data_list, genres, shuffle=True, is_test=False):
        all_windows = []
        all_labels = []
        song_labels = []
        for ix, (datapoint, label) in tqdm(enumerate(zip(data_list, genres))):
            windows = self._window(datapoint,
                                   window_size=self.window_size,
                                   overleap=self.overleap)
            all_windows.append(windows)
            all_labels.extend([label] * len(windows))
            if is_test:
                song_labels.extend([ix] * len(windows))
        data = np.expand_dims(np.vstack(all_windows), axis=-1)
        labels = np.array([self.label2idx[label] for label in all_labels])
        self._song_labels = np.array(song_labels)
        if shuffle:
            idx = np.arange(0, len(data_list))
            np.random.shuffle(idx)
            data = data[idx]
            labels = labels[idx]
            if is_test:
                self._song_labels = self._song_labels[idx]
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
        vs = val_size/(1-test_size)
        print("Splitting: ", end="")
        train_val_X, test_X, train_val_y, test_y = train_test_split(self._data_list,
                                                                    self._genres, 
                                                                    test_size=0.1,
                                                                    stratify=self._genres)
        train_X, val_X, train_y, val_y = train_test_split(train_val_X, train_val_y, 
                                                          test_size=vs, stratify=train_val_y)

        print("Done")
        train_X, train_y = self._make_windows(train_X, train_y)
        train_X = self._scale(train_X, fit=True)
        val_X, val_y = self._make_windows(val_X, val_y)
        val_X = self._scale(val_X)
        test_X, test_y = self._make_windows(test_X, test_y, is_test=True)
        test_X = self._scale(test_X)

        return train_X, val_X, test_X, train_y, val_y, test_y
