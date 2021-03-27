import numpy as np
from scipy.io import wavfile
from glob import glob
import os
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
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

        all_windows = []
        all_labels = []
        for datapoint, label in tqdm(zip(data_list, genres)):
            windows = self._window(datapoint,
                                   window_size=self.window_size,
                                   overleap=self.overleap)
            all_windows.append(windows)
            all_labels.extend([label] * len(windows))

        self.data = np.vstack(all_windows)

        self.label2idx = {l: idx for idx, l in enumerate(np.unique(genres))}
        self.labels = np.array([self.label2idx[label] for label in all_labels])
        idx = np.random.randint(0, self.data.shape[0], self.data.shape[0])
        self.data = self.data[idx]
        self.labels = self.labels[idx]

    def _train_val_test_scale(self, X, y, val_size=0.1, test_size=0.1,
                              scale=True):
        vs = 1 - (test_size + val_size)
        ts = 1 - test_size
        print("Splitting: ", end="")
        train_X, val_X, test_X = np.split(X, [int(vs*len(X)), int(ts*len(X))])
        train_y, val_y, test_y = np.split(y, [int(vs*len(y)), int(ts*len(y))])
        print("Done")
        if scale:
            print("Scaling: ", end="")
            mean = np.mean(train_X)
            std = np.std(train_X)
            print("mean and std computing done, scaling arrs: ", end="")
            train_X = (train_X - mean) / std
            val_X = (val_X - mean) / std
            test_X = (test_X - mean) / std
            print("Done")
        return (train_X, val_X, test_X, train_y, val_y, test_y)

    def _window(self, a, window_size=512, overleap=256):
        shape = (a.size - window_size + 1, window_size)
        strides = a.strides * 2
        view = np.lib.stride_tricks.as_strided(a, strides=strides,
                                               shape=shape)[0::overleap]
        return view

    def get_data(self, val_size=0.1, test_size=0.1, scaler=StandardScaler):
        if self.data is None:
            self._init_data()
        X = np.expand_dims(self.data, axis=-1)
        onehot_y = to_categorical(self.labels)
        return self._train_val_test_scale(X, onehot_y)
