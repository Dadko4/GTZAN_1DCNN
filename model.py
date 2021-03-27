from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv1D, Dense, Flatten, Dropout,
                                     BatchNormalization)
from tensorflow.keras import backend as K


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


def build_model(n_hidden_conv=1, timesteps=1024, num_classes=10,
                loss='categorical_crossentropy', optimizer='adam',
                filters=64, metrics=['accuracy']):
    model = Sequential()
    model.add(Conv1D(filters=filters, kernel_size=3, activation='relu',
                     input_shape=(timesteps, 1)))
    for _ in range(n_hidden_conv):
        model.add(Dropout(0.2))
        model.add(Conv1D(filters=filters, kernel_size=3, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=optimizer,
                  metrics=metrics)
    return model
