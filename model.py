from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv1D, Dense, Flatten, Dropout,
                                     BatchNormalization)


def build_model(n_hidden_conv=1, timesteps=1024, num_classes=10,
                loss='categorical_crossentropy', optimizer='adam', filters=64,
                metrics=['accuracy']):
    model = Sequential()
    model.add(Conv1D(filters=filters, kernel_size=3, activation='relu',
                     input_shape=(timesteps, 1)))
    for _ in range(n_hidden_conv):
        model.add(Dropout(0.2))
        model.add(Conv1D(filters=filters, kernel_size=3, activation='relu'))
        model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=optimizer,
                  metrics=metrics)
    return model
