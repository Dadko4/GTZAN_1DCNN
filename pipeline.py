import wandb
from wandb.keras import WandbCallback
from data_loader import DataLoader
from model import build_model, f1_m
from tensorflow import keras
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint,
                                        ReduceLROnPlateau)
from datetime import datetime
import numpy as np
np.random.seed(0)


# model_name = "first_try.h5"
window_size = 2**14
overleap = 2**11

data_loader = DataLoader(window_size=window_size, overleap=overleap)
train_X, val_X, test_X, train_y, val_y, test_y = data_loader.get_data()

np.savez("./test_data", **{"X": test_X, "y": test_y})

n_epochs = 90
batch_size = 64
lr = 0.001
optimizer = keras.optimizers.Adam(lr)
loss = keras.losses.categorical_crossentropy
n_hidden_conv = 5
n_filters = 64
kernel_size = 3

model = build_model(n_hidden_conv=n_hidden_conv, kernel_size=kernel_size,
                    timesteps=train_X.shape[1], loss=loss,
                    optimizer=optimizer, filters=n_filters,
                    metrics=['accuracy', f1_m])

# API key = 8d9fd59df428fadd8926c268bf32588eb4c560f6
wandb.init(project='nn2_valid_split',
           config={"optimizer": str(optimizer).split()[0].split('.')[-1],
                   "loss": loss.__name__.split('.')[-1],
                   "batch_size": batch_size, "lr": lr,
                   "n_hidden_conv": n_hidden_conv,
                   "kernel_size": kernel_size, "n_filters": n_filters,
                   "window_size": window_size, "overleap": overleap})

es = EarlyStopping(monitor='val_accuracy', mode='max', patience=15,
                   restore_best_weights=True)
lr_cb = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=5,
                          verbose=1, min_lr=0.0001)
mc = ModelCheckpoint(filepath=f"best_model_{datetime.now()}.h5",
                     save_best_only=True)
callbacks = [es, lr_cb, mc, WandbCallback()]

history = model.fit(train_X, train_y, epochs=n_epochs,
                    validation_data=(val_X, val_y),
                    callbacks=callbacks, verbose=1,
                    batch_size=batch_size)

test_loss, test_acc, test_f1 = model.evaluate(test_X, test_y)

wandb.config.update({"test_loss": test_loss,
                     "test_acc": test_acc,
                     "test_f1": test_f1})
wandb.finish()

np.savez("./pred_data", **{"y_pred": model.predict(test_y)})

