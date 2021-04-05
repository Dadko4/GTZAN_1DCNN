import wandb
from data_loader import DataLoader
from model import build_model, f1_m, init_env
from tensorflow import keras
from tensorflow.keras.callbacks import (EarlyStopping, ReduceLROnPlateau)
import numpy as np
np.random.seed(0)


window_size = 2**14
overleap = 2**11

data_loader = DataLoader(window_size=window_size, overleap=overleap)

n_epochs = 90
batch_size = 64
lr = 0.001
optimizer = keras.optimizers.Adam(lr)
loss = keras.losses.categorical_crossentropy
n_hidden_conv = 5
n_filters = 64
kernel_size = 3

# wandb.init(project='nn2_cross_val',
#            config={"optimizer": str(optimizer).split()[0].split('.')[-1],
#                    "loss": loss.__name__.split('.')[-1],
#                    "batch_size": batch_size, "lr": lr,
#                    "n_hidden_conv": n_hidden_conv,
#                    "kernel_size": kernel_size, "n_filters": n_filters,
#                    "window_size": window_size, "overleap": overleap})
cv_acc = []
cv_loss = []
cv_f1 = []
best_model = None
best_acc = 0.
for ix, (train_X, val_X, train_y, val_y) in enumerate(data_loader.get_folds()):
    # API key = 8d9fd59df428fadd8926c268bf32588eb4c560f6
    model = build_model(n_hidden_conv=n_hidden_conv, kernel_size=kernel_size,
                        timesteps=window_size, loss=loss,
                        optimizer=optimizer, filters=n_filters,
                        metrics=['accuracy', f1_m])
    es = EarlyStopping(monitor='val_loss', mode='max', patience=15,
                       restore_best_weights=True)
    lr_cb = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5,
                              verbose=1, min_lr=0.0001)
    callbacks = [es, lr_cb]

    history = model.fit(train_X, train_y, epochs=n_epochs,
                        validation_data=(val_X, val_y),
                        callbacks=callbacks, verbose=1,
                        batch_size=batch_size)

    val_loss, val_acc, val_f1 = model.evaluate(val_X, val_y)
    if val_acc > best_acc:
        best_acc = val_acc
        model.save("best_model.h5")
    cv_acc.append(val_acc)
    cv_loss.append(val_loss)
    cv_f1.append(val_f1)

wandb.config.update({"mean_cv_acc": np.mean(cv_acc),
                     "mean_cv_loss": np.mean(cv_loss),
                     "mean_cv_f1": np.mean(cv_f1),
                     "cv_acc": cv_acc,
                     "cv_loss": cv_loss,
                     "cv_f1": cv_f1})

init_env()
model = keras.models.load_model("best_model.h5", custom_objects={'f1_m': f1_m})

test_X, test_y = data_loader.get_test_data(0.1)
np.savez("./test_data", **{"X": test_X, "y": test_y})
test_loss, test_acc, test_f1 = model.evaluate(test_X, test_y)

wandb.config.update({"test_loss": test_loss,
                     "test_acc": test_acc,
                     "test_f1": test_f1})
# wandb.finish()
