from data_loader import DataLoader
from model import build_model
from tensorflow.keras.callbacks import (EarlyStopping, ReduceLROnPlateau,
                                        ModelCheckpoint)

model_name = "first_try.h5"
n_epochs = 25
batch_size = 512

data_loader = DataLoader(window_size=1024, overleap=256)
train_X, val_X, test_X, train_y, val_y, test_y = data_loader.get_data()

model = build_model(n_hidden_conv=5)

es = EarlyStopping(monitor='val_loss', mode='min', patience=10,
                   restore_best_weights=False)
lr_cb = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1,
                          min_lr=0.0001)
mc = ModelCheckpoint(filepath="best_model.h5", save_best_only=True)
callbacks = [es, lr_cb, mc]

history = model.fit(train_X, train_y, epochs=n_epochs,
                    validation_data=(val_X, val_y),
                    callbacks=callbacks, verbose=1,
                    batch_size=batch_size)
model.save(model_name)
