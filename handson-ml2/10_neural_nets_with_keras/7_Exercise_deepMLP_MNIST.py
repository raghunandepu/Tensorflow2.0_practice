"""
Exercise: Train a deep MLP on the MNIST dataset (you can load it using keras.datasets.mnist.load_data().
See if you can get over 98% precision. Try searching for the optimal learning rate by using the approach presented in
this chapter (i.e., by growing the learning rate exponentially, plotting the loss, and finding the point where the loss
shoots up). Try adding all the bells and whistles—save checkpoints, use early stopping, and plot learning curves using
TensorBoard.
"""

from tensorflow import keras
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os


(X_train_full, y_train_full), (X_test, y_test) = keras.datasets.mnist.load_data()

print(X_train_full.shape)
print(y_train_full.shape)
print(X_test.shape)
print(y_test.shape)

# Splitting the dataset further for validation set
X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] / 255.0
y_valid, y_train = y_train_full[:5000] , y_train_full[5000:]

"""plt.imshow(X_train[0], cmap='binary')
plt.axis('off')
plt.show()"""

print(y_train)

"""
Let's build a simple dense network and find the optimal learning rate. 
We will need a callback to grow the learning rate at each iteration. It will also record the learning rate and the loss 
at each iteration:
"""
K = keras.backend

class ExponentialLearningRate(keras.callbacks.Callback):
    def __init__(self, factor):
        self.factor = factor
        self.rates = []
        self.losses = []
    def on_batch_end(self, batch, logs):
        self.rates.append(K.get_value(self.model.optimizer.lr))
        self.losses.append(logs["loss"])
        K.set_value(self.model.optimizer.lr, self.model.optimizer.lr * self.factor)


model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28,28]),
    keras.layers.Dense(300, activation="relu"),
    keras.layers.Dense(100, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])

print(model.summary())

model.compile(loss="sparse_categorical_crossentropy",
              optimizer=keras.optimizers.SGD(lr=1e-3),
              metrics=['accuracy'])
expon_lr = ExponentialLearningRate(factor=1.005)


history = model.fit(X_train, y_train, validation_data=(X_valid, y_valid),
                    callbacks=[expon_lr])

run_index = 1 # increment this at every run
run_logdir = os.path.join(os.curdir, "my_mnist_logs",
                          "run_{:03d}".format(run_index))
print(run_logdir)

early_stopping_cb = keras.callbacks.EarlyStopping(patience=20)
checkpoint_cb = keras.callbacks.ModelCheckpoint("my_mnist_model.h5", save_best_only=True)
tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)

history = model.fit(X_train, y_train, epochs=100,
                    validation_data=(X_valid, y_valid),
                    callbacks=[early_stopping_cb, checkpoint_cb, tensorboard_cb])

model = keras.models.load_model("my_mnist_model.h5") # rollback to best model
model.evaluate(X_test, y_test)

