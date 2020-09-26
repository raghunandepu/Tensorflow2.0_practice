from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
import numpy as np
import os


housing = fetch_california_housing()
X_train_full, X_test, y_train_full, y_test = train_test_split(housing.data, housing.target)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)


# Creating model
model = keras.models.Sequential([
    keras.layers.Dense(30, activation="relu", input_shape=X_train.shape[1:]),
    keras.layers.Dense(1)
])

model.compile(loss='mean_squared_error', optimizer="sgd")

history = model.fit(X_train, y_train, epochs=20, validation_data = (X_valid, y_valid))
mse_test = model.evaluate(X_test, y_test)

X_new = X_test[:3]
y_pred = model.predict(X_new)

print(y_pred)
print(y_test[:3])

# As you can see, the Sequential API is quite easy to use.
# However, although Sequential models are extremely common, it is sometimes useful to build neural networks with more complex topologies, or with multiple inputs or outputs.
# For this purpose, Keras offers the Functional API


# Saving and Restoring
"""
Keras will use the HDF5 format to save both the model’s architecture (including every layer’s hyperparameters) 
and the values of all the model parameters for every layer (e.g., connection weights and biases). It also saves the 
optimizer (including its hyperparameters and any state it may have)
"""
model.save("my_keras_model.h5")


"""
This will work when using the Sequential API or the Functional API, but unfortunately not when using model subclassing. 
You can use save_weights() and load_weights() to at least save and restore the model parameters, but you will need to 
save and restore everything else yourself."""
model = keras.models.load_model("my_keras_model.h5")

model.save_weights("my_keras_weights.ckpt")

model.load_weights("my_keras_weights.ckpt")

# Using Callbacks
"""
But what if training lasts several hours? This is quite common, especially when training on large datasets. 
In this case, you should not only save your model at the end of training, but also save checkpoints at regular intervals
during training, to avoid losing everything if your computer crashes. But how can you tell the fit() method to save 
checkpoints? Use callbacks."""


#Using Callbacks during Training
keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)
model = keras.models.Sequential([
    keras.layers.Dense(30, activation="relu", input_shape=[8]),
    keras.layers.Dense(30, activation="relu"),
    keras.layers.Dense(1)
])
model.compile(loss="mse", optimizer=keras.optimizers.SGD(lr=1e-3))

"""
Moreover, if you use a validation set during training, you can set save_best_only=True when creating the 
ModelCheckpoint. In this case, it will only save your model when its performance on the validation set is the best 
so far. This way, you do not need to worry about training for too long and overfitting the training set: 
simply restore the last model saved after training, and this will be the best model on the validation set. 
The following code is a simple way to implement early stopping"""

checkpoint_cb = keras.callbacks.ModelCheckpoint("my_keras_model.h5",
                                                save_best_only=True)

"""
Another way to implement early stopping is to simply use the EarlyStopping callback. It will interrupt training when it 
measures no progress on the validation set for a number of epochs (defined by the patience argument), and it will 
optionally roll back to the best model. You can combine both callbacks to save checkpoints of your model (in case your 
computer crashes) and interrupt training early when there is no more progress (to avoid wasting time and resources):"""


early_stopping_cb = keras.callbacks.EarlyStopping(patience=10,
                                                restore_best_weights=True)
history = model.fit(X_train,y_train,
                    epochs=50,
                    validation_data=(X_valid, y_valid),
                    callbacks=[checkpoint_cb, early_stopping_cb])

#model = keras.models.load_model("my_keras_model.h5") # rollback to best model
mse_test = model.evaluate(X_test, y_test)


# Custom callback
class PrintValTrainRatioCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        print("\nval/train: {:.2f}".format(logs["val_loss"] / logs["loss"]))

val_train_ratio_cb = PrintValTrainRatioCallback()
history = model.fit(X_train, y_train, epochs=1,
                    validation_data=(X_valid, y_valid),
                    callbacks=[val_train_ratio_cb])

# ===================================================================
# TensorBoard
root_logdir = os.path.join(os.curdir, "my_logs")

def get_run_logdir():
    import time
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)

run_logdir = get_run_logdir()
print(run_logdir)

keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)

model = keras.models.Sequential([
    keras.layers.Dense(30, activation="relu", input_shape=[8]),
    keras.layers.Dense(30, activation="relu"),
    keras.layers.Dense(1)
])
model.compile(loss="mse", optimizer=keras.optimizers.SGD(lr=1e-3))

tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)
history = model.fit(X_train, y_train, epochs=30,
                    validation_data=(X_valid, y_valid),
                    callbacks=[checkpoint_cb, tensorboard_cb])


# Second run to observe in tensorboard
run_logdir2 = get_run_logdir()
print(run_logdir2)

keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)

model = keras.models.Sequential([
    keras.layers.Dense(30, activation="relu", input_shape=[8]),
    keras.layers.Dense(30, activation="relu"),
    keras.layers.Dense(1)
])
model.compile(loss="mse", optimizer=keras.optimizers.SGD(lr=0.05))

tensorboard_cb = keras.callbacks.TensorBoard(run_logdir2)
history = model.fit(X_train, y_train, epochs=30,
                    validation_data=(X_valid, y_valid),
                    callbacks=[checkpoint_cb, tensorboard_cb])


# Hyperparameter Tuning
keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)

def build_model(n_hidden=1, n_neurons=30, learning_rate=3e-3, input_shape=[8]):
    model = keras.models.Sequential()
    model.add(keras.layers.InputLayer(input_shape=input_shape))
    for layer in range(n_hidden):
        model.add(keras.layers.Dense(n_neurons, activation="relu"))
    model.add(keras.layers.Dense(1))
    optimizer = keras.optimizers.SGD(lr=learning_rate)
    model.compile(loss="mse", optimizer=optimizer)
    return model

keras_reg = keras.wrappers.scikit_learn.KerasRegressor(build_model)

"""
The KerasRegressor object is a thin wrapper around the Keras model built using build_model(). Since we did not specify 
any hyperparameters when creating it, it will use the default hyperparameters we defined in build_model(). Now we can 
use this object like a regular Scikit-Learn regressor: we can train it using its fit() method, then evaluate it using 
its score() method, and use it to make predictions using its predict() method, as you can see in the following code:
"""

keras_reg.fit(X_train, y_train, epochs=100,
              validation_data=(X_valid, y_valid),
              callbacks=[keras.callbacks.EarlyStopping(patience=10)])
mse_test = keras_reg.score(X_test, y_test)
y_pred = keras_reg.predict(X_new)


from scipy.stats import reciprocal
from sklearn.model_selection import RandomizedSearchCV

param_distribs = {
    "n_hidden": [0, 1, 2, 3],
    "n_neurons": np.arange(1, 100),
    "learning_rate": reciprocal(3e-4, 3e-2),
}

rnd_search_cv = RandomizedSearchCV(keras_reg, param_distribs, n_iter=10, cv=3, verbose=2)
rnd_search_cv.fit(X_train, y_train, epochs=100,
                  validation_data=(X_valid, y_valid),
                  callbacks=[keras.callbacks.EarlyStopping(patience=10)])



print(rnd_search_cv.best_params_)

print(rnd_search_cv.best_score_)

model = rnd_search_cv.best_estimator_.model

model.evaluate(X_test, y_test)