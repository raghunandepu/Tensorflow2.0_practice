"""
You may have multiple independent tasks based on the same data.
Sure, you could train one neural network per task, but in many cases you will get better results on all tasks by
training a single neural network with one output per task. This is because the neural network can learn features in the
data that are useful across tasks. For example, you could perform multitask classification on pictures of faces, using
one output to classify the person’s facial expression (smiling, surprised, etc.) and another output to identify whether
they are wearing glasses or not.
"""

"""
Another use case is as a regularization technique (i.e., a training constraint whose objective is to reduce 
overfitting and thus improve the model’s ability to generalize). For example, you may want to add some auxiliary outputs 
in a neural network architecture (see Figure 10-16) to ensure that the underlying part of the network learns something 
useful on its own, without relying on the rest of the network."""

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras


housing = fetch_california_housing()
X_train_full, X_test, y_train_full, y_test = train_test_split(housing.data, housing.target)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)

input_A = keras.layers.Input(shape=[5], name="wide_input")
input_B = keras.layers.Input(shape=[6], name="deep_input")

hidden1 = keras.layers.Dense(30, activation="relu")(input_B)
hidden2 = keras.layers.Dense(30, activation="relu")(hidden1)
concat = keras.layers.Concatenate()([input_A, hidden2])
output = keras.layers.Dense(1, name="output")(concat)
aux_output = keras.layers.Dense(1, name="aux_output")(hidden2) # adding new aux output layer

model = keras.Model(inputs=[input_A, input_B], outputs=[output, aux_output])

print(model.summary())

model.compile(loss=["mse", "mse"],   # Each output will need its own loss function
              loss_weights=[0.9, 0.1], # We care much more about the main output than about the auxiliary output
              optimizer=keras.optimizers.SGD(lr=1e-3))


X_train_A, X_train_B = X_train[:, :5], X_train[:, 2:]
X_valid_A, X_valid_B = X_valid[:, :5], X_valid[:, 2:]
X_test_A, X_test_B = X_test[:, :5], X_test[:, 2:]
X_new_A, X_new_B = X_test_A[:3], X_test_B[:3]

history = model.fit([X_train_A, X_train_B],
                    [y_train, y_train],
                    epochs=20,
                    validation_data=([X_valid_A, X_valid_B],
                                     [y_valid, y_valid]))

# When we evaluate the model, Keras will return the total loss, as well as all the individual losses:
total_loss, main_loss, aux_loss = model.evaluate([X_test_A, X_test_B], [y_test, y_test])

y_pred_main, y_pred_aux = model.predict([X_new_A, X_new_B])

print(y_pred_main[:3])
print(y_test[:3])


