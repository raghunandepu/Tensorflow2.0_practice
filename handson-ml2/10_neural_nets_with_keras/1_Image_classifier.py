import tensorflow as tf
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

print(tf.__version__)
print(keras.__version__)

# Loading dataset using Keras
fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

print(X_train_full.shape)
print(X_train_full.dtype)

# Creating validation set and scaling pixel intensities in the range of 0-1
# by dividing with 255.0
X_valid, X_train = X_train_full[:5000]/255.0, X_train_full[5000:]/255.0
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
X_test = X_test/255.0

class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# Creating the model using the Sequential API
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(300, activation='relu'),
    keras.layers.Dense(100, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])

#print(model.summary())

print(model.layers)

hidden1 = model.layers[1]
weights, biases = hidden1.get_weights()
print(weights.shape)
print(biases.shape)

# Compiling the model
model.compile(loss="sparse_categorical_crossentropy",
              optimizer="sgd",
              metrics=["accuracy"])

# Training the model
history = model.fit(X_train, y_train, epochs=10,
                    validation_data=(X_valid, y_valid))


pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1) # set the vertical range to [0-1]
plt.show()


print(model.evaluate(X_test, y_test))

# Using the model to make predictions (to get probabilities for each class)
X_new = X_test[:3]
y_proba = model.predict(X_new)
print(y_proba.round(2))

# If you care about the class with highest estimated prob
y_pred = model.predict_classes(X_new)
print(y_pred)
print(np.array(class_names)[y_pred])

#Here, the classifier actually classified all three images correctly
y_new = y_test[:3]
print(y_new)


