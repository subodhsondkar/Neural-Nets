import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models

fashion_mnist = keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

model = tf.keras.models.load_weights("model.h5")
prediction = model.predict()

