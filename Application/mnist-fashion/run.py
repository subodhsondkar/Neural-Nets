'''
import tensorflow
from tensorflow.keras.layers import *

tensorflow.nn.relu(z)

inputs = Inputs(m)
hidden = Dense(d1)(inputs)
outputs = Dense(2)(hidden)

loss = tensorflow.reduce_mean(tensorflow.nn.softmax_cross_entropy_with_logits(model.y, model.pred))
loss = tensorflow.reduce_mean(tf.square(tf.subtract(model.y, model.pred)))

weights = tensorflow.random_normal(shape, stddev = sigma)
grads = tensorflow.gradients(ys = loss, xs = weights)
weights_new = weights.assign(weights - lr * grads)

tensorflow.train.AdamOptimizer

tensorflow.keras.layers.Dropout(p = 0.5)
'''
'''
import tensorflow as tf
from tf import keras
from keras.models import Sequential
from keras.layers import Dense
import numpy
import matplotlib.pyplot

print(tf.__version__)

model = Sequential()

model.add(Dense(units = 64, activation = 'relu', input_dim = 784))
model.add(Dense(units = 10, activation = 'softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = 'accuracy'])

model.fit(x_train, y_train, epochs = 2, batch_size = 32)

model.train_on_batch(x_batch, y_batch)

loss_and_metrics = model.evaluate(x_test, y_test, batch_size = 128)

classes = model.predict(x_test, batch_size = 128)

'''
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt

fashion_mnist = keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
'''
datagen = ImageDataGenerator(
    featurewise_center = True,
    featurewise_std_normalization = True,
    rotation_range = 20,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    zoom_range = [0.5, 1],
    horizontal_flip = True,
    vertical_flip = True)
datagen.fit(x_train)
'''
x_train = x_train / 255.0
x_test = x_test / 255.0

model = keras.Sequential([
    keras.layers.Flatten(input_shape = (28, 28)),
    keras.layers.Dense(128, activation = tf.nn.relu),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(10, activation = tf.nn.softmax),
])

model.compile(optimizer = 'adam', 
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])

nep = int(input("No. of epochs: "))
model.fit(x_train, y_train, batch_size = 128, epochs = nep)
score = model.evaluate(x_test, y_test, batch_size = 128)
predictions = model.predict(x_test, steps = 2)
model.save("model.h5")

# Early stopping: keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto', baseline=None, restore_best_weights=False)
# Learning rate scheduler: keras.callbacks.LearningRateScheduler(schedule, verbose=0)

#model.summary()

