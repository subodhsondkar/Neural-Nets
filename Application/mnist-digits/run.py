import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
print(x_train, y_train, x_test, y_test)
class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
x_train, x_test = x_train / 255.0, x_test / 255.0

model = keras.Sequential([
    keras.layers.Flatten(input_shape = (28, 28)),
    keras.layers.Dense(64, activation = tf.nn.relu),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(24, activation = tf.nn.relu),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(10, activation = tf.nn.softmax),
])

model.compile(optimizer = 'adam', 
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])

history = model.fit(x_train, y_train, batch_size = 512, epochs = int(input("No. of epochs: ")), validation_split = 0.1)
score = model.evaluate(x_test, y_test, batch_size = 512)
predictions = model.predict(x_test)

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

