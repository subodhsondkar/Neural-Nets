import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.datasets import boston_housing
(train_data, train_label), (test_data, test_label) = boston_housing.load_data()
train_data, test_data = train_data / 1000, test_data / 1000

model = keras.Sequential([
    keras.layers.Dense(13),
    keras.layers.Dense(8, activation = tf.nn.relu),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(4, activation = tf.nn.relu),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1, activation = tf.nn.softmax),
])

model.compile(optimizer = 'adam',
        loss = 'sparse_categorical_crossentropy',
        metrics = ['accuracy'])

history = model.fit(train_data, train_label, batch_size = 4, epochs = int(input("Epochs: ")), validation_split = 0.1)
score = model.evaluate(test_data, test_label, batch_size = 4)
predictions = model.predict(test_data)

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

