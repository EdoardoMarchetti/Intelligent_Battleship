# libraries needed
import tensorflow as tf

import matplotlib.pyplot as plt
import numpy as np
import random
from tqdm import tqdm
from tensorflow.keras import layers, models

mnist = tf.keras.datasets.mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()



def build_cnn_model3_improved():

  cnn_model3 = models.Sequential()

  # Feature Learning block:
  cnn_model3.add(layers.Conv2D(30, (3,3), activation='relu', input_shape=(28,28,1)))
  cnn_model3.add(layers.MaxPooling2D((2,2)))
  cnn_model3.add(layers.Conv2D(15, (3,3), activation='relu'))
  cnn_model3.add(layers.MaxPooling2D((2,2)))

  #Dropuot layer
  cnn_model3.add(layers.Dropout(rate=0.2))

  # Classification block:
  cnn_model3.add(layers.Flatten())
  cnn_model3.add(layers.Dense(128, activation='relu'))
  cnn_model3.add(layers.Dense(10, activation='softmax'))

  return cnn_model3

cnn_model3 = build_cnn_model3_improved()


# Print the summary of the model
#print(cnn_model3.summary())


cnn_model3.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)


hist = cnn_model3.fit(
    train_images,
    train_labels,
    validation_data=(test_images, test_labels),
    batch_size = 64,
    epochs = 10,
)

plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['Train', 'Val'], loc = 'upper left')
plt.show()

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Loss Function')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['Train', 'Val'], loc = 'upper left')
plt.show()

#cnn_model3.save('cnn_model_3')




