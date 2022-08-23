# libraries needed
import tensorflow as tf
 
import matplotlib.pyplot as plt
import numpy as np
import random
from tqdm import tqdm
 
 
 
mnist = tf.keras.datasets.mnist
 
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

 
#Normalize
train_images = tf.keras.utils.normalize(train_images, axis = 1)
test_images = tf.keras.utils.normalize(test_images, axis = 1)

def build_fc_model():
  fc_model = tf.keras.Sequential([
                                  tf.keras.layers.Flatten(),#appiattisco l'arrey
                                  tf.keras.layers.Dense(128, activation="relu"),
                                  tf.keras.layers.Dense(128, activation="relu"),
                                  tf.keras.layers.Dense(10,activation="softmax"),
  ])
  return fc_model
model = build_fc_model()

model.compile(optimizer= 'adam',
              loss='sparse_categorical_crossentropy' ,
              metrics= ['accuracy'])

hist = model.fit(train_images, train_labels, 
            validation_data=(test_images, test_labels),
            epochs=3)

#Visualize the models accuracy
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




model.save('model_for_exam_2')




