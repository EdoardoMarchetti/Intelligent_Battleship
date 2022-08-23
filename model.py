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

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten()) #Input Layer 
model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu)) #Primo strato nascosto
model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu)) #Secondo strato nascosto
model.add(tf.keras.layers.Dense(10, activation = tf.nn.softmax)) #Output layer (activation = sigmoid poichè in ouput si ragiona su distribuzione di porbabilità)

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




#model.save('model_for_exam')



