from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, Flatten
from keras.datasets import mnist
from keras.utils import to_categorical


import matplotlib.pyplot as plt
import numpy as np



#Load and split data
(X_train, y_train) , (X_test, y_test) = mnist.load_data()


#Reshape data
X_train = X_train.reshape(60000,28,28,1) #1 sta per grayscale
X_test = X_test.reshape(10000,28,28,1)

#One hot encoding
y_train_one_hot = to_categorical(y_train)
y_test_one_hot = to_categorical(y_test)



#Build the CNN model
model = Sequential()
#add model layers
model.add(Conv2D(64, kernel_size= 3, activation='relu', input_shape= (28,28,1))) #input size poichè primo strato
model.add(Conv2D(32, kernel_size=3, activation='relu'))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))


#Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#Train the model
hist = model.fit(X_train, y_train_one_hot,
                    validation_data=(X_test, y_test_one_hot), 
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


# model.save('cnn_model')




