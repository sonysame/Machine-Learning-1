import numpy as np
import matplotlib.pyplot as plt
import time
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Embedding
import os

def show_prediction():
	n_show=96
	y=model.predict(x_test)
	plt.figure(2,figsize=(12,8))
	plt.gray()
	for i in range(n_show):
		plt.subplot(8,12,i+1)
		x=x_test[i,:]
		x=x.reshape(28,28)
		plt.pcolor(1-x)
		wk=y[i,:]
		prediction=np.argmax(wk)
		plt.text(22,25.5, "%d"%prediction, fontsize=12)
		if prediction!=np.argmax(y_test[i,:]):
			plt.plot([0,27],[1,1], color='cornflowerblue', linewidth=5)
		plt.xlim(0,27)
		plt.ylim(27,0)
		plt.xticks([], "")
		plt.yticks([], "")

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
outfile=np.load('mnist_data.npz')

x_train=outfile['x_train']
y_train=outfile['y_train']
x_test=outfile['x_test']
y_test=outfile['y_test']

x_train=x_train.astype('float32')
x_train=x_train/255
num_classes=10
y_train=tf.keras.utils.to_categorical(y_train, num_classes)

x_test=x_test.astype('float32')
x_test=x_test/255
y_test=tf.keras.utils.to_categorical(y_test, num_classes)

np.random.seed(1)
model=Sequential()
model.add(Dense(16, input_dim=784, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])

startTime=time.time()
history=model.fit(x_train, y_train, epochs=10, batch_size=1000, verbose=1, validation_data=(x_test, y_test))
score=model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
print("Computation time:{0:.3f} sec".format(time.time()-startTime))

plt.figure(1,figsize=(10,4))
plt.subplots_adjust(wspace=0.5)

plt.subplot(1,2,1)
plt.plot(history.history['loss'], color='black', label='training')
plt.plot(history.history['val_loss'],color='cornflowerblue',label='test')
plt.ylim(0,10)
plt.legend()
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')


plt.subplot(1,2,2)
plt.plot(history.history['accuracy'],color='black',label='training')
plt.plot(history.history['val_accuracy'], color='cornflowerblue',label='test')
plt.ylim(0,1)
plt.legend()
plt.grid()
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show()

show_prediction()
plt.show()

w=model.layers[0].get_weights()[0]
plt.figure(3,figsize=(12,3))
plt.gray()
plt.subplots_adjust(wspace=0.35, hspace=0.5)
for i in range(16):
	plt.subplot(2,8,i+1)
	w1=w[:,i]
	w1=w1.reshape(28,28)
	plt.pcolor(-w1)
	plt.xlim(0,27)
	plt.ylim(27,0)
	plt.xticks([],"")
	plt.yticks([],"")
	plt.title("%d"%i)
plt.show()