import numpy as np
import matplotlib.pyplot as plt
import time
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Embedding
import os

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
"""
train_data_file=open("./input/mnist_train.csv",'r')
train_list=train_data_file.readlines()
train_data_file.close()

test_data_file=open("./input/mnist_test.csv",'r')
test_list=test_data_file.readlines()
test_data_file.close()

x_train=np.zeros((60000,784))
y_train=np.zeros((60000))

x_test=np.zeros((10000,784))
y_test=np.zeros((10000))

for i in range(len(train_list)):
	x_train[i]=train_list[i].strip().split(",")[1:]
	y_train[i]=train_list[i].strip().split(",")[0]
	
for i in range(len(test_list)):
	x_test[i]=test_list[i].strip().split(",")[1:]
	y_test[i]=test_list[i].strip().split(",")[0]

np.savez('mnist_data.npz', x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
"""
outfile=np.load('mnist_data.npz')

x_train=outfile['x_train']
y_train=outfile['y_train']
x_test=outfile['x_test']
y_test=outfile['y_test']

x_train=x_train.reshape(60000,28,28)

plt.figure(1,figsize=(12,3.2))
plt.subplots_adjust(wspace=0.5)
plt.gray()
for id in range(3):
	plt.subplot(1,3,id+1)
	img=x_train[id,:,:]
	plt.pcolor(255-img)
	plt.text(24.5, 26, "%d"%y_train[id], color='cornflowerblue', fontsize=18)
	plt.xlim(0,27)
	plt.ylim(27,0)
	plt.grid('on', color='white')
plt.show()
