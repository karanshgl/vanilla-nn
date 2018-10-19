import numpy as np
from model import Model
from dense_layer import Dense
from dropout import Dropout
from convnet import Conv2D
from maxpool import Maxpool
from flatten import Flatten
from batchnorm import Batchnorm
from adam import Adam
import pandas as pd
import utils

#from keras.models import Sequential
#import keras.layers as L
#from keras.optimizers import SGD
def getModel(X, Y, epochs):
	model = Model(learning_rate = 0.01, batch_size = 64, epochs = 200, optimizer = Adam())
	model.add(Dropout(0.5))
	model.add(Dense(1024, 512, activation = 'sigmoid'))
	model.add(Dropout(0.5))
	model.add(Dense(512, 64, activation = 'sigmoid'))
	model.add(Dropout(0.5))
	model.add(Dense(64, 1))
	print("starting to train")
	model.train(X,Y)

	# model.add(Conv2D(4,(3,3), activation = 'sigmoid'))
	# model.add(Maxpool((3,3), stride = 2)) # 16x16
	# model.add(Conv2D(1,(3,3), activation = 'sigmoid'))
	# model.add(Maxpool((3,3), stride = 2)) # 8x8
	# # model.add(Dropout(0.5))
	# model.add(Flatten())
	# model.add(Dense(64,1))
	# model.train(X,Y)


	# model.add(Dense(20,10, activation = 'relu'))
	# model.add(Dropout(0.5))
	# model.add(Dense(10,1))
	# model.train(X,Y)
		

	#model.train(X, Y, lr, epochs)
	return model

def get_minibatch(X,Y, batch_size):
	random_indices = np.random.choice(X.shape[0], batch_size)
	return X[random_indices], Y[random_indices]

def prepareData(X, Y):
	for i in range(X.shape[1]):
		X[:, i] = (X[:, i]-X[:, i].mean())/X[:, i].std()
	# Y = Y-Y.min()
	# Y = Y/Y.max()
	# X -= X.min()
	# X /= X.max()

	return X, Y


def getKerasModel(X, Y):
	model = Sequential()
	model.add(L.Dense(512, input_shape=(1024, ), activation='sigmoid'))
	model.add(L.Dense(64, activation='sigmoid'))
	model.add(L.Dense(1))
	sgd = SGD(lr=0.005)
	model.compile(optimizer=sgd, loss='mse', metrics=['mse'])
	model.fit(X, Y, batch_size=500, epochs=1000)
	return model


# data = pd.read_csv('datanew.csv')
# data = np.array(data.values)
# data_x = data[:, :-1]
# data_y = data[:, -1]
# data_x, data_y = prepareData(data_x, data_y)
# X = data_x
# Y = data_y[:, np.newaxis]
# print(X.shape, Y.shape)
X, Y = utils.get_data('steering', 0)
# print(X.shape)
X_test, Y_test = X[1500:], Y[1500:]
X, Y = prepareData(X[:1500],Y[:1500])
model = getModel(X, Y, 10000)
X, Y = prepareData(X_test, Y_test)
print("TEST")
model.test(X,Y)
"""
images, angles = loadImages('steering')
images, angles = prepareData(images, angles)
print("aquired images")
#model = getKerasModel(images, angles)
model = getModel(images, angles, 0.000005, 0.4, 500, 1000)
"""