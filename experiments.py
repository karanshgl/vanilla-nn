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
import copy


def experiment_one(X, Y, val_X, val_Y):
	"""
	A plot of sum of squares error on the training and validation set as a function of
	training iterations (for 5000 epochs) with a learning rate of 0.01. (no dropout,
	minibatch size of 64)
	"""
	model = Model(learning_rate = 0.01, batch_size = 64, epochs = 5000, optimizer = None)
	model.add(Dense(1024, 512, activation = 'sigmoid'))
	model.add(Dense(512, 64, activation = 'sigmoid'))
	model.add(Dense(64,1))
	print("Begin Training")
	model.train(X,Y, val_X, val_Y)
	model.save_history("experiments/experiment_one.csv")
	print("The CSV file is saved in the experiments folder. You can plot the graph using plot.py")

def experiment_two_32(X, Y, val_X, val_Y):
	"""
	A plot of sum of squares error on the training and validation set as a function of
	training iterations (for 1000 epochs) with a fixed learning rate of 0.01 for three
	minibatch sizes – 32, 64, 128.
	"""
	print("Batch Size: 32")
	model = Model(learning_rate = 0.01, batch_size = 32, epochs = 1000, optimizer = None)
	model.add(Dense(1024, 512, activation = 'sigmoid'))
	model.add(Dense(512, 64, activation = 'sigmoid'))
	model.add(Dense(64,1))
	print("Begin Training")
	model.train(X,Y, val_X, val_Y)
	model.save_history("experiments/experiment_two_32.csv")
	print("The CSV file is saved in the experiments folder. You can plot the graph using plot.py")

def experiment_two_64(X, Y, val_X, val_Y):
	"""
	A plot of sum of squares error on the training and validation set as a function of
	training iterations (for 1000 epochs) with a fixed learning rate of 0.01 for three
	minibatch sizes – 32, 64, 128.
	"""
	print("Batch Size: 64")
	model = Model(learning_rate = 0.01, batch_size = 64, epochs = 1000, optimizer = None)
	model.add(Dense(1024, 512, activation = 'sigmoid'))
	model.add(Dense(512, 64, activation = 'sigmoid'))
	model.add(Dense(64,1))
	print("Begin Training")
	model.train(X,Y, val_X, val_Y)
	model.save_history("experiments/experiment_two_64.csv")
	print("The CSV file is saved in the experiments folder. You can plot the graph using plot.py")

def experiment_two_128(X, Y, val_X, val_Y):
	"""
	A plot of sum of squares error on the training and validation set as a function of
	training iterations (for 1000 epochs) with a fixed learning rate of 0.01 for three
	minibatch sizes – 32, 64, 128.
	"""
	print("Batch Size: 128")
	model = Model(learning_rate = 0.01, batch_size = 128, epochs = 1000, optimizer = None)
	model.add(Dense(1024, 512, activation = 'sigmoid'))
	model.add(Dense(512, 64, activation = 'sigmoid'))
	model.add(Dense(64,1))
	print("Begin Training")
	model.train(X,Y, val_X, val_Y)
	model.save_history("experiments/experiment_two_128.csv")
	print("The CSV file is saved in the experiments folder. You can plot the graph using plot.py")


def experiment_three(X, Y, val_X, val_Y):
	"""
	A plot of sum of squares error on the training and validation set as a function of
	training iterations (for 1000 epochs) with a learning rate of 0.001, and dropout
	probability of 0.5 for the first, second and third layers.
	"""
	model = Model(learning_rate = 0.001, batch_size = 32, epochs = 1000, optimizer = None)
	model.add(Dropout(0.5))
	model.add(Dense(1024, 512, activation = 'sigmoid'))
	model.add(Dropout(0.5))
	model.add(Dense(512, 64, activation = 'sigmoid'))
	model.add(Dropout(0.5))
	model.add(Dense(64,1))
	print("Begin Training")
	model.train(X,Y, val_X, val_Y)
	model.save_history("experiments/experiment_three.csv")
	print("The CSV file is saved in the experiments folder. You can plot the graph using plot.py")

def experiment_four_05(X, Y, val_X, val_Y):
	"""
	A plot of sum of squares error on the training and validation set as a function of
	training iterations (for 1000 epochs) with different learning rates – 0.05, 0.001,
	0.005 (no drop out, minibatch size – 64)
	"""
	print("The following function is for learning rate 0.05 which was found to be very high. Hence does not converge.")
	print("To make it converge, kindly divide the error by the batch size in the model.py file.")
	model = Model(learning_rate = 0.05, batch_size = 64, epochs = 1000, optimizer = None)
	model.add(Dense(1024, 512, activation = 'sigmoid'))
	model.add(Dense(512, 64, activation = 'sigmoid'))
	model.add(Dense(64,1))
	print("Begin Training")
	model.train(X,Y, val_X, val_Y)
	model.save_history("experiments/experiment_four_05.csv")
	print("The CSV file is saved in the experiments folder. You can plot the graph using plot.py")


def experiment_four_001(X, Y, val_X, val_Y):
	"""
	A plot of sum of squares error on the training and validation set as a function of
	training iterations (for 1000 epochs) with different learning rates – 0.05, 0.001,
	0.005 (no drop out, minibatch size – 64)
	"""
	print("Learning Rate: 0.001")
	model = Model(learning_rate = 0.001, batch_size = 64, epochs = 1000, optimizer = None)
	model.add(Dense(1024, 512, activation = 'sigmoid'))
	model.add(Dense(512, 64, activation = 'sigmoid'))
	model.add(Dense(64,1))
	print("Begin Training")
	model.train(X,Y, val_X, val_Y)
	model.save_history("experiments/experiment_four_001.csv")
	print("The CSV file is saved in the experiments folder. You can plot the graph using plot.py")

def experiment_four_005(X, Y, val_X, val_Y):
	"""
	A plot of sum of squares error on the training and validation set as a function of
	training iterations (for 1000 epochs) with different learning rates – 0.05, 0.001,
	0.005 (no drop out, minibatch size – 64)
	"""
	print("Learning Rate: 0.005")
	model = Model(learning_rate = 0.005, batch_size = 64, epochs = 1000, optimizer = None)
	model.add(Dense(1024, 512, activation = 'sigmoid'))
	model.add(Dense(512, 64, activation = 'sigmoid'))
	model.add(Dense(64,1))
	print("Begin Training")
	model.train(X,Y, val_X, val_Y)
	model.save_history("experiments/experiment_four_005.csv")
	print("The CSV file is saved in the experiments folder. You can plot the graph using plot.py")


def ann_adam_layers(X, Y, val_X, val_Y):
	"""
	Extra Layers with sigmoid and Adam
	"""
	model = Model(learning_rate = 0.001, batch_size = 32, epochs = 200, optimizer = Adam())
	model.add(Dropout(0.5))
	model.add(Dense(1024, 512, activation = 'sigmoid'))
	model.add(Dropout(0.5))
	model.add(Dense(512, 256, activation = 'sigmoid'))
	model.add(Dropout(0.5))
	model.add(Dense(256,32, activation = 'sigmoid'))
	model.add(Dropout(0.5))
	model.add(Dense(32,1))
	print("Begin Training")
	model.train(X,Y, val_X, val_Y)
	model.save_history("experiments/ann-adam-1024-512-256-32-1.csv")
	print("The CSV file is saved in the experiments folder. You can plot the graph using plot.py")

def ann_layers(X, Y, val_X, val_Y):
	"""
	Extra Layers with sigmoid
	"""
	model = Model(learning_rate = 0.001, batch_size = 32, epochs = 1000, optimizer = None)
	model.add(Dropout(0.5))
	model.add(Dense(1024, 512, activation = 'sigmoid'))
	model.add(Dropout(0.5))
	model.add(Dense(512, 256, activation = 'sigmoid'))
	model.add(Dropout(0.5))
	model.add(Dense(256,32, activation = 'sigmoid'))
	model.add(Dropout(0.5))
	model.add(Dense(32,1))
	print("Begin Training")
	model.train(X,Y, val_X, val_Y)
	model.save_history("experiments/ann-1024-512-256-32-1.csv")
	print("The CSV file is saved in the experiments folder. You can plot the graph using plot.py")

def cnn(X, Y, val_X, val_Y):
	"""
	CNN
	"""
	model = Model(learning_rate = 0.001, batch_size = 32, epochs = 200, optimizer = None)
	model.add(Conv2D(2,(3,3), activation = 'tanh'))
	model.add(Maxpool((2,2), stride = 2)) # 16x16
	model.add(Dropout(0.5))
	model.add(Conv2D(4,(3,3), activation = 'tanh'))
	model.add(Maxpool((2,2), stride = 2)) # 8x8
	model.add(Dropout(0.5))
	model.add(Flatten())
	model.add(Dense(256,32, activation = 'tanh'))
	model.add(Dense(32,1))
	print("Begin Training")
	model.train(X,Y, val_X, val_Y)
	model.save_history("experiments/cnn.csv")


def cnn_adam(X, Y, val_X, val_Y):
	"""
	CNN - ADAM
	"""
	model = Model(learning_rate = 0.001, batch_size = 32, epochs = 200, optimizer = Adam())
	model.add(Conv2D(2,(3,3), activation = 'tanh'))
	model.add(Maxpool((2,2), stride = 2)) # 16x16
	model.add(Dropout(0.5))
	model.add(Conv2D(4,(3,3), activation = 'tanh'))
	model.add(Maxpool((2,2), stride = 2)) # 8x8
	model.add(Dropout(0.5))
	model.add(Flatten())
	model.add(Dense(256,32, activation = 'tanh'))
	model.add(Dense(32,1))
	print("Begin Training")
	model.train(X,Y, val_X, val_Y)
	model.save_history("experiments/cnn-adam.csv")


def relu_activation(X, Y, val_X, val_Y):
	"""
	RELU
	"""
	model = Model(learning_rate = 0.001, batch_size = 32, epochs = 1000, optimizer = None)
	model.add(Dropout(0.5))
	model.add(Dense(1024, 512, activation = 'relu'))
	model.add(Dropout(0.5))
	model.add(Dense(512, 64, activation = 'relu'))
	model.add(Dropout(0.5))
	model.add(Dense(64,1))
	print("Begin Training")
	model.train(X,Y, val_X, val_Y)
	model.save_history("experiments/relu.csv")
	print("The CSV file is saved in the experiments folder. You can plot the graph using plot.py")



def prepareTrainData(X, Y):
	

	for i in range(X.shape[0]):
		X[i,:] = X[i,:] -X[i,:].min()
		X[i,:] = X[i,:]/X[i,:].max() if X[i,:].max()>0 else 0

	means = []
	stds = []
	for i in range(X.shape[1]):
		means.append(X[:,i].mean())
		stds.append(X[:,i].std())
		X[:, i] = (X[:, i]-X[:, i].mean())/X[:, i].std()

	return X, Y, means, stds

def prepareTestData(X,Y,means,stds):
	
	for i in range(X.shape[0]):
		X[i,:] = X[i,:] -X[i,:].min()
		X[i,:] = X[i,:]/X[i,:].max() if X[i,:].max()>0 else 0


	for i in range(X.shape[1]):
		X[:,i] = (X[:,i] - means[i])/stds[i]

	return X,Y

print("ENTER THE EXPERIMENT NUMBER: ")
print("1. Experiment 1")
print("2. Experiment 2")
print("3. Experiment 3")
print("4. Experiment 4")
print("5. ANN with layers 1024-512-256-32-1 and sigmoid")
print("6. ANN with layers 1024-512-256-32-1 and sigmoid with Adam")
print("7. CNN with tanh activation")
print("8. CNN with tanh activation and Adam")
print("9. ANN with original layers and relu as activation")




choice = int(input())
if(choice == 1):
	X, Y = utils.get_data('steering', 0)
	num_examples = X.shape[0]
	split = int(num_examples*0.8)
	X_train, Y_train, means, stds = prepareTrainData(X[:split],Y[:split])
	X_test, Y_test = X[split:], Y[split:]
	X_val, Y_val = prepareTestData(X_test, Y_test, means, stds)
	experiment_one(X_train, Y_train, X_val, Y_val)
elif(choice == 2):
	X, Y = utils.get_data('steering', 0)
	num_examples = X.shape[0]
	split = int(num_examples*0.8)

	X_train, Y_train, means, stds = prepareTrainData(X[:split],Y[:split])
	X_test, Y_test = X[split:], Y[split:]
	X_val, Y_val = prepareTestData(X_test, Y_test, means, stds)
	experiment_two_32(X_train, Y_train, X_val, Y_val)
	experiment_two_64(X_train, Y_train, X_val, Y_val)
	experiment_two_128(X_train, Y_train, X_val, Y_val)
elif(choice == 3):
	X, Y = utils.get_data('steering', 0)
	num_examples = X.shape[0]
	split = int(num_examples*0.8)
	X_train, Y_train, means, stds = prepareTrainData(X[:split],Y[:split])
	X_test, Y_test = X[split:], Y[split:]
	X_val, Y_val = prepareTestData(X_test, Y_test, means, stds)
	experiment_three(X_train, Y_train, X_val, Y_val)
elif(choice == 4):
	X, Y = utils.get_data('steering', 0)
	X_train, Y_train, means, stds = prepareTrainData(X[:split],Y[:split])
	X_test, Y_test = X[split:], Y[split:]
	X_val, Y_val = prepareTestData(X_test, Y_test, means, stds)
	experiment_four_001(X_train, Y_train, X_val, Y_val)
	experiment_four_005(X_train, Y_train, X_val, Y_val)
	experiment_four_05(X_train, Y_train, X_val, Y_val)
elif(choice == 5):
	X, Y = utils.get_data('steering', 0)
	num_examples = X.shape[0]
	split = int(num_examples*0.8)
	X_train, Y_train, means, stds = prepareTrainData(X[:split],Y[:split])
	X_test, Y_test = X[split:], Y[split:]
	X_val, Y_val = prepareTestData(X_test, Y_test, means, stds)
	ann_layers(X_train, Y_train, X_val, Y_val)
elif(choice == 6):
	X, Y = utils.get_data('steering', 0)
	num_examples = X.shape[0]
	split = int(num_examples*0.8)
	X_train, Y_train, means, stds = prepareTrainData(X[:split],Y[:split])
	X_test, Y_test = X[split:], Y[split:]
	X_val, Y_val = prepareTestData(X_test, Y_test, means, stds)
	ann_adam_layers(X_train, Y_train, X_val, Y_val)
elif(choice == 7):
	X, Y = utils.get_data('steering', 1)
	num_examples = X.shape[0]
	split = int(num_examples*0.8)
	X_train, Y_train, means, stds = prepareTrainData(X[:split],Y[:split])
	X_test, Y_test = X[split:], Y[split:]
	X_val, Y_val = prepareTestData(X_test, Y_test, means, stds)
	cnn(X_train, Y_train, X_val, Y_val)
elif(choice == 8):
	X, Y = utils.get_data('steering', 1)
	num_examples = X.shape[0]
	split = int(num_examples*0.8)
	X_train, Y_train, means, stds = prepareTrainData(X[:split],Y[:split])
	X_test, Y_test = X[split:], Y[split:]
	X_val, Y_val = prepareTestData(X_test, Y_test, means, stds)
	cnn_adam(X_train, Y_train, X_val, Y_val)
elif(choice == 9):
	X, Y = utils.get_data('steering', 0)
	num_examples = X.shape[0]
	split = int(num_examples*0.8)
	X_train, Y_train, means, stds = prepareTrainData(X[:split],Y[:split])
	X_test, Y_test = X[split:], Y[split:]
	X_val, Y_val = prepareTestData(X_test, Y_test, means, stds)
	relu_activation(X_train, Y_train, X_val, Y_val)
else: print("Invalid Choice")



