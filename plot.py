import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys

if len(sys.argv) < 2:
	print("Enter the CSV file as an argument")
	exit()

data = pd.read_csv(sys.argv[1], delimiter = ',', header = None)
# data2 = pd.read_csv(sys.argv[2], delimiter = ',', header = None)
# data2 = np.array(data2)
# data3 = pd.read_csv(sys.argv[3], delimiter = ',', header = None)
# data3 = np.array(data3)
data = np.loadtxt(sys.argv[1], delimiter = ',')
data = np.array(data)
fig, ax = plt.subplots()

data2 = np.array([data[i,:] for i in range(data.shape[0]) if i%1 == 0])



plt.suptitle("Squared Loss")
plt.title("Learning Rate: 0.001, Batch Size: 32")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.ylim(-0.01,0.21)
ax.plot(data2[:,0], data2[:,1], color = 'r', label = 'Training Loss')
ax.plot(data2[:,0], data2[:,2], color = 'b', label = 'Validation Loss')
# ax.plot(data3[:,0], data3[:,1], color = 'g', label = 'Test Accuracy')
legend = plt.legend(loc='upper right')
plt.show()