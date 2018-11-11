Name: Karan Sehgal
Entry Number: 2016CSB1080

The project includes the following files:

# root directory
experiments.py: The experiments script file (to use to check the experiments)
utils.py: Contains some helper functions
main.py: File to experiment with the network
plot.py: File to create the graphs from the csv file stored in the experiments directory

# experiments directory
Contains the .csv files for the results for each experiment

# main network code
activations.py: Activation Class
adam.py: Class for ADAM optimizer
convnet.py: Class for CNNs
dense_layer.py: Class for dense_layer
dropout.py: Class for Dropouts
flatten.py: Class for Flatten Layer
maxpool.py: Class for Maxpooling Layers
im2col.py: helper function from Stanford's CS231n course.
batchnorm.py: Having some problem, not a part of the submission

### HOW TO RUN ###

to run, from root directory:
	$ python experiments.py
	
Then answer the commandline queries.

Note: Kindly install numpy and scipy before running the code.
Note #2: The dataset used did not have img_0.jpg hence it was removed from data.txt
Note #3: Make sure to keep the steering folder in the root(code) directory
