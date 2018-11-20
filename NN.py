# This file is still in progress

import numpy as np
import pandas as pd


class NeuralNetwork(object):
    """"""

    def __init__(self,
                 input_layer_size = 784,
                 hidden_layer_sizes = (16,16),
                 output_layer_size = 10):

        self.input_layer_size = input_layer_size
        self.hidden_layer_sizes = hidden_layer_sizes
        self.output_layer_size = output_layer_size

        # initializing the weight matrix and activations
        self.weight_matrices = []
        self.activations = [None]
        self.biases = []
        for index, val in enumerate(hidden_layer_sizes):
            if index == 0:
                self.weight_matrices.append(np.random.randn(val, input_layer_size))
            else:
                self.weight_matrices.append(np.random.randn(val, hidden_layer_sizes[index - 1]))

            self.activations.append(np.zeros(val))
            self.biases.append(np.random.randn(val))

        #the weight matrix to the output layer
        self.weight_matrices.append(np.random.randn(output_layer_size, hidden_layer_sizes[-1:][0]))
        self.biases.append(np.random.randn(output_layer_size))

    def sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def get_training_data(self, filename):
        """"""
        df = pd.read_csv(filename)
        # input layers are put into a numpy array. Each row is one number
        self.input_layers = df.iloc[:,1:].values
        # actual numbers are put in a numpy array
        self.actual_numbers = df.iloc[:,0].values

    def get_test_data(self, filename):
        """"""

        df = pd.read_csv(filename)
        self.input_layers = df.iloc[:,:].values

    def forward_propagation(self, sample):
        """ All avtivations are calculated for one sample"""

        #TDOO: maybe the queue should be randomized
        # calculate the activations of the hidden layers
        self.activations[0] = self.input_layers[sample]
        for index, val in enumerate(self.hidden_layer_sizes):
            self.activations[index + 1] = self.sigmoid(self.weight_matrices[index].dot(self.activations[index]) + self.biases[index])

        # calculating the output layer activations
        output_activation = self.weight_matrices[-1:][0].dot(self.activations[-1:][0])
        actual_output_activation = np.zeros(shape = np.shape(output_activation))
        actual_output_activation[self.actual_numbers[sample]] = 1

        return output_activation, actual_output_activation


    def backward_propagation(self, sample):
        """ determines the gradient of one sample through backpropagation"""
        predicted_output_activation, actual_output_activation = nn.forward_propagation(sample)
        difference = predicted_output_activation - actual_output_activation

        self.activations.append(predicted_output_activation)
        #TODO: in this step the gradient values must be estimated
        gradient = []
        for index, val in enumerate(self.activations[::-1]):
            print(val)
            #TODO: calculate the weights and biases for each hidden layer following the calculus



    def training(self):
        """Iterating the backward propagation over all test samples"""

    def prediciton(self):
        """"""

# initialize NN
nn = NeuralNetwork(input_layer_size = 784, hidden_layer_sizes = (16,16), output_layer_size=10)

# import trainings data
nn.get_training_data('train.csv')

a = nn.forward_propagation(20)

print(a)

nn.backward_propagation(20)


# train the NN
#nn.training()

# import test data
#nn.get_test_data('test.csv')

# predict with the NN
#nn.prediction()
