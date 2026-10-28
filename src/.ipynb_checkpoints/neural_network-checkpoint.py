#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import random

# Useful functions for activation function and its derivative
def sigmoid(z):
    """
    Calculate the sigmoid function of z: sigmoid(z) = 1 / (1 + e^{-z})
    """
    return 1/(1+np.exp(-z))


def sigmoid_prime(z):
    """
    Calculate the derivative of the sigmoid function as sigmoid'(z) = sigmoid(z)*(1-sigmoid(z))
    """
    sig = sigmoid(z)
    return sig*(1-sig)

# Neural network module
class NeuralNetwork(object):
    """
    This class defines methods used to create a neural network.
    In the training, I use the SGD (stochastic gradient descent) method
    """

    def __init__(self, layer_sizes):
        """
        Layer size is a list of positive integers that specifies the number of neuros per layer.
        len(layer_sizes) -> number of layers
        layer_sizes[0]   -> number of neurons in input layer
        layer_sizes[1]   -> number of neuros in first hidden layer
        .
        .
        .
        layer_sizes[-1]  -> number of neurons in output layer

        Weights and biases are created sampling a standard distribution.
        These quantities are arranged in a matrix (weights) and a vector (biases).
        They represent the initial configuration of the neural network

        Raises ValueError if the number of layers is smaller than two
        or if not all specified numbers are positive integers.
        """
        # Simple sanity checks
        if len(layer_sizes) < 2:
            raise ValueError("layer_sizes must contain at least 2 layers.")
        if not all(isinstance(n, int) and n > 0 for n in layer_sizes):
            raise ValueError(f"All layer sizes must be positive integers, got {layer_sizes}.")

        
        self.num_layers = len(layer_sizes)
        self.layer_sizes = layer_sizes

        # Biases are generated for hidden layers, excluding input neurons
        # Weights are stored in matrices, nn.weights[n] containes the weights connectin the n+1 and n+2 layers
        self.biases = [np.random.randn(n,1) for n in layer_sizes[1:]]
        self.weights = [np.random.randn(n,m) for n,m in zip(layer_sizes[1:], layer_sizes[:-1])]


    def set_weights_and_biases(self, w, b):
        """
        This method allows to set the weights and biases.
        It is useful to load a saved configuration or
        if we want to start the network with a specific configuration
        """
        # TODO: add sanity check
        self.weights = w
        self.biases = b
        
    
    def feed_forward(self, a):
        """
        This method returns the output state of the neural network
        after taking the input state a.

        Sigmoid -> nonlinear activation function
        """
        for w,b in zip(self.weights, self.biases):
            a = sigmoid(np.dot(w,a) + b)
        return a


    def back_propagation(self, a, target):
        """
        Back propagation method to compute the gradients of the Euclidean distance
        between the output vector of the network 'a' and the target vector 'target'.

        First there is a feed forward step with a to compute the output.
        Relevant terms for calculating gradients are stored in the lists
        'activations' and 'zed'.
        """
        # List for nabla storing
        nabla_W = [np.zeros(w.shape) for w in self.weights]
        nabla_B = [np.zeros(b.shape) for b in self.biases]
        # List to store all the activations and the z vectors layer by layer
        activations = [a]
        activation = a
        zs = []

        # Feed forward
        for w, b in zip(self.weights, self.biases):
            # z vector
            z = np.dot(w,activation) + b
            zs.append(z)

            # activations
            activation = sigmoid(z)
            activations.append(activation)

        # Calculate gradient with respect to z for output layer
        delta = self.cost_derivative(activations[-1],target)*sigmoid_prime(zs[-1])

        # Gradients with respect to weights and biases are stored here
        nabla_W[-1] = np.dot(delta, activations[-2].transpose())
        nabla_B[-1] = delta

        """
        Propagates backward: compute gradients for previous layer 
        with quantities computed above
        """
        for l in range(2,self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_B[-l] = delta
            nabla_W[-l] = np.dot(delta, activations[-l-1].transpose())
            
        return (nabla_W, nabla_B)

    def evaluate(self, test_data):
        """
        Return the number of correct results against the test data
        """
        # Cycle over test data and evaluate on trained network
        test_results = [(np.argmax(self.feed_forward(x)), y) for (x,y) in test_data]

        return sum(int(x==y) for (x,y) in test_results)

    def cost_derivative(self, output_activations, y):
        """
        Return the vector of partial derivative of cost function
        for the output activations \partial C_x / \partial a
        """
        return (output_activations - y)
    
    def update(self, mini_batch, eta):
        """
        This method updates network weights and biases by applying backpropagation and gradient descent method.
        The learning rate of the gradient descent is eta.
        The mini_batch is a list of tuple (x,y) where x is an input vector and y is the target vector.
        """
        nabla_W = [np.zeros(w.shape) for w in self.weights]
        nabla_B = [np.zeros(b.shape) for b in self.biases]

        # Number of data in mini-batch
        n = len(mini_batch)

        for x,y in mini_batch:
            # Backpropagates the error
            delta_nabla_W, delta_nabla_B = self.back_propagation(x, y)
            # Store gradients
            nabla_W = [nW + dnW for nW, dnW in zip(nabla_W, delta_nabla_W)]
            nabla_B = [nB + dnB for nB, dnB in zip(nabla_B, delta_nabla_B)]

        # Update with gradient descent
        self.weights = [W - (eta/n)*nW for W,nW in zip(self.weights, nabla_W)]
        self.biases  = [B - (eta/n)*nB for B,nB in zip(self.biases, nabla_B)]
    
    
    def SGD(self, train_data, epochs, mini_batch_size, eta, test_data=None):
        """
        This method trains the network using stochastic gradient descent method (SGD).
        train_data       -> list of tuple of (x,y) as (input/target) vectors
        epochs           -> number of epochs to train the data
        mini_batch_size  -> size of a single mini_batch.
        eta              -> learning rate
        test_data        -> optional test_data for evaluation after each epoch useful for tracking process

        The total number of batches is len(train)/mini_batch_size
        and an epoch is concluded when all the batches have been considered.
        """

        # Define optional number of test data
        if test_data: n_test = len(test_data)

        # Number of training data
        n = len(train_data)

        for j in range(epochs):
            # Shuffle data
            random.shuffle(train_data)
            
            # Define mini batches
            mini_batches = [train_data[k:k+mini_batch_size] for k in range(0,n,mini_batch_size)]
            
            # Update mini batch loop
            for mini_batch in mini_batches:
                self.update(mini_batch, eta)

            # Evaluate against test data
            if test_data:
                print("Epoch {0}: {1}/{2}".format(j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j))

















        