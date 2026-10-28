#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Improved neural network for MNISt classification, following Nielsen's 'Neural networks and deep learning'
Implemented:
    - cross-entropy as cost/loss function
    - regularization (L1, L2, dropout, artificial training data)
    - improved initial weights and biases
    - (heuristically) better hyperparameters

Neural network is based on SGD learning algorithm for a feedforward pass and backpropagation of errors.
"""
# Import Libraries
import json
import random
import sys
import numpy as np


##############################################
# Cross-entropy and quadratic cost functions #
##############################################

class CrossEntropyCost(object):
    """
    Class containing the methods for cross-entropy cost computation.
    """
    @staticmethod
    def fn(a, y):
        """
        fn method to compute cross-entropy cost. np.nan_to_num deas with log of numbers very close to zero.
        """
        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))

    @staticmethod
    def delta(z, a, y):
        """
        delta method to compute output error. It is useful in backpropagation.
        """
        return (a-y)

class QuadraticCost(object):
    """
    Class containing the methods for quadratic cost computation
    """
    @staticmethod
    def fn(a, y):
        """
        fn method to compute quadratic cost.
        """
        return 0.5*np.linalg.norm(a-y)**2

    @staticmethod
    def delta(z, a, y):
        """
        delta method to compute output error. It is useful in backpropagation.
        """
        return (a-y) * sigmoid_prime(z)

#################################
# Improved neural network class #
#################################

class Network(object):
    """
    Network class represent the neural network.
    Sizes -> list of sizes for each layer
    cost  -> cost or loss function. Default is CrossEntropyCost
    """
    def __init__(self, sizes, cost=CrossEntropyCost):
        """
        Initialize the network.
        num_layers -> number of layers = len(sizes)
        default_weight_initializer -> initialize the weights and biases with dedicated method
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.cost = cost
        self.default_weight_initializer()

    def default_weight_initializer(self):
        """
        Initialize the weights and biases.
        Biases are drawn from normal distribution with mean 0 and standard deviation 1. No biases are needed for the input neurons.
        Weights are drawn from normal distribution with mean 0 and standard deviation 1/sqrt{x} where x is the number of connections input to the neuron.
        """
        self.biases = [np.random.randn(y,1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y,x)/np.sqrt(x) for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def large_weight_initializer(self):
        """
        Initialize the weights and biases.
        Biases are drawn from normal distribution with mean 0 and standard deviation 1. No biases are needed for the input neurons.
        Weights are drawn from normal distribution with mean 0 and standard deviation 1.
        """
        self.biases = [np.random.randn(y,1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y,x) for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def feedforward(self, a):
        """
        Feedforward pass. Return the output of the network when ``a`` is input.
        Activation function is sigmoid function.
        Feedforward is given by consecutive affine transformation plus activation function.
        """
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w,a)+b)
        return a

    def SGD(
        self,
        training_data, 
        epochs, 
        mini_batch_size, 
        eta,
        lmbda = 0.0,
        evaluation_data=None,
        monitor_evaluation_cost=False,
        monitor_evaluation_accuracy=False,
        monitor_training_cost=False,
        monitor_training_accuracy=False):
        """
        Train the neural network with stochastic gradient descent.
        Training_data               -> tuple (x,y) of labeled training data
        epochs                      -> number of epochs for training
        mini_batch_size             -> size of each batch of train data. Each epoch is completed when all the len(training_data)//mini_batches_size batches are used.
        eta                         -> learning rate
        lmbda                       -> regularization parameter 
        evaluation_data             -> if passed then evaluates training against evaluationd data
        monitor_evaluation_cost     -> compute cost on evaluation data
        monitor_evaluation_accuracy -> compute accuracy on evaluation data
        monitor_training_cost       -> compute cost on training data
        monitor_training_accuracy   -> compute accuracy on training data
        """
        if evaluation_data: n_eval = len(evaluation_data)
        n = len(training_data)
        evaluation_cost, evaluation_accuracy = [], []
        training_cost, training_accuracy =  [], []
        # Loop on each epoch
        for j in range(epochs):
            # Shuffle
            random.shuffle(training_data)
            # Create list of mini_batches
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0,n,mini_batch_size)]
            # Loop on each mini batch
            for mini_batch in mini_batches:
                # SGD happens here 
                self.update_mini_batch(mini_batch, eta, lmbda, len(training_data))
            print("Epoch %s training complete" % j)
            # Monitor trainign/evaluation cost and accuracy
            if monitor_training_cost:
                cost = self.total_cost(training_data, lmbda)
                training_cost.append(cost)
                print("Cost on training data: {}".format(cost))
            if monitor_training_accuracy:
                accuracy = self.accuracy(training_data, convert=True)
                training_accuracy.append(accuracy)
                print("Accuracy on training data: {} / {}".format(accuracy, n))
            
            if monitor_evaluation_cost:
                cost = self.total_cost(evaluation_data, lmbda, convert=True)
                evaluation_cost.append(cost)
                print("Cost on evaluation data: {}".format(cost))
            if monitor_evaluation_accuracy:
                accuracy = self.accuracy(evaluation_data)
                evaluation_accuracy.append(accuracy)
                print("Accuracy on evaluation data: {} / {}".format(accuracy, n_eval))
            print("")
        return evaluation_cost, evaluation_accuracy, training_cost, training_accuracy

    def update_mini_batch(self, mini_batch, eta, lmbda, n):
        """
        Update weights and biases using gradient descent via backpropagation for a single mini-batch..
        mini_batch -> tuple (x,y) of labeled data
        eta -> learning rate
        lmbda -> regularization parameter 
        n -> total size of the training data set 
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x,y in mini_batch:
            # Calculate gradients with backpropagation
            delta_nabla_b, delta_nabla_w = self.backprop(x,y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        # Gradient descent update
        self.weights = [(1-eta*lmbda/n)*w-(eta/len(mini_batch))*nw for w,nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """
        Backpropagation method  to compute the gradient of the cost function.
        x is activation and y is output
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        # Feedforward pass
        activation = x
        activations = [x]       # Store all activations
        zs = []                 # List to store all weighted inputs
        for b,w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # Backpropagate error
        delta = (self.cost).delta(zs[-1], activations[-1], y)   # Calculate output error
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for i in range(2, self.num_layers):
            z = zs[-i]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-i+1].transpose(), delta) * sp
            nabla_b[-i] = delta
            nabla_w[-i] = np.dot(delta, activations[-i-1].transpose())
        return (nabla_b, nabla_w)

    def accuracy(self, data, convert=False):
        """
        Return the number of correctly classified inputs.
        The convert flag needs to be False if function is used on validation or test data, while True if the dataset is the training one. 
        """
        if convert:
            results = [(np.argmax(self.feedforward(x)), np.argmax(y)) for (x,y) in data]
        else:
            results = [(np.argmax(self.feedforward(x)), y) for (x,y) in data]
        return sum(int(x==y) for (x,y) in results)

    def total_cost(self, data, lmbda, convert=False):
        """
        Method to compute total cost or loss for the data set. 
        The convert flag should be False if the data set is the training data set, while 
        True if the dataset is validation or test data.
        """
        cost = 0.0
        for x,y in data:
            a = self.feedforward(x)
            if convert: y = vectorized_result(y)
            cost += self.cost.fn(a,y)/len(data)
        cost += 0.5*(lmbda/len(data))*sum(np.linalg.norm(w)**2 for w in self.weights)
        return cost

    def save(self, filename):
        """
        Save the neural network into file with name filename
        """
        data = {
            "sizes": self.sizes,
            "weights": [w.tolist() for w in self.weights],
            "biases": [b.tolist() for b in self.biases],
            "cost": str(self.cost.__name__)
        }
        f = open(filename, "w")
        json.dump(data, f)
        f.close()



####################
# Useful functions #
####################

def load(filename):
    """
    Load neural network fromfile
    """
    f = open(filename, "r")
    data = json.load(f)
    f.close()
    cost = getattr(sys.modules[__name__], data["cost"])
    net = Network(data["sizes"], cost=cost)
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    return net

def vectorized_result(j):
    """
    Return a 10 component unit vetor with a 1 in the j-th position and zero elsewhere.
    """
    e = np.zeros((10,1))
    e[j] = 1.0
    return e

def sigmoid(z):
    """
    Sigmoid function
    """
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """
    Derivative of sigmoid function
    """
    return sigmoid(z)*(1-sigmoid(z))






        