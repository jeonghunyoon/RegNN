# -*- coding: utf-8 -*-

import random
import numpy as np

class RegNN(object):
    # input_list : an input list containing the node number of each layer. ex, [10, 5, 6]
    def __init__(self, input_list):
        self.num_layers = len(input_list)
        self.input_list = input_list
        # initial value setting
        # randn() returns samples from drived from standard normal dist.
        self.biases = [np.random.randn(y, 1) for y in input_list[1:]]
        self.weights = [np.random.randn(x, y) for x, y in zip(input_list[1:], input_list[:-1])]

    # input : a, ouput : model's output
    def feed_forward(self, z):
        for b, w in zip(self.biases, self.weights):
            # multiplication of matrix w and input vector z
            a = np.dot(w, z) + b
            z = sigmoid(a)
        return z

    # stochastic gradient descent
    def gradient_descent(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        # after each epochs, model will be tested partially.
        if test_data: num_of_test = len(test_data)
        n = len(training_data)
        for i in xrange(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k + mini_batch_size] for k in xrange(0, n, mini_batch_size)]
            # on-line gradient descent(one data point)  => mini-batch gradient descent
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print "Epochs {0}: {1} / {2}".format(i, self.evaluate(test_data), num_of_test)
            else:
                print "Epochs {0} complete.".format(i)

    # update, eta : learning rate, mini_batch : (x, y)
    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.back_propagation(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w - (eta / len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)]

    def back_propagation(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        activation = x
        activations = [x]
        set_a = []

        # feed forward
        for b, w in zip(self.biases, self.weights):
            a = np.dot(w, activation) + b
            set_a.append(a)
            activation = sigmoid(a)
            activations.append(activation)

        # backward pass
        # in output layer
        # output layer의 activation 함수: linear function for regression
        delta = self.cost_derivatives(activations[-1], y)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in xrange(2, self.num_layers):
            a = set_a[-l]
            sp = sigmoid_prime(a)
            delta = np.dot(self.weights[-l + 1].T, delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].T)

        return (nabla_b, nabla_w)

    def cost_derivatives(self, output_activation, y):
        return (output_activation - y)

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feed_forward(x)), y) for x, y in test_data]
        return sum(int(x == y) for (x, y) in test_results)


def sigmoid(a):
    z = 1.0 / (1.0 + np.exp(-1 * a))
    return z


def sigmoid_prime(a):
    return sigmoid(a) * (1 - sigmoid(a))