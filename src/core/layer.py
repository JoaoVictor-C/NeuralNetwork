import numpy as np
import json
from ..activation import Activation
from ..optimizers import Optimizers
from ..regularization import Regularization

class Layer:
    def __init__(self, input_size, neurons, activation, config):
        self.input_size = input_size
        self.neurons = neurons
        self.activation = activation
        
        # Xavier/Glorot initialization
        limit = np.sqrt(6 / (input_size + neurons))
        self.weights = np.random.uniform(-limit, limit, (input_size, neurons))
        self.biases = np.zeros((1, neurons))

        self.config = config
        self.optimizer = Optimizers(config['optimizer']).get_optimizer()
        self.regularization = Regularization(config['regularization']['type'], config['regularization']['lambda'])

    def forward(self, inputs):
        self.inputs = inputs
        inputs_reshaped = inputs.reshape(inputs.shape[0], -1)
        self.z = np.dot(inputs_reshaped, self.weights) + self.biases
        self.output = self.activation.forward(self.z)
        return self.output

    def backward(self, delta, num_samples):
        if isinstance(self.activation, Activation().Softmax):
            # For Softmax, the delta is already correct (combined with CrossEntropy)
            delta_activated = delta
        else:
            delta_activated = delta * self.activation.derivative(self.output)
        
        self.dweights = np.dot(self.inputs.reshape(self.inputs.shape[0], -1).T, delta_activated) / num_samples
        self.dbiases = np.sum(delta_activated, axis=0, keepdims=True) / num_samples
        
        # Apply regularization to gradients
        reg_term, reg_grad = self.regularization.regularize(self.weights)
        self.dweights += reg_grad / num_samples
        
        if self.optimizer:
            self.weights = self.optimizer.update_weights(weights=self.weights, gradients=self.dweights)
            self.biases = self.optimizer.update_biases(biases=self.biases, gradients=self.dbiases)
        
        return np.dot(delta_activated, self.weights.T)

    def get_regularization_loss(self):
        reg_term, _ = self.regularization.regularize(self.weights)
        return reg_term