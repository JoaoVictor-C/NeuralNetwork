import numpy as np
import json
import os

class Optimizers:
    def __init__(self, name): 
        self.name = name
        self.config = self.load_config()
        self.optimizers = {
            'sgd': self.SGD(self.config['sgd']['learning_rate'], self.config['sgd']['momentum']),
            'adam': self.Adam(self.config['adam']['learning_rate'], self.config['adam']['beta1'], self.config['adam']['beta2'], self.config['adam']['epsilon']),
            'rmsprop': self.RMSprop(self.config['rmsprop']['learning_rate'], self.config['rmsprop']['rho'], self.config['rmsprop']['epsilon']),
            'adagrad': self.Adagrad(self.config['adagrad']['learning_rate'], self.config['adagrad']['epsilon']),
            'adadelta': self.Adadelta(self.config['adadelta']['learning_rate'], self.config['adadelta']['rho'], self.config['adadelta']['epsilon']),
            'adamax': self.Adamax(self.config['adamax']['learning_rate'], self.config['adamax']['beta1'], self.config['adamax']['beta2'], self.config['adamax']['epsilon']),
            'nadam': self.Nadam(self.config['nadam']['learning_rate'], self.config['nadam']['beta1'], self.config['nadam']['beta2'], self.config['nadam']['epsilon']),
            'pso': self.PSO(self.config['pso']['learning_rate'], self.config['pso']['inertia_weight'], self.config['pso']['cognitive_weight'], self.config['pso']['social_weight'], 
                            self.config['pso']['particle_count'], self.config['pso']['max_iterations'], self.config['pso']['min_velocity'], self.config['pso']['max_velocity']),
            'momentum': self.Momentum(self.config['momentum']['learning_rate'], self.config['momentum']['beta']),
            'nesterov': self.Nesterov(self.config['nesterov']['learning_rate'], self.config['nesterov']['beta'])
        }

    def load_config(self):
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'optimizers.json')
        with open(config_path, 'r') as f:
            return json.load(f)

    def get_optimizer(self):
        if self.name not in self.optimizers:
            raise ValueError(f"Optimizer {self.name} not found")
        return self.optimizers[self.name]

    class Optimizer:
        def __init__(self):
            pass
        
        def update_weights(self, weights, gradients):
            pass

        def update_biases(self, biases, gradients):
            pass

    class SGD(Optimizer):
        def __init__(self, learning_rate, momentum=0):
            self.learning_rate = learning_rate
            self.momentum = momentum
            self.velocity_weights = 0
            self.velocity_biases = 0

        def update_weights(self, weights, gradients):
            self.velocity_weights = self.momentum * self.velocity_weights - self.learning_rate * gradients
            return weights + self.velocity_weights

        def update_biases(self, biases, gradients):
            self.velocity_biases = self.momentum * self.velocity_biases - self.learning_rate * gradients
            return biases + self.velocity_biases

    class Adam(Optimizer):
        def __init__(self, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8):
            self.learning_rate = learning_rate
            self.beta1 = beta1
            self.beta2 = beta2
            self.epsilon = epsilon
            self.t = 0
            self.m_weights = None
            self.v_weights = None
            self.m_biases = None
            self.v_biases = None

        def update_weights(self, weights, gradients):
            self.t += 1
            if self.m_weights is None:
                self.m_weights = np.zeros_like(weights)
                self.v_weights = np.zeros_like(weights)
            
            self.m_weights = self.beta1 * self.m_weights + (1 - self.beta1) * gradients
            self.v_weights = self.beta2 * self.v_weights + (1 - self.beta2) * np.square(gradients)
            m_hat = self.m_weights / (1 - self.beta1 ** self.t)
            v_hat = self.v_weights / (1 - self.beta2 ** self.t)
            return weights - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

        def update_biases(self, biases, gradients):
            if self.m_biases is None:
                self.m_biases = np.zeros_like(biases)
                self.v_biases = np.zeros_like(biases)
            
            self.m_biases = self.beta1 * self.m_biases + (1 - self.beta1) * gradients
            self.v_biases = self.beta2 * self.v_biases + (1 - self.beta2) * np.square(gradients)
            m_hat = self.m_biases / (1 - self.beta1 ** self.t)
            v_hat = self.v_biases / (1 - self.beta2 ** self.t)
            return biases - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

    class RMSprop(Optimizer):
        def __init__(self, learning_rate, rho=0.9, epsilon=1e-8):
            self.learning_rate = learning_rate
            self.rho = rho
            self.epsilon = epsilon
            self.cache_weights = None
            self.cache_biases = None
            
        def update_weights(self, weights, gradients):
            if self.cache_weights is None:
                self.cache_weights = np.zeros_like(weights)
            self.cache_weights = self.rho * self.cache_weights + (1 - self.rho) * np.square(gradients)
            return weights - self.learning_rate * gradients / (np.sqrt(self.cache_weights) + self.epsilon)

        def update_biases(self, biases, gradients):
            if self.cache_biases is None:
                self.cache_biases = np.zeros_like(biases)
            self.cache_biases = self.rho * self.cache_biases + (1 - self.rho) * np.square(gradients)
            return biases - self.learning_rate * gradients / (np.sqrt(self.cache_biases) + self.epsilon)

    class Adagrad(Optimizer):
        def __init__(self, learning_rate, epsilon=1e-8):
            self.learning_rate = learning_rate
            self.epsilon = epsilon
            self.cache_weights = 0
            self.cache_biases = 0
            
        def update_weights(self, weights, gradients):
            self.cache_weights += np.square(gradients)
            return weights - self.learning_rate * gradients / (np.sqrt(self.cache_weights) + self.epsilon)  

        def update_biases(self, biases, gradients):
            self.cache_biases += np.square(gradients)
            return biases - self.learning_rate * gradients / (np.sqrt(self.cache_biases) + self.epsilon)

    class Adadelta(Optimizer):
        def __init__(self, learning_rate, rho=0.95, epsilon=1e-8):
            self.learning_rate = learning_rate
            self.rho = rho
            self.epsilon = epsilon
            self.cache_weights = 0
            self.cache_delta_weights = 0
            self.cache_biases = 0
            self.cache_delta_biases = 0
            
        def update_weights(self, weights, gradients):
            self.cache_weights = self.rho * self.cache_weights + (1 - self.rho) * np.square(gradients)
            delta = np.sqrt((self.cache_delta_weights + self.epsilon) / (self.cache_weights + self.epsilon)) * gradients
            self.cache_delta_weights = self.rho * self.cache_delta_weights + (1 - self.rho) * np.square(delta)
            return weights - self.learning_rate * delta

        def update_biases(self, biases, gradients):
            self.cache_biases = self.rho * self.cache_biases + (1 - self.rho) * np.square(gradients)
            delta = np.sqrt((self.cache_delta_biases + self.epsilon) / (self.cache_biases + self.epsilon)) * gradients
            self.cache_delta_biases = self.rho * self.cache_delta_biases + (1 - self.rho) * np.square(delta)
            return biases - self.learning_rate * delta

    class Adamax(Optimizer):
        def __init__(self, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8):
            self.learning_rate = learning_rate
            self.beta1 = beta1
            self.beta2 = beta2
            self.epsilon = epsilon
            self.t = 0
            self.m_weights = None
            self.u_weights = None
            self.m_biases = None
            self.u_biases = None
            
        def update_weights(self, weights, gradients):
            self.t += 1
            if self.m_weights is None:
                self.m_weights = np.zeros_like(weights)
                self.u_weights = np.zeros_like(weights)
            
            self.m_weights = self.beta1 * self.m_weights + (1 - self.beta1) * gradients
            self.u_weights = np.maximum(self.beta2 * self.u_weights, np.abs(gradients))
            m_hat = self.m_weights / (1 - self.beta1 ** self.t)
            return weights - self.learning_rate * m_hat / (self.u_weights + self.epsilon)

        def update_biases(self, biases, gradients):
            if self.m_biases is None:
                self.m_biases = np.zeros_like(biases)
                self.u_biases = np.zeros_like(biases)
            
            self.m_biases = self.beta1 * self.m_biases + (1 - self.beta1) * gradients
            self.u_biases = np.maximum(self.beta2 * self.u_biases, np.abs(gradients))
            m_hat = self.m_biases / (1 - self.beta1 ** self.t)
            return biases - self.learning_rate * m_hat / (self.u_biases + self.epsilon)

    class Nadam(Optimizer):
        def __init__(self, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8):
            self.learning_rate = learning_rate
            self.beta1 = beta1
            self.beta2 = beta2
            self.epsilon = epsilon
            self.t = 0
            self.m_weights = None
            self.v_weights = None
            self.m_biases = None
            self.v_biases = None
            
        def update_weights(self, weights, gradients):
            self.t += 1
            if self.m_weights is None:
                self.m_weights = np.zeros_like(weights)
                self.v_weights = np.zeros_like(weights)
            
            self.m_weights = self.beta1 * self.m_weights + (1 - self.beta1) * gradients
            self.v_weights = self.beta2 * self.v_weights + (1 - self.beta2) * np.square(gradients)
            m_hat = self.m_weights / (1 - self.beta1 ** self.t)
            v_hat = self.v_weights / (1 - self.beta2 ** self.t)
            return weights - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

        def update_biases(self, biases, gradients):
            if self.m_biases is None:
                self.m_biases = np.zeros_like(biases)
                self.v_biases = np.zeros_like(biases)
            
            self.m_biases = self.beta1 * self.m_biases + (1 - self.beta1) * gradients
            self.v_biases = self.beta2 * self.v_biases + (1 - self.beta2) * np.square(gradients)
            m_hat = self.m_biases / (1 - self.beta1 ** self.t)
            v_hat = self.v_biases / (1 - self.beta2 ** self.t)
            return biases - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

    class PSO(Optimizer):
        def __init__(self, learning_rate, inertia_weight, cognitive_weight, social_weight, 
                     particle_count, max_iterations, min_velocity, max_velocity):
            self.learning_rate = learning_rate
            self.inertia_weight = inertia_weight
            self.cognitive_weight = cognitive_weight
            self.social_weight = social_weight
            self.particle_count = particle_count
            self.max_iterations = max_iterations
            self.min_velocity = min_velocity
            self.max_velocity = max_velocity
            
        def update_weights(self, weights, gradients):
            return weights - self.learning_rate * gradients

        def update_biases(self, biases, gradients):
            return biases - self.learning_rate * gradients

    class Momentum(Optimizer):
        def __init__(self, learning_rate, beta=0.9, v=0):
            self.learning_rate = learning_rate
            self.beta = beta
            self.v_weights = v
            self.v_biases = v
            
        def update_weights(self, weights, gradients):
            self.v_weights = self.beta * self.v_weights + (1 - self.beta) * gradients
            return weights - self.learning_rate * self.v_weights

        def update_biases(self, biases, gradients):
            self.v_biases = self.beta * self.v_biases + (1 - self.beta) * gradients
            return biases - self.learning_rate * self.v_biases

    class Nesterov(Optimizer):
        def __init__(self, learning_rate, beta=0.9, v=0):
            self.learning_rate = learning_rate
            self.beta = beta
            self.v_weights = v
            self.v_biases = v
            
        def update_weights(self, weights, gradients):
            self.v_weights = self.beta * self.v_weights + (1 - self.beta) * gradients
            return weights - self.learning_rate * self.v_weights

        def update_biases(self, biases, gradients):
            self.v_biases = self.beta * self.v_biases + (1 - self.beta) * gradients
            return biases - self.learning_rate * self.v_biases

