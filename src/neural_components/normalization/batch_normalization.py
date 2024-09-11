import numpy as np

class BatchNormalization:
    def __init__(self, num_features, epsilon=1e-5, momentum=0.9):
        self.num_features = num_features
        self.epsilon = epsilon
        self.momentum = momentum
        self.gamma = np.ones((1, num_features))
        self.beta = np.zeros((1, num_features))
        self.moving_mean = np.zeros((1, num_features))
        self.moving_var = np.ones((1, num_features))
        self.cache = None

    def forward(self, x, training=True):
        if training:
            mean = np.mean(x, axis=0, keepdims=True)
            var = np.var(x, axis=0, keepdims=True)
            
            # Update moving averages
            self.moving_mean = self.momentum * self.moving_mean + (1 - self.momentum) * mean
            self.moving_var = self.momentum * self.moving_var + (1 - self.momentum) * var

            mean = self.moving_mean
            var = self.moving_var

        x_normalized = (x - mean) / np.sqrt(var + self.epsilon)
        out = self.gamma * x_normalized + self.beta
        
        if training:
            self.cache = (x, x_normalized, mean, var)
        
        return out

    def backward(self, dout):
        if self.cache is None:
            raise ValueError("Backward pass called before forward pass")

        x, x_normalized, mean, var = self.cache

        N, D = x.shape
        
        # Gradient w.r.t. gamma and beta
        dgamma = np.sum(dout * x_normalized, axis=0, keepdims=True)
        dbeta = np.sum(dout, axis=0, keepdims=True)

        # Gradient w.r.t. x_normalized
        dx_normalized = dout * self.gamma

        # Gradient w.r.t. var
        dvar = np.sum(dx_normalized * (x - mean) * -0.5 * (var + self.epsilon)**(-1.5), axis=0, keepdims=True)

        # Gradient w.r.t. mean
        dmean = np.sum(dx_normalized * -1 / np.sqrt(var + self.epsilon), axis=0, keepdims=True)
        dmean += dvar * np.mean(-2 * (x - mean), axis=0, keepdims=True)

        # Gradient w.r.t. x
        dx = dx_normalized / np.sqrt(var + self.epsilon)
        dx += dvar * 2 * (x - mean) / N
        dx += dmean / N

        return dx