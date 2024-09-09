import numpy as np

class Regularization:
    def __init__(self, reg_type='l2', lambda_param=0.01):
        self.reg_type = reg_type.lower()
        self.lambda_param = lambda_param

        if self.reg_type not in ['l1', 'l2']:
            raise ValueError("Regularization type must be 'l1' or 'l2'")

    def regularize(self, weights):
        if self.reg_type == 'l1':
            return self.l1(weights)
        elif self.reg_type == 'l2':
            return self.l2(weights)

    def l1(self, weights):
        """
        L1 regularization (Lasso)
        """
        reg_term = self.lambda_param * np.sum(np.abs(weights))
        reg_grad = self.lambda_param * np.sign(weights)
        return reg_term, reg_grad

    def l2(self, weights):
        """
        L2 regularization (Ridge)
        """
        reg_term = 0.5 * self.lambda_param * np.sum(np.square(weights))
        reg_grad = self.lambda_param * weights
        return reg_term, reg_grad

    def get_config(self):
        return {
            "reg_type": self.reg_type,
            "lambda_param": self.lambda_param
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)
