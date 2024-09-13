import numpy as np

class Image:
    def __init__(self, inputs, label, size):
        self.inputs = inputs
        self.image_array = np.array(inputs)
        self.label = label
        self.size = size
        