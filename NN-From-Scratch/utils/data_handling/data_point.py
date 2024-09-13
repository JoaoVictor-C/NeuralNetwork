import numpy as np

class DataPoint:
    def __init__(self, inputs, label, numLabels):
        self.inputs = inputs # 28x28 image
        self.inputs_array = np.array(inputs) # 784x1 array
        self.label = label
        self.expected_output = self.CreateOneHot(label, numLabels)


    def CreateOneHot(self, label, numLabels):
        one_hot = np.zeros(numLabels)
        one_hot[label] = 1
        return one_hot