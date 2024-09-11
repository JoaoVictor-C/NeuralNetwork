import numpy as np
import json
from src.core.neural_network import NeuralNetwork
from src.utils import fancy_print
from colorama import Fore, Style
import time

class Ensemble:
    def __init__(self, dataset):
        self.dataset = dataset
        self.load_config(dataset)
        self.models = []        
        self.training_loss = []
        self.training_acc = []
        self.test_acc = []

    def load_config(self, dataset):
        with open('src/config/NN_parameters.json', 'r') as f:
            self.config = json.load(f)[dataset]
        with open('src/config/base_parameters.json', 'r') as f:
            self.base_config = json.load(f)[dataset]

    def create_models(self):
        for i in range(self.config['num_models']):
            model = NeuralNetwork(self.dataset, i)
            self.models.append(model)

    def train(self, X_train, y_train, X_test, y_test):
        start_time = time.time()
        self.create_models()
        for model in self.models:
            fancy_print(f"Training model {model.model_index+1}/{self.config['num_models']}", Fore.CYAN, Style.BRIGHT)
            model.train(X_train, y_train, X_test, y_test)
            self.training_loss.append(np.mean(model.training_loss))
            self.training_acc.append(np.mean(model.training_acc))
            self.test_acc.append(np.mean(model.test_acc))
            fancy_print(f"Model {model.model_index+1} trained successfully!", Fore.GREEN)
        self.final_training_log(start_time)
        self.save_models()

    def predict(self, X):
        if self.config['num_models'] == 1:
            return self.models[0].predict(X)
        else:
            predictions = np.array([model.predict(X) for model in self.models])
            return np.mean(predictions, axis=0)

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        if y_test.ndim == 2:
            accuracy = np.mean(np.argmax(y_pred, axis=1) == np.argmax(y_test, axis=1))
        else:
            accuracy = np.mean(np.argmax(y_pred, axis=1) == y_test)
        return accuracy
    
    def save_models(self):
        for i, model in enumerate(self.models):
            model.save_model(i)

    def load_models(self):
        for i, model in enumerate(self.models):
            model.load(f"models/model_{i}.pkl")

    def final_training_log(self, start_time):
        fancy_print("Training completed!", Fore.GREEN, Style.BRIGHT)
        fancy_print(f"Training Accuracy average: {np.mean(self.training_acc)*100:.2f}%", Fore.CYAN)
        fancy_print(f"Training Loss average: {np.mean(self.training_loss):.4f}", Fore.CYAN)
        fancy_print(f"Test Accuracy average: {np.mean(self.test_acc)*100:.2f}%", Fore.CYAN)
        fancy_print(f"Total training time: {time.time() - start_time:.2f}s", Fore.CYAN)