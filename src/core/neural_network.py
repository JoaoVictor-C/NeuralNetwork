from ..neural_components.learning_rate import LearningRateScheduler
from ..neural_components.activation import Activation
from ..neural_components.cost import Cost
from .layer import Layer
from ..utils import fancy_print
import numpy as np
import json
import time
import pickle
import os
from colorama import init, Fore, Style

init(autoreset=True)  # Initialize colorama

def fancy_print(message, color=Fore.WHITE, style=Style.NORMAL):
    print(f"{style}{color}{message}{Style.RESET_ALL}")

class NeuralNetwork:
    def __init__(self, dataset, model_index=None,  verbose=True):
        if verbose:
            fancy_print(f"Initializing Neural Network {model_index}...", Fore.CYAN, Style.BRIGHT)
        self.load_config(dataset)
        self.activation = Activation()
        self.layers = []
        self.create_layers()
        self.model_index = model_index
        self.cost = Cost()
        self.cost_function = self.cost.get_cost_function(self.config['cost_function'])
        self.verbose = verbose
        self.training_acc = []
        self.training_loss = []
        self.test_acc = []

        self.lr_scheduler = LearningRateScheduler(self.config)

        if verbose:
            fancy_print(f"Neural Network {model_index} initialized successfully!", Fore.GREEN)

        self.best_loss = float('inf')
        self.best_model = None
        self.patience = self.config['patience']
        self.no_improvement_count = 0

    def load_config(self, dataset):
        with open('src/config/NN_parameters.json', 'r') as f:
            self.config = json.load(f)[dataset]
        with open('src/config/base_parameters.json', 'r') as f:
            self.base_config = json.load(f)[dataset]

    def create_layers(self):
        layer_sizes = self.config['layers']
        
        for i in range(len(layer_sizes) - 1):
            input_size = layer_sizes[i]
            output_size = layer_sizes[i+1]
            
            if i == len(layer_sizes) - 2:
                # Output layer
                activation = self.activation.get_activation_function(self.config['output_activation'])
                use_batch_norm = False  # Don't use batch norm for output layer
            else:
                # Input and hidden layers
                activation = self.activation.get_activation_function(self.config['activation_function'])
                use_batch_norm = self.config.get('use_batch_norm', False)
            
            layer_config = self.config.copy()
            layer_config['use_batch_norm'] = use_batch_norm
            self.layers.append(Layer(input_size, output_size, activation, layer_config))

    def forward_propagation(self, X):
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def backward_propagation(self, y_true, y_pred):
        delta = self.cost_function.derivative(y_true, y_pred)

        num_samples = y_true.shape[0]
        
        for layer in reversed(self.layers):
            delta = layer.backward(delta, num_samples)

    def train(self, X_train, y_train, X_test, y_test):
        if self.verbose:
            fancy_print(f"Starting training process for model {self.model_index}...\n", Fore.MAGENTA, Style.BRIGHT)

        epochs = self.config['epochs']
        batch_size = self.config['batch_size']

        for epoch in range(epochs):
            current_lr = self.lr_scheduler.get_lr_scheduler()(epoch)
            for layer in self.layers:
                layer.optimizer.learning_rate = current_lr
            
            if self.verbose:
                epoch_start_time = time.time()

            # Shuffle the data
            indices = np.arange(X_train.shape[0])
            np.random.shuffle(indices)
            X_train_shuffled = X_train[indices]
            y_train_shuffled = y_train[indices]

            # Set training mode for all layers
            for layer in self.layers:
                if hasattr(layer, 'use_batch_norm') and layer.use_batch_norm:
                    layer.batch_norm.training = True

            # Process batches
            num_batches = len(X_train_shuffled) // batch_size
            for i in range(num_batches):
                start = i * batch_size
                end = start + batch_size
                X_batch = X_train_shuffled[start:end]
                y_batch = y_train_shuffled[start:end]

                self.train_batch(X_batch, y_batch)

            # Process remaining samples
            if len(X_train_shuffled) % batch_size != 0:
                start = num_batches * batch_size
                X_batch = X_train_shuffled[start:]
                y_batch = y_train_shuffled[start:]

                self.train_batch(X_batch, y_batch)
            # Remove processing of remaining samples to speed up training
            
            # Compute loss on a subset of the training data
            subset_size = min(10000, len(X_train))
            indices = np.random.choice(len(X_train), subset_size, replace=False)
            X_subset = X_train[indices]
            y_subset = y_train[indices]
            y_pred_subset = self.predict(X_subset)
            loss = self.cost_function.loss(y_subset, y_pred_subset)
            
            # Clip the loss to avoid overflow
            loss = np.clip(loss, -1e10, 1e10)

            improvement = self.check_and_save_best_model(loss)

            if self.verbose:
                training_accuracy = self.evaluate(X_train, y_train)
                test_accuracy = self.evaluate(X_test, y_test)

                self.training_loss.append(loss)
                self.training_acc.append(training_accuracy)
                self.test_acc.append(test_accuracy)

                self.print_log(epoch, loss, epoch_start_time, training_accuracy, self.layers[0].optimizer.learning_rate, improvement, self.no_improvement_count, test_accuracy)

            if self.no_improvement_count >= self.patience:
                if self.verbose:
                    fancy_print("\nEarly stopping triggered", Fore.RED)
                break

        # Load the best model at the end of training
        if self.best_model is not None:
            self.load_best_model()

        return self

    def train_batch(self, X_batch, y_batch):
        y_pred = self.forward_propagation(X_batch)
        self.backward_propagation(y_batch, y_pred)

    def predict(self, X):
        for layer in self.layers:
            if hasattr(layer, 'use_batch_norm') and layer.use_batch_norm:
                layer.batch_norm.training = False
        return self.forward_propagation(X)

    def evaluate(self, X, y):
        y_pred = self.predict(X)
        if y.ndim == 2:
            accuracy = np.mean(np.argmax(y_pred, axis=1) == np.argmax(y, axis=1))
        else:
            accuracy = np.mean(np.argmax(y_pred, axis=1) == y)
        return accuracy
    
    def print_log(self, epoch, loss, start_time, training_accuracy, learning_rate, improvement, no_improvement_count, test_accuracy):
        fancy_print(f"Epoch {epoch+1}/{self.config['epochs']}", Fore.CYAN)
        if improvement:
            fancy_print("   Improvement: ✅", Fore.GREEN)
        else:
            fancy_print(f"   No improvement count: {no_improvement_count}", Fore.YELLOW)
        fancy_print(f"   Loss: {loss:.6f}", Fore.YELLOW)
        fancy_print(f"   Time: {time.time() - start_time:.2f}s", Fore.YELLOW)
        fancy_print(f"   Learning Rate: {learning_rate:.6f}", Fore.YELLOW)
        fancy_print(f"   Training Accuracy: {training_accuracy*100:.2f}%", Fore.GREEN)
        fancy_print(f"   Test Accuracy: {test_accuracy*100:.2f}%", Fore.GREEN)
        print("\n----------------------------------\n")

    def save_model(self, model_index):
        if self.verbose:
            fancy_print(f"Saving model_{model_index}.pkl...", Fore.YELLOW)
        model_path = 'src/models/'
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        if os.path.exists(model_path + f'model_{model_index}.pkl'):
            os.remove(model_path + f'model_{model_index}.pkl')
        with open(model_path + f'model_{model_index}.pkl', 'wb') as f:
            pickle.dump(self, f)
        if self.verbose:
            fancy_print("Model saved successfully", Fore.GREEN)

    def load_model(self, model_index):
        if self.verbose:
            fancy_print(f"Loading model_{model_index}.pkl...", Fore.YELLOW)
        model_path = 'src/models/'
        if not os.path.exists(model_path):
            if self.verbose:
                fancy_print("❌ Model not found", Fore.RED)
            raise FileNotFoundError("Model not found")
        
        with open(model_path + f'model_{model_index}.pkl', 'rb') as f:
            model = pickle.load(f)
        if self.verbose:
            fancy_print("Model loaded successfully", Fore.GREEN)
        return model



    def check_and_save_best_model(self, loss):
        if loss < self.best_loss:
            self.best_loss = loss
            self.best_model = self.get_model_state()
            self.no_improvement_count = 0
            return True
        else:
            self.no_improvement_count += 1
            return False

    def get_model_state(self):
        return [layer.get_weights() for layer in self.layers] + [layer.get_biases() for layer in self.layers]

    def set_model_state(self, state):
        for layer, weights in zip(self.layers, state):
            layer.set_weights(weights)
            

    def load_best_model(self):
        if self.best_model is not None:
            self.set_model_state(self.best_model)
            if self.verbose:
                fancy_print("Best model loaded successfully", Fore.GREEN)