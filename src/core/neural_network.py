from ..activation import Activation
from ..cost import Cost
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
    def __init__(self, dataset):
        fancy_print("Initializing Neural Network...", Fore.CYAN, Style.BRIGHT)
        self.load_config(dataset)
        self.activation = Activation()
        self.layers = []
        self.create_layers()
        
        self.cost = Cost()
        self.cost_function = self.cost.get_cost_function(self.config['cost_function'])

        self.training_acc = []
        self.training_loss = []

        self.lr_scheduler = self.get_lr_scheduler()
        fancy_print("Neural Network initialized successfully!", Fore.GREEN)

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
            else:
                # Input and hidden layers
                activation = self.activation.get_activation_function(self.config['activation_function'])
            
            self.layers.append(Layer(input_size, output_size, activation, self.config))

    def forward_propagation(self, X):
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def backward_propagation(self, y_true, y_pred):
        delta = self.cost_function.derivative(y_true, y_pred)
        
        for layer in reversed(self.layers):
            delta = layer.backward(delta, self.base_config['num_samples'])

    def train(self, input, label):
        fancy_print("Starting training process...\n", Fore.MAGENTA, Style.BRIGHT)
        epochs = self.config['epochs']
        batch_size = self.config['batch_size']
        start_time = time.time()
        patience = self.config['patience']
        best_loss = 0
        no_improvement_count = 0

        for epoch in range(epochs):
            current_lr = self.lr_scheduler(epoch)
            for layer in self.layers:
                layer.optimizer.learning_rate = current_lr
            
            epoch_start_time = time.time()

            # Shuffle the data
            indices = np.arange(input.shape[0])
            np.random.shuffle(indices)
            input_shuffled = input[indices]
            label_shuffled = label[indices]

            # Process batches
            num_batches = len(input) // batch_size
            for i in range(num_batches):
                start = i * batch_size
                end = start + batch_size
                X_batch = input_shuffled[start:end]
                y_batch = label_shuffled[start:end]

                self.train_batch(X_batch, y_batch)
            
            # Process remaining samples
            if len(input) % batch_size != 0:
                start = num_batches * batch_size
                X_batch = input_shuffled[start:]
                y_batch = label_shuffled[start:]

                self.train_batch(X_batch, y_batch)
            
            # Compute loss on full dataset
            y_pred_full = self.predict(input)
            loss = self.cost_function.loss(label, y_pred_full)
            
            # Clip the loss to avoid overflow
            loss = np.clip(loss, -1e10, 1e10)

            training_accuracy = self.evaluate(input_shuffled, label_shuffled)

            self.training_loss.append(loss)
            self.training_acc.append(training_accuracy)
            
            self.print_log(epoch, loss, epoch_start_time, training_accuracy, self.layers[0].optimizer.learning_rate)

            if np.abs(best_loss - loss) < 1e-4:
                no_improvement_count += 1
                fancy_print(f"No improvement count: {no_improvement_count}", Fore.YELLOW)
            else:
                best_loss = loss
                no_improvement_count = 0

            if no_improvement_count >= patience:
                fancy_print("Early stopping triggered", Fore.RED)
                break

        self.final_training_log(start_time)

    def train_batch(self, X_batch, y_batch):
        y_pred = self.forward_propagation(X_batch)
        self.backward_propagation(y_batch, y_pred)

    def predict(self, X):
        return self.forward_propagation(X)

    def evaluate(self, X, y):
        y_pred = self.predict(X)
        accuracy = np.mean(np.argmax(y_pred, axis=1) == np.argmax(y, axis=1))
        return accuracy
    
    def print_log(self, epoch, loss, start_time, training_accuracy, learning_rate):
        fancy_print(f"Epoch {epoch+1}/{self.config['epochs']}", Fore.CYAN)
        fancy_print(f"   Loss: {loss:.6f}", Fore.YELLOW)
        fancy_print(f"   Time: {time.time() - start_time:.2f}s", Fore.YELLOW)
        fancy_print(f"   Learning Rate: {learning_rate:.6f}", Fore.YELLOW)
        fancy_print(f"   Training Accuracy: {training_accuracy*100:.2f}%", Fore.GREEN)
        print("\n----------------------------------\n")

    def final_training_log(self, start_time):
        fancy_print("Training completed!", Fore.GREEN, Style.BRIGHT)
        fancy_print(f"Training Accuracy average: {np.mean(self.training_acc)*100:.2f}%", Fore.CYAN)
        fancy_print(f"Training Loss average: {np.mean(self.training_loss):.4f}", Fore.CYAN)
        fancy_print(f"Total training time: {time.time() - start_time:.2f}s", Fore.CYAN)

    def save_model(self):
        fancy_print("Saving model...", Fore.YELLOW)
        model_path = 'src/models/'
        total_models = len(os.listdir(model_path))
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        with open(model_path + f'model_{total_models}.pkl', 'wb') as f:
            pickle.dump(self, f)
        fancy_print(f"✅ Model saved as 'model_{total_models}.pkl'", Fore.GREEN)

    def load_model(self, model_index):
        fancy_print(f"Loading model_{model_index}.pkl...", Fore.YELLOW)
        model_path = 'src/models/'
        if not os.path.exists(model_path):
            fancy_print("❌ Model not found", Fore.RED)
            raise FileNotFoundError("Model not found")
        with open(model_path + f'model_{model_index}.pkl', 'rb') as f:
            model = pickle.load(f)
        fancy_print("Model loaded successfully", Fore.GREEN)
        return model

    def get_lr_scheduler(self):
        scheduler_type = self.config.get('lr_scheduler', 'constant')
        if scheduler_type == 'step':
            return lambda epoch: self.config['learning_rate'] * (self.config['lr_decay'] ** (epoch // self.config['lr_step']))
        elif scheduler_type == 'exponential':
            return lambda epoch: self.config['learning_rate'] * (self.config['lr_decay'] ** epoch)
        else:
            return lambda epoch: self.config['learning_rate']