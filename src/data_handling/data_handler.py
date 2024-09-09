import numpy as np
import os
from ..utils import fancy_print
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from colorama import init, Fore, Style
import time

init(autoreset=True)  # Initialize colorama

class DataHandler:
    def __init__(self, parameters_dataset):
        self.images_path = parameters_dataset['images_path']
        self.labels_path = parameters_dataset['labels_path']
        self.parameters_dataset = parameters_dataset

    def load_data(self, train_test_split_ratio):
        start_time = time.time()

        image_size = self.parameters_dataset['input_size']
        num_samples = self.parameters_dataset['num_samples']

        images = self._read_binary_file(self.images_path, num_samples * image_size, skip_bytes=16)
        labels = self._read_binary_file(self.labels_path, num_samples, skip_bytes=8)

        images = images.reshape((num_samples, image_size))
        labels = np.eye(self.parameters_dataset['output_size'])[labels.astype(int)]

        X_test, X_train, y_test, y_train = train_test_split(images, labels, test_size=train_test_split_ratio, random_state=42)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        elapsed_time = time.time() - start_time
        fancy_print(f"Data loading and preprocessing completed in {elapsed_time:.2f} seconds!", Fore.GREEN, Style.BRIGHT)
        fancy_print(f"Training samples: {X_train.shape[0]}", Fore.GREEN)
        fancy_print(f"Test samples: {X_test.shape[0]}", Fore.GREEN)

        return X_train, y_train, X_test, y_test

    def _read_binary_file(self, file_path, num_bytes, skip_bytes=0):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        fancy_print(f"Reading file: {file_path}", Fore.CYAN)
        with open(file_path, 'rb') as f:
            f.seek(skip_bytes)  # Skip the specified number of bytes
            data = np.frombuffer(f.read(num_bytes), dtype=np.uint8)
        return data

    def get_batch(self, batch_size, start_index=0):
        fancy_print(f"Getting batch of size {batch_size} starting from index {start_index}", Fore.YELLOW)
        images, labels = self.load_data(self.parameters_dataset['train_test_split_ratio'])
        end_index = start_index + batch_size
        fancy_print(f"Batch retrieved successfully", Fore.GREEN)
        return images[start_index:end_index], labels[start_index:end_index]