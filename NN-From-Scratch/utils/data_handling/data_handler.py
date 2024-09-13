import numpy as np
import random
import os
from ..fancy_print import fancy_print
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from colorama import init, Fore, Style
import matplotlib.pyplot as plt
import time
from scipy.ndimage import rotate, shift, gaussian_filter, map_coordinates

init(autoreset=True)  # Initialize colorama

class DataHandler:
    def __init__(self, parameters_dataset):
        self.images_path = parameters_dataset['images_path']
        self.labels_path = parameters_dataset['labels_path']
        self.parameters_dataset = parameters_dataset

    def load_data(self, train_test_split_ratio, augmentation=True, num_augmented=2, show_samples=False):
        start_time = time.time()
        image_size = self.parameters_dataset['input_size']
        num_samples = self.parameters_dataset['num_samples']
        images = self._read_binary_file(self.images_path, num_samples * image_size, skip_bytes=16)
        labels = self._read_binary_file(self.labels_path, num_samples, skip_bytes=8)
        images = images.reshape((num_samples, image_size))
        labels = np.eye(self.parameters_dataset['output_size'])[labels.astype(int)]
        X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=1-train_test_split_ratio, random_state=42)
        
        # Apply data augmentation to the training set
        if augmentation:
            X_train, y_train = self.augment_data(X_train, y_train, num_augmented=num_augmented)
        
        if show_samples:    
            plt.figure(figsize=(10, 5))
            rng = random.Random(42)
            indices = rng.sample(range(X_train.shape[0]), 25)
            for i, index in enumerate(indices):
                plt.subplot(5, 5, i + 1)
                plt.imshow(X_train[i].reshape(28, 28), cmap='binary')
                plt.title(f"Label: {np.argmax(y_train[i])}")
                plt.axis('off')
            plt.show()
        
        elapsed_time = time.time() - start_time
        fancy_print(f"Data loading and preprocessing completed in {elapsed_time:.2f} seconds!", Fore.GREEN, Style.BRIGHT)
        fancy_print(f"Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}", Fore.YELLOW)
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

    def augment_data(self, X, y, num_augmented=2):
        fancy_print("Applying data augmentation...", Fore.CYAN)
        X_augmented = []
        y_augmented = []

        for image, label in zip(X, y):
            X_augmented.append(image)
            y_augmented.append(label)
            
            for _ in range(num_augmented):
                aug_image = self._apply_augmentation(image.reshape(28, 28))
                X_augmented.append(aug_image.flatten())
                y_augmented.append(label)

        X_augmented = np.array(X_augmented)
        y_augmented = np.array(y_augmented)
        
        fancy_print(f"Data augmentation completed. New dataset size: {len(X_augmented)}", Fore.GREEN)
        return X_augmented, y_augmented

    def _apply_augmentation(self, image):
        # Random rotation
        angle = np.random.uniform(-15, 15)
        image = rotate(image, angle, reshape=False, mode='nearest')

        # Random shift
        shift_x = np.random.randint(-4, 6)
        shift_y = np.random.randint(-4, 6)
        image = shift(image, [shift_y, shift_x], mode='nearest')

        # Random noise
        noise = np.random.normal(0, 0.1, image.shape)
        image = np.clip(image + noise, 0, 1)

        # Random elastic distortion
        alpha = np.random.uniform(6, 12)
        sigma = np.random.uniform(3, 5)
        random_state = np.random.RandomState(None)
        shape = image.shape
        dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
        indices = np.reshape(x+dx, (-1, 1)), np.reshape(y+dy, (-1, 1))
        image = map_coordinates(image, indices, order=1).reshape(shape)

        return image