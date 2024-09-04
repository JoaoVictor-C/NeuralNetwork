import os
import requests
import gzip
import numpy as np

train_images_url = "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz"
train_labels_url = "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz"
test_images_url = "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz"
test_labels_url = "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz"

path = "data/"

if not os.path.exists(path):
    os.makedirs(path)

def download_file(url, output_path):
    response = requests.get(url)
    with open(output_path, 'wb') as f:
        f.write(response.content)

def extract_images(file_path):
    with gzip.open(file_path, 'rb') as f:
        data = f.read()
        return np.frombuffer(data, dtype=np.uint8, offset=16).reshape(-1, 28, 28)

def extract_labels(file_path):
    with gzip.open(file_path, 'rb') as f:
        data = f.read()
        return np.frombuffer(data, dtype=np.uint8, offset=8)

def download_and_extract_mnist():
    download_file(train_images_url, path + 'train-images.gz')
    download_file(train_labels_url, path + 'train-labels.gz')
    download_file(test_images_url, path + 't10k-images.gz')
    download_file(test_labels_url, path + 't10k-labels.gz')
    
    train_images = extract_images(path + 'train-images.gz')
    train_labels = extract_labels(path + 'train-labels.gz')
    test_images = extract_images(path + 't10k-images.gz')
    test_labels = extract_labels(path + 't10k-labels.gz')

    return train_images, train_labels, test_images, test_labels

train_images, train_labels, test_images, test_labels = download_and_extract_mnist()

print(train_images.shape)
print(train_labels.shape)
print(test_images.shape)
print(test_labels.shape)
