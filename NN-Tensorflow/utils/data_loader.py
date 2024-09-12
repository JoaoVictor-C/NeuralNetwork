import tensorflow as tf
import tensorflow as tf
from utils.preprocessing import load_config

def load_data():
    config = load_config()
    if config['data']['dataset'] == 'mnist':
        (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    elif config['data']['dataset'] == 'fashion_mnist':
        (X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    elif config['data']['dataset'] == 'cifar10':
        (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
    elif config['data']['dataset'] == 'cifar100':
        (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar100.load_data()
    else:
        raise ValueError(f"Dataset {config['data']['dataset']} not found")

    if config['data']['normalize']:
        X_train = X_train.reshape(-1, 28, 28, 1) / 255.0
        X_test = X_test.reshape(-1, 28, 28, 1) / 255.0
    else:
        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')

    # Create data generator for augmentation
    if config['data']['augmentation']['enabled']:
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=config['data']['augmentation']['rotation_range'],
            width_shift_range=config['data']['augmentation']['width_shift_range'],
            height_shift_range=config['data']['augmentation']['height_shift_range'],
            zoom_range=config['data']['augmentation']['zoom_range'],
            horizontal_flip=config['data']['augmentation']['horizontal_flip'],
            fill_mode=config['data']['augmentation']['fill_mode']
        )

        # Fit the data generator on the training data
        datagen.fit(X_train)
    else:
        datagen = None

    return X_train, y_train, X_test, y_test, datagen
