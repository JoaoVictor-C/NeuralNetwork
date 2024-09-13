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
        X_train = X_train.astype('float32') / 255.0
        X_test = X_test.astype('float32') / 255.0
    else:
        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')

    # Reshape the data based on the dataset
    if config['data']['dataset'] in ['mnist', 'fashion_mnist']:
        X_train = X_train.reshape(-1, 28, 28, 1)
        X_test = X_test.reshape(-1, 28, 28, 1)
    elif config['data']['dataset'] in ['cifar10', 'cifar100']:
        X_train = X_train.reshape(-1, 32, 32, 3)
        X_test = X_test.reshape(-1, 32, 32, 3)

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

        # Create an infinite generator
        train_generator = datagen.flow(X_train, y_train, batch_size=config['data']['batch_size'])
        train_dataset = tf.data.Dataset.from_generator(
            lambda: train_generator,
            output_signature=(
                tf.TensorSpec(shape=(None,) + X_train.shape[1:], dtype=tf.float32),
                tf.TensorSpec(shape=(None,) + y_train.shape[1:], dtype=tf.float32)
            )
        ).repeat()
    else:
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(config['data']['batch_size']).repeat()

    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(config['data']['batch_size'])

    return train_dataset, test_dataset, X_train.shape[0], X_test.shape[0]
