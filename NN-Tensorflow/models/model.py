import tensorflow as tf
from utils.preprocessing import load_config

import tensorflow as tf
from typing import Dict, Any

def create_layer(layer_config: Dict[str, Any], regularizer: tf.keras.regularizers.Regularizer) -> tf.keras.layers.Layer:
    """Create a single layer based on the configuration."""
    layer_type = list(layer_config.keys())[0]
    layer_params = layer_config[layer_type]

    if layer_type == 'conv2d':
        return tf.keras.layers.Conv2D(
            filters=layer_params['filters'],
            kernel_size=layer_params['kernel_size'],
            activation=layer_params['activation'],
            kernel_regularizer=regularizer,
            input_shape=layer_params.get('input_shape')
        )
    elif layer_type == 'maxpooling2d':
        return tf.keras.layers.MaxPooling2D(
            pool_size=layer_params['pool_size']
        )
    elif layer_type == 'flatten':
        return tf.keras.layers.Flatten()
    elif layer_type == 'dense':
        return tf.keras.layers.Dense(
            units=layer_params['units'],
            activation=layer_params['activation'],
            kernel_regularizer=regularizer
        )
    elif layer_type == 'dropout':
        return tf.keras.layers.Dropout(rate=layer_params['rate'])
    elif layer_type == 'batchnormalization':
        return tf.keras.layers.BatchNormalization()
    elif layer_type == 'reshape':
        return tf.keras.layers.Reshape(target_shape=layer_params['target_shape'])
    elif layer_type == 'permute':
        return tf.keras.layers.Permute(dims=layer_params['dims'])
    elif layer_type == 'resizing':
        return tf.keras.layers.Resizing(height=layer_params['height'], width=layer_params['width'])
    elif layer_type == 'randomzoom':
        return tf.keras.layers.RandomZoom(height_factor=layer_params['height_factor'], width_factor=layer_params.get('width_factor'))
    elif layer_type == 'randomtranslation':
        return tf.keras.layers.RandomTranslation(height_factor=layer_params['height_factor'], width_factor=layer_params['width_factor'])
    elif layer_type == 'randomrotation':
        return tf.keras.layers.RandomRotation(factor=layer_params['factor'])
    elif layer_type == 'randomflip':
        return tf.keras.layers.RandomFlip(mode=layer_params['mode'])
    # Add more layer types as needed
    else:
        raise ValueError(f"Unsupported layer type: {layer_type}")

def create_model(config: Dict[str, Any], compile: bool = True) -> tf.keras.Model:
    """Create and return a TensorFlow model based on the configuration."""
    # Get regularization parameters
    l1 = config['training'].get('l1', 0.0)
    l2 = config['training'].get('l2', 0.0)
    regularizer = tf.keras.regularizers.l1_l2(l1=l1, l2=l2)

    # Create the model
    model = tf.keras.Sequential()
    
    # Add layers from config
    for layer_config in config['model']['layers']:
        layer = create_layer(layer_config, regularizer)
        model.add(layer)

    model.summary()

    if compile:
        compile_model(model, config)

    return model



def compile_model(model, config):
    model.compile(
        optimizer=config['training']['optimizer'],
        loss=config['training']['loss'],
        metrics=config['training']['metrics']
    )
    return model

def train_model(model, X_train, y_train, X_val, y_val, datagen, config, callbacks):
    return model.fit(
        datagen.flow(X_train, y_train, batch_size=config['data']['batch_size']),
        steps_per_epoch=len(X_train) // config['data']['batch_size'],
        epochs=config['training']['epochs'], 
        validation_data=(X_val, y_val),
        callbacks=callbacks
    )

def evaluate_model(model, X, y):
    return model.evaluate(X, y)