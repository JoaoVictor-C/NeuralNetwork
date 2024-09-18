import tensorflow as tf
from keras import layers, models, mixed_precision

def create_actor(config, compile=True):
    model = tf.keras.Sequential()
    for layer in config['model']['actor']['layers']:
        if 'dense' in layer:
            model.add(layers.Dense(layer['dense']['units'], activation=layer['dense']['activation']))
    

    # Scale output to action bounds if necessary
    if config['training']['optimizer_actor'] == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=config['training']['learning_rate_actor'])
    elif config['training']['optimizer_actor'] == 'rmsprop':
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=config['training']['learning_rate_actor'])
    else:
        raise ValueError(f"Invalid optimizer: {config['training']['optimizer_actor']}")
    
    if compile:
        model.compile(
            optimizer=optimizer,
            loss=config['training']['loss'],
            metrics=config['training']['metrics']
        )
    
    model.build(input_shape=(None, config['data']['input_shape'][0]))
    return model