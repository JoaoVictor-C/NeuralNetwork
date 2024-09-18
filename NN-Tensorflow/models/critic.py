import tensorflow as tf
from keras import layers, mixed_precision
from keras.layers import Input
from keras.models import Model
from keras.mixed_precision import LossScaleOptimizer

def create_critic(config):
    # Define two inputs: state and action
    state_input = Input(shape=config['data']['input_shape'][0], name='state_input')
    action_input = Input(shape=(config['training']['action_size'],), name='action_input')

    # Process state input
    x = state_input
    for layer in config['model']['critic']['layers']:
        if 'dense' in layer:
            x = layers.Dense(layer['dense']['units'], activation=layer['dense']['activation'])(x)
    # Optionally add dropout or batch normalization for efficiency

    # Process action input
    a = action_input
    for layer in config['model']['critic']['action_layers']:
        if 'dense' in layer:
            a = layers.Dense(layer['dense']['units'], activation=layer['dense']['activation'])(a)

    # Combine state and action pathways
    combined = layers.Concatenate()([x, a])

    # Add final layers if any
    for layer in config['model']['critic']['post_concatenate_layers']:
        if 'dense' in layer:
            combined = layers.Dense(layer['dense']['units'], activation=layer['dense']['activation'])(combined)

    # Output layer
    output = layers.Dense(1, activation='linear', name='q_value')(combined)

    # Define the model with state and action as inputs
    model = Model(inputs=[state_input, action_input], outputs=output, name='CriticModel')

    if config['training']['optimizer_critic'] == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=config['training']['learning_rate_critic'])
    elif config['training']['optimizer_critic'] == 'rmsprop':
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=config['training']['learning_rate_critic'])
    else:
        raise ValueError(f"Invalid optimizer: {config['training']['optimizer_critic']}")

    optimizer = mixed_precision.LossScaleOptimizer(optimizer)

    model.compile(
        optimizer=optimizer,
        loss=config['training']['loss'],
        metrics=config['training']['metrics']
    )

    model.build(input_shape=(None, config['data']['input_shape'][0], config['training']['action_size']))
    return model