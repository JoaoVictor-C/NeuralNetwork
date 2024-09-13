import os
import tensorflow as tf
import datetime

def create_callbacks(config):
    callbacks = []

    if config['training']['callbacks']['early_stopping']['enabled']:
        callbacks.append(create_early_stopping_callback(config))

    if config['training']['callbacks']['lr_scheduler']['enabled']:
        callbacks.append(create_lr_scheduler_callback(config))

    if config['training']['callbacks']['tensorboard']['enabled']:
        callbacks.append(create_tensorboard_callback())

    if config['training']['callbacks']['checkpoint']['enabled']:
        callbacks.append(create_checkpoint_callback())

    return callbacks

def create_checkpoint_callback():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')

    total_model_saved = len(os.listdir('checkpoints')) + 1
    
    folder_name = f'checkpoints/model_{total_model_saved}'
    os.makedirs(folder_name)

    return tf.keras.callbacks.ModelCheckpoint(
        filepath= folder_name + '/model.keras',
        monitor='accuracy',
        mode='max',
        save_best_only=True,
        initial_value_threshold=0.9
    )

def create_early_stopping_callback(config):
    return tf.keras.callbacks.EarlyStopping(
        monitor='loss',
        patience=config['training']['callbacks']['early_stopping']['patience'],
        restore_best_weights=True
    )

def create_lr_scheduler_callback(config):
    initial_lr = config['training']['callbacks']['lr_scheduler']['lr']
    lr_step_size = config['training']['callbacks']['lr_scheduler']['lr_step_size']
    lr_decay_rate = config['training']['callbacks']['lr_scheduler']['lr_decay_rate']
    
    def scheduler(epoch, lr):
        if config['training']['callbacks']['lr_scheduler']['type'] == 'exponential':
            if epoch % lr_step_size == 0 and epoch:
                return lr * lr_decay_rate
        elif config['training']['callbacks']['lr_scheduler']['type'] == 'step':
            if epoch % lr_step_size == 0 and epoch:
                return lr * lr_decay_rate
        elif config['training']['callbacks']['lr_scheduler']['type'] == 'cosine':
            if epoch % lr_step_size == 0 and epoch:
                return lr * lr_decay_rate
        elif config['training']['callbacks']['lr_scheduler']['type'] == 'polynomial':
            if epoch % lr_step_size == 0 and epoch:
                return lr * lr_decay_rate
        return lr

    return tf.keras.callbacks.LearningRateScheduler(scheduler)

def create_tensorboard_callback():
    log_dir = f"logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    return tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        update_freq='epoch'
    )
