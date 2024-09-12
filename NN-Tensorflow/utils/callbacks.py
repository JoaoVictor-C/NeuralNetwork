import os
import tensorflow as tf
import datetime

def create_callbacks(config, fold=None):
    callbacks = []

    if config['training']['callbacks']['checkpoint']['enabled']:
        callbacks.append(create_checkpoint_callback(config, fold))

    if config['training']['callbacks']['early_stopping']['enabled']:
        callbacks.append(create_early_stopping_callback(config))

    if config['training']['callbacks']['lr_scheduler']['enabled']:
        callbacks.append(create_lr_scheduler_callback(config))

    if config['training']['callbacks']['tensorboard']['enabled']:
        callbacks.append(create_tensorboard_callback(fold))

    return callbacks

def create_checkpoint_callback(config, fold):
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')

    total_model_saved = len(os.listdir('checkpoints')) + 1
    
    folder_name = f'checkpoints/model_{total_model_saved}'
    if fold is not None:
        folder_name += f'_fold_{fold + 1}'
    os.makedirs(folder_name)

    save_freq = 'epoch'
    
    return tf.keras.callbacks.ModelCheckpoint(
        filepath=f'{folder_name}/cp-{{epoch:04d}}.weights.h5', 
        save_weights_only=True,
        verbose=0,
        save_freq=save_freq
    )

def create_early_stopping_callback(config):
    return tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=config['training']['callbacks']['early_stopping']['patience'],
        restore_best_weights=True
    )

def create_lr_scheduler_callback(config):
    def lr_schedule(epoch, lr):
        if epoch < config['training']['epochs']:
            return config['training']['callbacks']['lr_scheduler']['lr'] * \
                   config['training']['callbacks']['lr_scheduler']['lr_decay_rate'] ** \
                   (epoch // config['training']['callbacks']['lr_scheduler']['lr_step_size'])
        return lr

    return tf.keras.callbacks.LearningRateScheduler(
        schedule=lr_schedule,
        verbose=0
    )

def create_tensorboard_callback(fold):
    log_dir = f"logs/fit/{'fold_' + str(fold + 1) + '/' if fold is not None else ''}" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    return tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        update_freq='epoch'
    )
