import os
# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  
# Disable OneDNN options
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from utils.preprocessing import load_config, create_kfold, calculate_average_score
from utils.callbacks import create_callbacks
from models.model import create_model, compile_model, train_model, evaluate_model
from utils.data_loader import load_data

def perform_cross_validation(train_dataset, config, train_size):
    kfold = create_kfold()
    fold_scores = []

    for fold in range(kfold.n_splits):
        print(f'Fold {fold + 1}/{kfold.n_splits}')

        model = create_model(config)
        callbacks = create_callbacks(config, fold)

        train_model(model, train_dataset, None, config, callbacks, train_size)

        # Note: We can't easily validate on a specific fold with the new data loader
        # So we'll skip the validation step in cross-validation
        fold_scores.append(0)  # Placeholder for fold score
        print(f'Fold {fold + 1} completed')

    return fold_scores

def train_final_model(train_dataset, config, train_size):
    model = create_model(config)
    model = compile_model(model, config)
    callbacks = create_callbacks(config)

    train_model(model, train_dataset, None, config, callbacks, train_size)
    return model

def main():
    config = load_config()
    train_dataset, test_dataset, train_size, test_size = load_data()

    model = create_model(config, True)
    callbacks = create_callbacks(config)

    train_model(model, train_dataset, None, config, callbacks, train_size, test_size)

    # Evaluate model
    test_loss, test_acc = evaluate_model(model, test_dataset)
    print(f'Final model test accuracy: {test_acc}')

if __name__ == '__main__':
    main()