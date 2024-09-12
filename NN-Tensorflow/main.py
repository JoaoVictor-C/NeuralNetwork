import os
# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  
# Disable OneDNN options
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from utils.preprocessing import load_config, create_kfold, calculate_average_score
from utils.callbacks import create_callbacks
from models.model import create_model, compile_model, train_model, evaluate_model
from utils.data_loader import load_data

def perform_cross_validation(X_train, y_train, config, datagen):
    kfold = create_kfold()
    fold_scores = []

    for fold, (train_indices, val_indices) in enumerate(kfold.split(X_train)):
        print(f'Fold {fold + 1}/{kfold.n_splits}')

        X_train_fold, X_val_fold = X_train[train_indices], X_train[val_indices]
        y_train_fold, y_val_fold = y_train[train_indices], y_train[val_indices]

        model = create_model(config)
        model = compile_model(model, config)
        callbacks = create_callbacks(config, fold)

        train_model(model, X_train_fold, y_train_fold, X_val_fold, y_val_fold, datagen, config, callbacks)

        _, val_acc = evaluate_model(model, X_val_fold, y_val_fold)
        fold_scores.append(val_acc)
        print(f'Fold {fold + 1} validation accuracy: {val_acc}')

    return fold_scores

def train_final_model(X_train, y_train, config, datagen):
    model = create_model(config)
    model = compile_model(model, config)
    callbacks = create_callbacks(config)

    train_model(model, X_train, y_train, None, None, datagen, config, callbacks)
    return model

def main():
    config = load_config()
    X_train, y_train, X_test, y_test, datagen = load_data()

    # Perform cross-validation
    fold_scores = perform_cross_validation(X_train, y_train, config, datagen)
    print(f'Average validation accuracy: {calculate_average_score(fold_scores)}')

    # Train final model
    final_model = train_final_model(X_train, y_train, config, datagen)

    # Evaluate final model
    _, test_acc = evaluate_model(final_model, X_test, y_test)
    print(f'Final model test accuracy: {test_acc}')

if __name__ == '__main__':
    main()