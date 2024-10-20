import numpy as np
import json
import pandas as pd

from data_preprocessing import preprocess_data
from knn_model import KNN

def roc_auc_score(y_true, y_scores):
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)

    pos_scores = y_scores[y_true == 1]
    neg_scores = y_scores[y_true == 0]

    n_pos = len(pos_scores)
    n_neg = len(neg_scores)

    if n_pos == 0 or n_neg == 0:
        return 0.0

    all_scores = np.concatenate([pos_scores, neg_scores])
    sorted_indices = np.argsort(all_scores)
    sorted_labels = np.concatenate([y_true[y_true == 1], y_true[y_true == 0]])[sorted_indices]
    ranks = np.arange(1, len(all_scores) + 1)
    rank_sum = np.sum(ranks[sorted_labels == 1])

    auc = (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return auc

def train_validation_split(X, y, test_size=0.2, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    split_idx = int(len(X) * (1 - test_size))
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    X_train = X[train_indices]
    X_val = X[val_indices]
    y_train = y[train_indices]
    y_val = y[val_indices]
    return X_train, X_val, y_train, y_val

def save_best_params(best_params, filename=None):
    # Generate a default filename if none is provided
    if filename is None:
        from datetime import datetime
        current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"best_params_{current_time}.log"
    
    # Convert any numpy arrays (like feature_weights) to lists for JSON serialization
    serializable_params = {}
    for key, value in best_params.items():
        if isinstance(value, np.ndarray):  # Check if the value is a numpy array
            serializable_params[key] = value.tolist()  # Convert it to a list
        else:
            serializable_params[key] = value  # Keep it as is if not a numpy array
    
    with open(filename, 'w') as f:
        json.dump(serializable_params, f)  # Save the serializable params as JSON
    print(f"Best parameters saved to {filename}")

def load_best_params(filename=None):
    if filename is None:
        filename = input("Please enter the filename to load best parameters from: ")

    try:
        with open(filename, 'r') as f:
            loaded_params = json.load(f)
        
        # Convert any lists that were numpy arrays back to numpy arrays
        best_params = {}
        for key, value in loaded_params.items():
            if isinstance(value, list) and isinstance(value[0], float):  # Check if the value was originally a numpy array
                best_params[key] = np.array(value)  # Convert it back to a numpy array
            else:
                best_params[key] = value  # Keep it as is if not a list
        
        return best_params
    except FileNotFoundError:
        return None
    
def reconstruct_submission(train_path, test_path, best_params):
    # Preprocess the entire training and test data with the best parameters
    X_train, y_train, X_test, test_ids = preprocess_data(
        train_path, test_path,
        scaling_method=best_params['scaling_method'],
        feature_transformations=best_params.get('feature_transformations', [])
    )
    feature_names = [f'feature_{i}' for i in range(X_train.shape[1])]

    # Handle feature weights
    if best_params.get('feature_weights') is not None:
        feature_weights = best_params['feature_weights']
    else:
        feature_weights = np.ones(len(feature_names))

    # Train final model with best hyperparameters
    final_knn = KNN(k=best_params['k'],
                    distance_metric=best_params['distance_metric'],
                    weighted=best_params['weighted'],
                    p=best_params.get('p', 2.0),
                    regularization=best_params.get('regularization', 1e-5),
                    feature_weights=feature_weights)
    final_knn.fit(X_train, y_train)
    print("Final model training completed.")

    # Predict on test set
    print("Making predictions on the test set...")
    test_predictions = final_knn.predict(X_test)
    print("Predictions completed.")

    # Prepare submission
    submission = pd.DataFrame({
        'id': test_ids,
        'Exited': test_predictions
    })
    submission.to_csv('submissions.csv', index=False)
    print("Submission file 'submissions.csv' created successfully.")
