import sys
import argparse

from data_preprocessing import preprocess_data
from hyperparameter_optimization import simulated_annealing
from utils import load_best_params, save_best_params, reconstruct_submission
from nn_transform import nn_feature_transform  # Import the neural network transformer

def main():
    parser = argparse.ArgumentParser(description='KNN Churn Prediction with Neural Network Feature Transformation')
    parser.add_argument('--train_path', type=str, default='train.csv', help='Path to training data')
    parser.add_argument('--test_path', type=str, default='test.csv', help='Path to test data')
    parser.add_argument('--init_params_path', type=str, help='Path to initial parameters file')
    parser.add_argument('--reconstruct', action='store_true', help='Reconstruct submissions.csv from best_params.log')
    args = parser.parse_args()

    if args.reconstruct:
        best_params = load_best_params()
        if best_params is None:
            print("No best parameters found in best_params.log")
            sys.exit(1)
        reconstruct_submission(args.train_path, args.test_path, best_params)
        print("Submission file 'submissions.csv' created successfully from best parameters.")
        sys.exit(0)

    if args.init_params_path:
        initial_params = load_best_params(args.init_params_path)
        if initial_params is None:
            print(f"Could not load parameters from {args.init_params_path}")
            sys.exit(1)
        print("Loaded initial parameters from file.")
    else:
        initial_params = {
            'k': 5,
            'distance_metric': 'euclidean',
            'weighted': False,
            'p': 2.0,
            'scaling_method': 'minmax',
            'regularization': 1e-5,
            'feature_transformations': [],
            'feature_weights': None
        }

    param_bounds = {
        'k': (1, 51),
        'distance_metric': ['euclidean', 'manhattan', 'minkowski', 'cosine', 'mahalanobis'],
        'weighted': [True, False],
        'p': (1.0, 5.0),
        'scaling_method': ['minmax', 'standard', 'none'],
        'regularization': (1e-5, 1e-2),
        'feature_transformations': ['log', 'sqrt', 'square', 'interaction'],
        'feature_weights': (0.1, 10.0)
    }

    # Preprocess data to get initial features
    X, y, X_test, test_ids, feature_names = preprocess_data(
        args.train_path, args.test_path, scaling_method='none', return_feature_names=True)
    print("Data preprocessing completed.")

    # First run of KNN hyperparameter optimization
    best_params = simulated_annealing(
        X, y, initial_params, param_bounds, feature_names,
        max_evals=50, initial_temp=100.0, cooling_rate=0.95, random_state=43,
        num_initializations=5, min_sample_size=500, max_sample_size=5000, patience=10
    )
    print(f"First KNN optimization completed with AUC: {best_params}")

    # Neural network feature transformation
    X_train_transformed, X_val_transformed = nn_feature_transform(X, X_test, hidden_size=64, epochs=100)
    print("Neural network transformation completed.")

    # Re-run KNN hyperparameter optimization on transformed features
    best_params = simulated_annealing(
        X_train_transformed, y, initial_params, param_bounds, feature_names,
        max_evals=50, initial_temp=100.0, cooling_rate=0.95, random_state=43,
        num_initializations=5, min_sample_size=500, max_sample_size=5000, patience=10
    )
    print(f"Final KNN optimization completed with AUC: {best_params}")

    # Save best parameters to log file
    save_best_params(best_params)

    # Reconstruct submissions.csv with best parameters
    reconstruct_submission(args.train_path, args.test_path, best_params)
    print("Submission file 'submissions.csv' created successfully with best parameters.")

if __name__ == "__main__":
    main()
