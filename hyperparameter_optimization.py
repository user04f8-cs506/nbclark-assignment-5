import numpy as np
import pandas as pd
import random

from data_preprocessing import preprocess_data
from knn_model import KNN
from utils import roc_auc_score, train_validation_split

def evaluate_params(X_train_full, y_train_full, X_val_full, y_val_full, params, feature_names):
    # Preprocess data with the specified scaling method and feature transformations
    X_combined = np.vstack((X_train_full, X_val_full))
    y_combined = np.concatenate((y_train_full, y_val_full))

    # Create DataFrame for combined features
    combined_features = pd.DataFrame(X_combined, columns=feature_names)

    # Apply feature transformations
    numerical_features = combined_features.columns.tolist()
    if params.get('feature_transformations'):
        for transformation in params['feature_transformations']:
            if transformation == 'log':
                for feature in numerical_features:
                    combined_features[feature + '_log'] = np.log1p(combined_features[feature])
            elif transformation == 'sqrt':
                for feature in numerical_features:
                    combined_features[feature + '_sqrt'] = np.sqrt(combined_features[feature])
            elif transformation == 'square':
                for feature in numerical_features:
                    combined_features[feature + '_square'] = combined_features[feature] ** 2
            elif transformation == 'interaction':
                for i in range(len(numerical_features)):
                    for j in range(i+1, len(numerical_features)):
                        f1 = numerical_features[i]
                        f2 = numerical_features[j]
                        combined_features[f'{f1}_x_{f2}'] = combined_features[f1] * combined_features[f2]

    # Update feature names after transformations
    feature_names = combined_features.columns.tolist()

    # Scaling
    if params['scaling_method'] == 'minmax':
        feature_min = combined_features.min()
        feature_max = combined_features.max()
        combined_features = (combined_features - feature_min) / (feature_max - feature_min)
    elif params['scaling_method'] == 'standard':
        feature_mean = combined_features.mean()
        feature_std = combined_features.std()
        combined_features = (combined_features - feature_mean) / feature_std
    elif params['scaling_method'] == 'none':
        pass

    # Split back into train and val data
    X_train = combined_features.iloc[:len(X_train_full), :].values
    X_val = combined_features.iloc[len(X_train_full):, :].values
    y_train = y_train_full
    y_val = y_val_full

    # Handle feature weights
    if params.get('feature_weights') is not None:
        feature_weights = params['feature_weights']
    else:
        feature_weights = np.ones(len(feature_names))

    knn = KNN(k=params['k'],
              distance_metric=params['distance_metric'],
              weighted=params['weighted'],
              p=params.get('p', 2.0),
              regularization=params.get('regularization', 1e-5),
              feature_weights=feature_weights)
    knn.fit(X_train, y_train)
    y_scores = knn.predict(X_val)
    auc = roc_auc_score(y_val, y_scores)
    print(f'(AUC = {auc})')
    return auc

def perturb_params(params, param_bounds, feature_names, step_size_factor=1.0):
    new_params = params.copy()
    hyperparameters = list(params.keys())
    hyper_to_perturb = random.choice(hyperparameters)

    # Adjust the step sizes based on step_size_factor
    if hyper_to_perturb == 'k':
        max_delta_k = max(1, int((param_bounds['k'][1] - param_bounds['k'][0]) * 0.1 * step_size_factor))
        delta = random.randint(-max_delta_k, max_delta_k)
        new_k = new_params['k'] + delta
        new_k = max(param_bounds['k'][0], min(new_k, param_bounds['k'][1]))
        new_params['k'] = int(new_k)
    elif hyper_to_perturb == 'distance_metric':
        possible_metrics = param_bounds['distance_metric'].copy()
        possible_metrics.remove(new_params['distance_metric'])
        new_params['distance_metric'] = random.choice(possible_metrics)
    elif hyper_to_perturb == 'weighted':
        new_params['weighted'] = not new_params['weighted']
    elif hyper_to_perturb == 'p':
        max_delta_p = (param_bounds['p'][1] - param_bounds['p'][0]) * 0.1 * step_size_factor
        delta = random.uniform(-max_delta_p, max_delta_p)
        new_p = new_params['p'] + delta
        new_p = max(param_bounds['p'][0], min(new_p, param_bounds['p'][1]))
        new_params['p'] = round(new_p, 2)
    elif hyper_to_perturb == 'scaling_method':
        possible_methods = param_bounds['scaling_method'].copy()
        possible_methods.remove(new_params['scaling_method'])
        new_params['scaling_method'] = random.choice(possible_methods)
    elif hyper_to_perturb == 'regularization':
        max_delta_reg = (param_bounds['regularization'][1] - param_bounds['regularization'][0]) * 0.1 * step_size_factor
        delta = random.uniform(-max_delta_reg, max_delta_reg)
        new_reg = new_params['regularization'] + delta
        new_reg = max(param_bounds['regularization'][0], min(new_reg, param_bounds['regularization'][1]))
        new_params['regularization'] = new_reg
    elif hyper_to_perturb == 'feature_transformations':
        possible_transformations = param_bounds['feature_transformations'].copy()
        current_transformations = set(new_params['feature_transformations'])
        if random.random() < 0.5:
            # Add a transformation
            remaining_transformations = list(set(possible_transformations) - current_transformations)
            if remaining_transformations:
                new_transformation = random.choice(remaining_transformations)
                new_params['feature_transformations'].append(new_transformation)
        else:
            # Remove a transformation
            if current_transformations:
                transformation_to_remove = random.choice(list(current_transformations))
                new_params['feature_transformations'].remove(transformation_to_remove)
    elif hyper_to_perturb == 'feature_weights':
        # Perturb feature weights
        feature_weights = new_params.get('feature_weights', np.ones(len(feature_names)))
        # Randomly select a feature to adjust its weight
        feature_idx = random.randint(0, len(feature_weights) - 1)
        max_delta_weight = (param_bounds['feature_weights'][1] - param_bounds['feature_weights'][0]) * 0.1 * step_size_factor
        delta = random.uniform(-max_delta_weight, max_delta_weight)
        new_weight = feature_weights[feature_idx] + delta
        new_weight = max(param_bounds['feature_weights'][0], min(new_weight, param_bounds['feature_weights'][1]))
        feature_weights[feature_idx] = new_weight
        new_params['feature_weights'] = feature_weights
    return new_params

def simulated_annealing(X, y, initial_params, param_bounds, feature_names,
                        max_evals=100, initial_temp=100.0,
                        cooling_rate=0.95, random_state=42, num_initializations=5,
                        min_sample_size=500, max_sample_size=None, patience=3):
    if random_state:
        np.random.seed(random_state)
        random.seed(random_state)
    total_samples = len(X)
    if max_sample_size is None:
        max_sample_size = len(X)
    sample_size = min_sample_size
    sample_increment = max(1, (max_sample_size - min_sample_size) // max_evals)

    best_params = None
    best_auc = -np.inf

    # Initial random initializations
    print("Starting random initializations...")
    for init in range(num_initializations):
        current_params = {}
        for param in initial_params.keys():
            if param == 'k':
                current_params['k'] = random.randint(param_bounds['k'][0], param_bounds['k'][1])
            elif param == 'distance_metric':
                current_params['distance_metric'] = random.choice(param_bounds['distance_metric'])
            elif param == 'weighted':
                current_params['weighted'] = random.choice(param_bounds['weighted'])
            elif param == 'p':
                current_params['p'] = round(random.uniform(param_bounds['p'][0], param_bounds['p'][1]), 2)
            elif param == 'scaling_method':
                current_params['scaling_method'] = random.choice(param_bounds['scaling_method'])
            elif param == 'regularization':
                current_params['regularization'] = random.uniform(param_bounds['regularization'][0], param_bounds['regularization'][1])
            elif param == 'feature_transformations':
                current_params['feature_transformations'] = random.sample(param_bounds['feature_transformations'],
                                                                          random.randint(0, len(param_bounds['feature_transformations'])))
            elif param == 'feature_weights':
                # Initialize feature weights to ones
                current_params['feature_weights'] = np.ones(len(feature_names))

        temp = initial_temp
        current_auc = -np.inf
        no_improvement_counter = 0
        try:
            for i in range(max_evals):
                # Increase sample size
                sample_size = min(max_sample_size, sample_size + sample_increment)
                sample_indices = np.random.choice(len(X), size=sample_size, replace=False)
                X_sample, y_sample = X[sample_indices], y[sample_indices]
                X_train_sa, X_val_sa, y_train_sa, y_val_sa = train_validation_split(X_sample, y_sample, test_size=0.2, random_state=random_state)

                new_params = perturb_params(current_params, param_bounds, feature_names, step_size_factor=(1 - i / max_evals))
                new_auc = evaluate_params(X_train_sa, y_train_sa, X_val_sa, y_val_sa, new_params, feature_names)

                delta_auc = new_auc - current_auc

                acceptance_prob = np.exp(delta_auc / (temp * (1 + abs(delta_auc)))) if temp > 1e-5 else 0

                if delta_auc > 0 or delta_auc > -.1 and acceptance_prob > random.random():
                    print(f'Accepted new params {new_params}')
                    current_params = new_params.copy()
                    current_auc = new_auc
                    no_improvement_counter = 0
                    if new_auc > best_auc:
                        best_params = new_params.copy()
                        best_auc = new_auc
                        print(f"Initialization {init+1}, Iteration {i+1}: New best AUC: {best_auc:.4f}")
                else:
                    print(f'Keeping old params {current_params}')
                    no_improvement_counter += 1

                temp *= cooling_rate

                if no_improvement_counter >= patience + 10 * new_auc:
                    print(f"No improvement in {int(patience + 10 * new_auc)} iterations. Stopping early.")
                    break

        except KeyboardInterrupt:
            print("Optimization interrupted by user.")
            break

    print(f"\nBest Hyperparameters found: {best_params} with AUC = {best_auc:.4f}")
    return best_params
