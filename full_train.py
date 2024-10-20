import numpy as np
import pandas as pd
import random
import sys
import time

def preprocess_data(train_path, test_path, scaling_method='minmax', feature_transformations=None):
    # Load data
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    # Drop unnecessary columns
    train_data = train_data.drop(['CustomerId', 'Surname'], axis=1)
    test_data = test_data.drop(['CustomerId', 'Surname'], axis=1)

    # Store 'id' column from test_data
    test_ids = test_data['id']

    # Drop 'id' column from train and test data
    train_data = train_data.drop(['id'], axis=1)
    test_data = test_data.drop(['id'], axis=1)

    # Identify numerical and categorical features
    numerical_features = ['CreditScore', 'Age', 'Tenure', 'Balance',
                          'NumOfProducts', 'EstimatedSalary']
    categorical_features = ['Geography', 'Gender']

    # Handle missing values
    # Numerical features: fill with mean
    for feature in numerical_features:
        mean_value = train_data[feature].mean()
        train_data[feature].fillna(mean_value, inplace=True)
        test_data[feature].fillna(mean_value, inplace=True)

    # Categorical features: fill with mode
    for feature in categorical_features:
        mode_value = train_data[feature].mode()[0]
        train_data[feature].fillna(mode_value, inplace=True)
        test_data[feature].fillna(mode_value, inplace=True)

    # Encode categorical variables
    # Map 'Gender' to numerical values
    gender_mapping = {'Male': 0, 'Female': 1}
    train_data['Gender'] = train_data['Gender'].map(gender_mapping)
    test_data['Gender'] = test_data['Gender'].map(gender_mapping)

    # One-hot encode 'Geography'
    train_data = pd.get_dummies(train_data, columns=['Geography'], drop_first=True)
    test_data = pd.get_dummies(test_data, columns=['Geography'], drop_first=True)

    # Align test data columns with train data
    missing_cols = set(train_data.columns) - set(test_data.columns) - set(['Exited'])
    for c in missing_cols:
        test_data[c] = 0
    test_data = test_data[train_data.drop('Exited', axis=1).columns]

    # Separate features and target
    X_train = train_data.drop('Exited', axis=1)
    y_train = train_data['Exited']

    # Combine features for transformations and scaling
    combined_features = pd.concat([X_train, test_data], axis=0)
    combined_features = combined_features.astype(float)

    # Apply feature transformations if any
    if feature_transformations is not None:
        for transformation in feature_transformations:
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
                # Add interaction terms between numerical features
                for i in range(len(numerical_features)):
                    for j in range(i+1, len(numerical_features)):
                        f1 = numerical_features[i]
                        f2 = numerical_features[j]
                        combined_features[f'{f1}_x_{f2}'] = combined_features[f1] * combined_features[f2]

    # Scaling
    if scaling_method == 'minmax':
        feature_min = combined_features.min()
        feature_max = combined_features.max()
        combined_features = (combined_features - feature_min) / (feature_max - feature_min)
    elif scaling_method == 'standard':
        feature_mean = combined_features.mean()
        feature_std = combined_features.std()
        combined_features = (combined_features - feature_mean) / feature_std
    elif scaling_method == 'none':
        pass
    else:
        raise ValueError("Unsupported scaling method.")

    # Split back into train and test data
    X_train = combined_features.iloc[:len(X_train), :].values
    X_test = combined_features.iloc[len(X_train):, :].values
    y_train = y_train.values

    return X_train, y_train, X_test, test_ids

class KNN:
    def __init__(self, k=3, distance_metric='euclidean', weighted=False, p=2.0, regularization=1e-5):
        self.k = k
        self.distance_metric = distance_metric
        self.weighted = weighted
        self.p = p  # Parameter for Minkowski distance
        self.regularization = regularization  # Regularization for Mahalanobis distance

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        if self.distance_metric == 'mahalanobis':
            # Compute the inverse covariance matrix with regularization
            cov_matrix = np.cov(self.X_train.T) + self.regularization * np.eye(self.X_train.shape[1])
            try:
                self.S_inv = np.linalg.inv(cov_matrix)
            except np.linalg.LinAlgError:
                # If the matrix is singular, use pseudo-inverse
                self.S_inv = np.linalg.pinv(cov_matrix)
        else:
            self.S_inv = None  # Not used

    def predict(self, X):
        predictions = []
        for idx, x in enumerate(X):
            distances = self.compute_distance(x, self.X_train)
            nearest_indices = np.argsort(distances)[:self.k]
            nearest_labels = self.y_train[nearest_indices]
            if self.weighted:
                nearest_distances = distances[nearest_indices]
                nearest_distances = np.where(nearest_distances == 0, 1e-5, nearest_distances)
                weights = 1 / nearest_distances
                prob = np.sum(weights * nearest_labels) / np.sum(weights)
            else:
                prob = np.mean(nearest_labels)
            predictions.append(prob)
            # monitor prediction progress
            if (idx + 1) % 1000 == 0 or (idx + 1) == len(X):
                print(f"Processed {idx + 1}/{len(X)} samples")
        return np.array(predictions)

    def compute_distance(self, x, X_train):
        if self.distance_metric == 'euclidean':
            return np.sqrt(np.sum((X_train - x) ** 2, axis=1))
        elif self.distance_metric == 'manhattan':
            return np.sum(np.abs(X_train - x), axis=1)
        elif self.distance_metric == 'minkowski':
            return np.sum(np.abs(X_train - x) ** self.p, axis=1) ** (1 / self.p)
        elif self.distance_metric == 'cosine':
            numerator = np.sum(X_train * x, axis=1)
            denominator = np.linalg.norm(X_train, axis=1) * np.linalg.norm(x)
            denominator = np.where(denominator == 0, 1e-5, denominator)
            return 1 - (numerator / denominator)
        elif self.distance_metric == 'mahalanobis':
            delta = X_train - x
            left_term = np.dot(delta, self.S_inv)
            distances = np.sqrt(np.sum(left_term * delta, axis=1))
            return distances
        else:
            raise ValueError("Unsupported distance metric.")

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

def evaluate_params(X_train_full, y_train_full, X_val_full, y_val_full, params):
    # Preprocess data with the specified scaling method and feature transformations
    X_combined = np.vstack((X_train_full, X_val_full))
    y_combined = np.concatenate((y_train_full, y_val_full))

    # Create DataFrame for combined features
    combined_features = pd.DataFrame(X_combined)
    combined_features.columns = [f'feature_{i}' for i in range(combined_features.shape[1])]

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

    knn = KNN(k=params['k'],
              distance_metric=params['distance_metric'],
              weighted=params['weighted'],
              p=params.get('p', 2.0),
              regularization=params.get('regularization', 1e-5))
    knn.fit(X_train, y_train)
    y_scores = knn.predict(X_val)
    auc = roc_auc_score(y_val, y_scores)
    return auc

def perturb_params(params, param_bounds, step_size_factor=1.0):
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
    return new_params

def simulated_annealing(X, y, initial_params, param_bounds, max_evals=100, initial_temp=100.0,
                        cooling_rate=0.95, random_state=42, num_initializations=5,
                        min_sample_size=500, max_sample_size=None, patience=3):
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

                new_params = perturb_params(current_params, param_bounds, step_size_factor=(1 - i / max_evals))
                new_auc = evaluate_params(X_train_sa, y_train_sa, X_val_sa, y_val_sa, new_params)

                delta_auc = new_auc - current_auc

                acceptance_prob = np.exp(delta_auc / temp) if temp > 1e-5 else 0

                if delta_auc > 0 or acceptance_prob > random.random():
                    current_params = new_params
                    current_auc = new_auc
                    if delta_auc > 0:
                        no_improvement_counter = 0
                    else:
                        no_improvement_counter += 1
                    if new_auc > best_auc:
                        best_params = new_params.copy()
                        best_auc = new_auc
                        print(f"Initialization {init+1}, Iteration {i+1}: New best AUC: {best_auc:.4f}")
                        print(best_params)
                else:
                    no_improvement_counter += 2

                temp *= cooling_rate

                if no_improvement_counter >= patience:
                    print(f"No improvement in {patience} iterations. Stopping early.")
                    break

        except KeyboardInterrupt:
            print("Optimization interrupted by user.")
            break

    print(f"\nBest Hyperparameters found: {best_params}\n with AUC = {best_auc:.4f}\n\n")
    return best_params

def main():
    train_path = 'train.csv'
    test_path = 'test.csv'

    # Load initial data to get the number of features
    X, y, X_test, test_ids = preprocess_data(train_path, test_path, scaling_method='none')
    print("Data preprocessing completed.")

    # Define possible feature transformations
    possible_transformations = ['log', 'sqrt', 'square', 'interaction']

    # Initial hyperparameters
    initial_params = {
        'k': 5,
        'distance_metric': 'euclidean',
        'weighted': False,
        'p': 2.0,
        'scaling_method': 'minmax',
        'regularization': 1e-5,
        'feature_transformations': []
    }

    # Define parameter bounds
    param_bounds = {
        'k': (1, 51),
        'distance_metric': ['euclidean', 'manhattan', 'minkowski', 'cosine', 'mahalanobis'],
        'weighted': [True, False],
        'p': (1.0, 5.0),
        'scaling_method': ['minmax', 'standard', 'none'],
        'regularization': (1e-5, 1e-2),
        'feature_transformations': possible_transformations
    }

    # Perform Simulated Annealing for hyperparameter optimization
    best_params = simulated_annealing(
        X, y, initial_params, param_bounds,
        max_evals=100, initial_temp=100.0, cooling_rate=0.95, random_state=42,
        num_initializations=25, min_sample_size=500, max_sample_size=5000, patience=10
    )
    print("Hyperparameter optimization completed.")

    # Preprocess the entire training and test data with the best parameters
    X_train, y_train, X_test, test_ids = preprocess_data(train_path, test_path,
                                                         scaling_method=best_params['scaling_method'],
                                                         feature_transformations=best_params.get('feature_transformations', []))
    print("Data preprocessing with best parameters completed.")

    # Train final model with best hyperparameters
    final_knn = KNN(k=best_params['k'],
                    distance_metric=best_params['distance_metric'],
                    weighted=best_params['weighted'],
                    p=best_params.get('p', 2.0),
                    regularization=best_params.get('regularization', 1e-5))
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
    print(f"Best AUC from optimization: {best_params}")

if __name__ == "__main__":
    main()
