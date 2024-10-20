import numpy as np

class SimpleKNN:
    def __init__(self, k=3, distance_metric='euclidean'):
        self.k = k
        self.distance_metric = distance_metric

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = []
        for x in X:
            distances = self.compute_distance(x, self.X_train)
            nearest_indices = np.argsort(distances)[:self.k]
            nearest_labels = self.y_train[nearest_indices]
            prob = np.mean(nearest_labels)  # Average of nearest neighbors
            predictions.append(prob)
        return np.array(predictions)
    
    def compute_distance(self, x, X_train):
        if self.distance_metric == 'euclidean':
            return np.sqrt(np.sum((X_train - x) ** 2, axis=1))
        
        # TODO maybe still have other distance metrics?
        else:
            raise ValueError("Unsupported distance metric.")

class KNN:
    def __init__(self, k=3, distance_metric='euclidean', weighted=False, p=2.0, regularization=1e-5, feature_weights=None):
        self.k = k
        self.distance_metric = distance_metric
        self.weighted = weighted
        self.p = p  # Parameter for Minkowski distance
        self.regularization = regularization  # Regularization for Mahalanobis distance
        self.feature_weights = feature_weights  # Feature weights for distance computation

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

        # Apply feature weights
        if self.feature_weights is not None:
            self.feature_weights = np.array(self.feature_weights)
            if self.X_train.shape[1] % len(self.feature_weights) != 0:
                print(f'WARN mismatch in sizes X_train {self.X_train.shape} and feature_weights {self.feature_weights.shape}')
                if len(self.feature_weights) > self.X_train.shape[1]:
                    self.feature_weights = self.feature_weights[:self.X_train.shape[1]]
                else:
                    self.feature_weights = self.feature_weights + np.ones(self.X_train.shape[1] - len(self.feature_weights))
            else:
                self.feature_weights = np.tile(self.feature_weights, self.X_train.shape[1] // len(self.feature_weights))
            self.X_train = self.X_train * self.feature_weights

    def predict(self, X):
        if self.feature_weights is not None:
            X = X * self.feature_weights

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
                print(f"Processed {idx + 1}/{len(X)} samples ", end='')
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