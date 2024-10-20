import numpy as np
import pandas as pd

def preprocess_data(train_path, test_path, scaling_method='minmax', feature_transformations=None, return_feature_names=False):
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

    # Store original feature names
    feature_names = combined_features.columns.tolist()

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

    # Update feature names after transformations
    feature_names = combined_features.columns.tolist()

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

    if return_feature_names:
        return X_train, y_train, X_test, test_ids, feature_names
    else:
        return X_train, y_train, X_test, test_ids
