"""
NumPy House Price Regression

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - impute_nan_with_mean
import numpy as np

def impute_nan_with_mean(X):
    """Replace every NaN in X with that column's nan-aware mean (all-NaN cols -> 0).

    Args:
        X: (N, F) array-like of floats, may contain NaN.

    Returns:
        (N, F) float ndarray with no NaNs.
    """
    # TODO: Replace every NaN with that column's nan-aware mean...
    X_clean = np.array(X, dtype = float).copy()

    nan_mask = np.isnan(X_clean) # array which contains true and false values (null == true)

    counts = np.sum(~nan_mask , axis = 0)

    sums = np.nansum(X_clean , axis = 0)

    cols_mean = np.where(counts > 0, sums/counts, 0.0)

    X_clean = np.where(nan_mask , cols_mean , X_clean)

    return X_clean

# Step 2 - compute_iqr_bounds
import numpy as np
def compute_iqr_bounds(X, k=1.5):
    # TODO: Compute per-column lower/upper clip bounds using the IQR rule.

    q1 = np.percentile(X , 25, axis = 0)
    q3 = np.percentile(X, 75, axis = 0)

    iqr = q3 - q1

    lower = q1 - k*iqr
    higher = q3 + k*iqr 

    return lower,higher

# Step 3 - clip_columns
import numpy as np


def clip_columns(X, lower, upper):
    # TODO: Clip every entry of a feature matrix to per-column lower/upper bounds.
    

    X_clipped = np.array(X, dtype = float).copy()

    X_clipped = np.clip(X_clipped, lower, upper)

    return X_clipped

# Step 4 - make_ratio_feature
def make_ratio_feature(numerator, denominator, eps=1e-8):
    # TODO: Form a derived ratio feature from two 1-D arrays with safe division.
    denominator += eps

    new_feature = numerator/denominator

    return new_feature

# Step 5 - append_column
import numpy as np

def append_column(X, col):
    # TODO: Horizontally append one 1-D feature column onto a design matrix.

    result = np.column_stack((X , col))

    return result

# Step 6 - one_hot_encode
import numpy as np

def one_hot_encode(labels):
    # TODO: Convert a 1-D array of categorical labels into a dense binary one-hot matrix.

    unique_labels = np.unique(labels)

    rows = len(labels)
    columns = len(unique_labels)

    one_hot = np.zeros((rows,columns), dtype= float)

    indices = np.searchsorted(unique_labels, labels)

    one_hot[np.arange(rows), indices] = 1.0

    return one_hot

# Step 7 - fit_standardizer
import numpy as np

def fit_standardizer(X):
    # TODO: Compute per-column mean and std used to standardize features...

    mean = np.mean(X, axis = 0)

    std = np.std(X, axis = 0)

    std = np.where(std == 0.0, 1.0, std)

    return mean, std

# Step 8 - apply_standardizer
import numpy as np

def apply_standardizer(X, mean, std):
    # TODO: Return the scaled matrix (X - mean) / std via broadcasting.

    X_clean = np.array(X ,dtype = float).copy()

    X_clean = (X_clean - mean) / std
    
    return X_clean

# Step 9 - add_bias_column
import numpy as np

def add_bias_column(X):
    # TODO: Prepend a column of ones to a 2-D feature matrix X...

    bias_col = np.ones(X.shape[0], dtype = float)

    X = np.column_stack((bias_col, X))

    return X

# Step 10 - make_shuffled_indices
import numpy as np

def make_shuffled_indices(n_samples, seed):
    # 1. Initialize a random number generator locked to our specific seed
    rng = np.random.default_rng(seed)
    
    # 2. Generate a reproducibly shuffled permutation from 0 up to n_samples-1
    indices = rng.permutation(n_samples)
    
    return indices

# Step 11 - partition_indices
import numpy as np

def partition_indices(indices, train_ratio, val_ratio):
    # TODO: Split a shuffled index array into train, validation, and test index arrays.

    N = len(indices)


    train_end = int(np.floor(N*train_ratio))
    val_end = train_end + int(np.floor(N*val_ratio))

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]
    
    return train_idx, val_idx, test_idx
    
    
    
    
    pass

# Step 12 - subset_xy
import numpy as np

def subset_xy(X, y, indices):
    # 1. Select the rows matching the indices from the 2-D matrix X
    X_sub = X[indices]
    
    # 2. Select the corresponding values matching the indices from the 1-D vector y
    y_sub = y[indices]
    
    return X_sub, y_sub

# Step 13 - ols_fit
import numpy as np 

def ols_fit(X, y):
    # TODO: return the ordinary-least-squares weight vector for a linear model.

    A = X.T @ X

    B = X.T @ y

    C = np.linalg.solve(A, B)

    return C

# Step 14 - ols_predict
import numpy as np

def ols_predict(X, theta):
    # TODO: Predict continuous targets with a fitted linear model.

    predictions = X @ theta

    return predictions

# Step 15 - mean_absolute_error
import numpy as np
def mean_absolute_error(y_true, y_pred):
    # TODO: return the mean absolute error between targets and predictions

    return np.mean(np.abs(y_pred - y_true))

# Step 16 - root_mean_squared_error
def root_mean_squared_error(y_true, y_pred):
    """Compute root mean squared error between targets and predictions.

    Args:
        y_true (np.ndarray): Ground-truth targets, shape (N,).
        y_pred (np.ndarray): Predicted targets, shape (N,).

    Returns:
        float: RMSE value.
    """
    # TODO: return the root mean squared error as a Python float
    errors = (y_true - y_pred) ** 2

    return np.sqrt(np.mean(errors))

# Step 17 - r_squared
import numpy as np

def r_squared(y_true, y_pred):
    # 1. Calculate the Residual Sum of Squares (SS_res)
    ss_res = np.sum((y_true - y_pred) ** 2)
    
    # 2. Calculate the Total Sum of Squares around the mean (SS_tot)
    mean_true = np.mean(y_true)
    ss_tot = np.sum((y_true - mean_true) ** 2)
    
    # 3. Handle the edge case where SS_tot is exactly zero to prevent division by zero
    if ss_tot == 0.0:
        return 0.0
        
    # 4. Compute the final R^2 value and convert to a standard Python float
    r2 = 1.0 - (ss_res / ss_tot)
    
    return float(r2)

# Step 18 - residual_summary
import numpy as np

def residual_summary(y_true, y_pred):
    # TODO: Return a compact dict summarizing prediction residuals...

    r = y_true - y_pred

    return {
        'mean' : float(np.mean(r)),
        'std' : float(np.std(r)),
        'median_abs' : float(np.median(abs(r)))
    }





    pass

# Step 19 - prepare_cleaned_features
import numpy as np

def prepare_cleaned_features(X, iqr_k=1.5):
    """Impute NaNs then IQR-clip columns to produce a clean numeric matrix.

    Args:
        X: (N, F) array-like of floats, may contain NaN.
        iqr_k: IQR multiplier passed to compute_iqr_bounds (default 1.5).

    Returns:
        (N, F) float ndarray with no NaNs, columns clipped to IQR bounds.
    """
    # TODO: Produce a clean numeric matrix via impute then IQR clip


    A = impute_nan_with_mean(X)

    B,C = compute_iqr_bounds(A, iqr_k)

    D = clip_columns(A, B, C)

    return D

# Step 20 - assemble_feature_matrix
import numpy as np

def assemble_feature_matrix(X_num, ratio_num_idx, ratio_den_idx, cat_labels=None):
    # 1. Extract the specific numerator and denominator columns from the numeric matrix
    numerator = X_num[:, ratio_num_idx]
    denominator = X_num[:, ratio_den_idx]
    
    # 2. Call your helper function to create the safe ratio feature
    ratio_col = make_ratio_feature(numerator, denominator)
    
    # 3. Append the ratio feature as the last column of the matrix
    X_extended = append_column(X_num, ratio_col)
    
    # 4. If categorical text labels exist, one-hot encode them and glue them side-by-side
    if cat_labels is not None:
        X_cat = one_hot_encode(cat_labels)
        X_extended = np.hstack((X_extended, X_cat))
        
    return X_extended

# Step 21 - make_train_val_test
import numpy as np

def make_train_val_test(X, y, train_ratio, val_ratio, seed):
    # 1. Get the total number of samples
    n_samples = X.shape[0]
    
    # 2. Use the legacy seed and permutation to match the test environment exactly
    np.random.seed(seed)
    shuffled_idx = np.random.permutation(n_samples)
    
    # 3. Partition the indices using your helper function
    train_idx, val_idx, test_idx = partition_indices(shuffled_idx, train_ratio, val_ratio)
    
    # 4. Extract the correct data splits
    X_train, y_train = subset_xy(X, y, train_idx)
    X_val, y_val = subset_xy(X, y, val_idx)
    X_test, y_test = subset_xy(X, y, test_idx)
    
    # 5. Return the splits dictionary with correct keys
    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test
    }

# Step 22 - standardize_and_add_bias (not yet solved)
# TODO: implement

# Step 23 - evaluate_predictions (not yet solved)
# TODO: implement

# Step 24 - house_price_pipeline (not yet solved)
# TODO: implement

