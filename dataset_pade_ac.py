#File created with purpose to create and transform datasets

from sklearn.datasets import fetch_openml
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import pandas as pd
import jax.numpy as jnp
from sklearn.model_selection import train_test_split


def load_adult_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """
    Load the Adult Income dataset from OpenML.

    Returns
    X : pd.DataFrame
        Feature matrix containing demographic and employment information.

    y : pd.Series
        Binary target variable:
        - 1 if income > 50K
        - 0 otherwise.
    """
    adult = fetch_openml("adult", version=2, as_frame=True)

    X = adult.data
    y = (adult.target == ">50K").astype(int)

    return X, y


def get_feature_types(
    X: pd.DataFrame,
) -> tuple[pd.Index, pd.Index]:
    """
    Identify numerical and categorical columns.

    Parameters
    ----------
    X : pd.DataFrame
        Input feature matrix.

    Returns
    -------
    categorical_cols : pd.Index
        Names of categorical columns.

    numeric_cols : pd.Index
        Names of numerical columns.
    """
    categorical_cols = X.select_dtypes(
        include=["category", "object"]
    ).columns

    numeric_cols = X.select_dtypes(
        exclude=["category", "object"]
    ).columns

    return categorical_cols, numeric_cols


def create_preprocessor(
    categorical_cols,
    numeric_cols,
    scaler = StandardScaler(),
) -> ColumnTransformer:
    """
    Create a preprocessing pipeline for numerical and categorical features. Numerical features are standardized using StandardScaler.
    Categorical features are one-hot encoded.

    Parameters
        Categorical_cols (array-like): Names of categorical columns.
        numeric_cols (array-like): Names of numerical columns.
        scaler (scaler-class): Scaler from sklearn. Default StandartScaler()

    Returns
        preprocess : ColumnTransformer
        Scikit-learn preprocessing transformer.
    """
    preprocess = ColumnTransformer(
        transformers=[
            (
                "num",
                scaler,
                numeric_cols,
            ),
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                categorical_cols,
            ),
        ]
    )

    return preprocess


def split_and_preprocess_data(
    X,
    y,
    train_size=0.6,
    val_size=0.2,
    test_size=0.2,
    random_state=42,
):
    """
    Split a dataset into training, validation, and test sets, apply preprocessing,
    and convert targets to JAX arrays.

    Parameters:
        X (pd.DataFrame): Feature data.
        y (pd.Series): Target labels.
        train_size (float, optional): Fraction of samples assigned to the training set. Default is 0.6.
        val_size (float, optional): Fraction of samples assigned to the validation set. Default is 0.2.
        test_size (float, optional): Fraction of samples assigned to the test set. Default is 0.2.
        random_state (int, optional): Random seed used for reproducibility. Default is 42.

    Returns:
        X_train_proc (np.ndarray): Separeted training features.
        X_val (np.ndarray): Separeted validation features.
        X_test (np.ndarray): Separeted test features.
        y_train (jnp.ndarray): Training targets with shape (n_train, 1).
        y_val (jnp.ndarray): Validation targets with shape (n_val, 1).
        y_test (jnp.ndarray): Test targets with shape (n_test, 1).
    """

    total = train_size + val_size + test_size

    if not abs(total - 1.0) < 1e-8:
        raise ValueError(
            "train_size + val_size + test_size must sum to 1.0"
        )

    # Split: test set
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # Validation fraction relative to train+validation subset
    val_relative_size = val_size / (train_size + val_size)

    # Split: train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=val_relative_size,
        random_state=random_state,
        stratify=y_trainval,
    )

    return (
        X_train, X_val, X_test, y_train, y_val, y_test,
    )