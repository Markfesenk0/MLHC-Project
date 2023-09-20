import constants
from sklearn.model_selection import train_test_split


def extract_X_y(hosps):
    """
    Split a matrix containing features and target labels to a feature matrix and a target labels matrix
    :param hosps: matrix containing features and target labels
    :return: A matrix containing features and a matrix containing target labels
    """
    y = hosps[constants.target_labels]
    X = hosps.drop(constants.target_labels, axis=1)
    X = X.reindex(sorted(X.columns), axis=1)
    return X, y


def make_partition(hosps):
    """
    Gets matrix that contains features and target labels and returns train and test feature matrices and train and test target labels matrices
    :param hosps: matrix containing features and target labels
    :return: Train and test feature matrices and train and test target labels matrices
    """
    X, y = extract_X_y(hosps)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    return X_train, X_test, y_train, y_test
