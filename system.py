"""Baseline classification system.

Solution outline for the COM2004/3004 assignment.

This solution will run but the dimensionality reduction and
the classifier are not doing anything useful, so it will
produce a very poor result.

version: v1.0
"""
from typing import List

import numpy as np
from scipy.stats import f_oneway



N_DIMENSIONS = 10


def classify(train: np.ndarray, train_labels: np.ndarray, test: np.ndarray) -> List[str]:
    """
    The `classify` function performs classification using the k-Nearest Neighbors (KNN) algorithm.
    It trains a KNN classifier on the provided training data and labels, and then predicts the
    labels for the given test data.

    Args:
        - train (np.ndarray): The training data, an array-like structure containing feature vectors.
        - train_labels (np.ndarray): The corresponding labels for the training data.
        - test (np.ndarray): The test data for which predictions are to be made.

    Returns:
        - List[str]: A list containing the predicted labels for the test data.

    """
    ##calling knn classifier
    knn = KNN(k=3)
    knn.fit(train, train_labels)
    predictions = knn.predict(test)
    return predictions


def reduce_dimensions(data: np.ndarray, model: dict) -> np.ndarray:
    """
    Reduce the dimensionality of input data using a two-step feature selection process.

    If the model dictionary contains only labels, the function assumes it is processing
    training data and performs the following steps:
    1. Conducts Univariate Feature Selection to select 700 feature indices.
    2. Applies Forward Feature Selection (FFS) on the selected indices to further narrow down to a specified number of dimensions.

    If the model dictionary contains selected indices, the function assumes it is processing
    testing data and returns the subset of features specified by the selected indices.

    Args:
        data (np.ndarray): The input data with feature vectors.
        model (dict): A dictionary storing model information.

    Returns:
        np.ndarray: The reduced data with the specified number of dimensions.
    """
    ##reducding dimentialties for training data
    if len(model) == 1:

        ##retriving the labels from the model
        labels_train = np.array(model["labels_train"])
        original_data = data.copy()

        # selection 700 features indices out of all features using Univariate feature selection
        selected_indices_univariate = univariate_feature_selection(data, labels_train, 700)

        # selection 10 features indices out of selected_indices_univariate using FFE feature selection
        data_univariate = original_data[:, selected_indices_univariate]
        selected_indices_ffs = selected_indices_univariate[forward_feature_selection(data_univariate, labels_train, N_DIMENSIONS)]

        
        return selected_indices_ffs
    else:
        ## reducing dimentialities for testing data

        ##retriving the selected indices from the model
        selected_indices = model["selected_indices"]
        
        ##returning the seleted data
        return data[:, selected_indices]


def process_training_data(fvectors_train: np.ndarray, labels_train: np.ndarray) -> dict:
    """Process the labeled training data and return model parameters stored in a dictionary.

    Args:
        fvectors_train (np.ndarray): training data feature vectors stored as rows.
        labels_train (np.ndarray): the labels corresponding to the feature vectors.

    Returns:
        dict: a dictionary storing the model data.
    """
    ##coverting the labels to integers
    label_mapping = {
        '.': 0, 'p': 1, 'r': 2, 'n': 3, 'b': 4, 'q': 5, 'k':6,
        'P': 7, 'R': 8, 'N': 9, 'B':10, 'Q': 11, 'K': 12 
    }

    labels_train = np.vectorize(label_mapping.get)(labels_train)

    model = {}
    ##storing the labels in the model
    model["labels_train"] = labels_train.tolist()

    ##reducing the dimentialities of the training data by retrivng the indices
    # from the reduce_dimensions function and filtering the fvectors_train
    selected_indices = reduce_dimensions(fvectors_train, model)
    fvectors_train_reduced = fvectors_train[:,selected_indices]
    
    ##storing the reduced data and the selected indices in the model
    model["fvectors_train"] = fvectors_train_reduced.tolist()
    model["selected_indices"] = selected_indices.tolist()
    
    return model


def images_to_feature_vectors(images: List[np.ndarray]) -> np.ndarray:
    """Takes a list of images (of squares) and returns a 2-D feature vector array.

    In the feature vector array, each row corresponds to an image in the input list.

    Args:
        images (list[np.ndarray]): A list of input images to convert to feature vectors.

    Returns:
        np.ndarray: An 2-D array in which the rows represent feature vectors.
    """
    h, w = images[0].shape
    n_features = h * w
    fvectors = np.empty((len(images), n_features))
    for i, image in enumerate(images):
        fvectors[i, :] = image.reshape(1, n_features)

    return fvectors


def classify_squares(fvectors_test: np.ndarray, model: dict) -> List[str]:
    """Run classifier on a array of image feature vectors presented in an arbitrary order.

    Note, the feature vectors stored in the rows of fvectors_test represent squares
    to be classified. The ordering of the feature vectors is arbitrary, i.e., no information
    about the position of the squares within the board is available.

    Args:
        fvectors_test (np.ndarray): An array in which feature vectors are stored as rows.
        model (dict): A dictionary storing the model data.

    Returns:
        list[str]: A list of one-character strings representing the labels for each square.
    """

    # Get some data out of the model. It's up to you what you've stored in here
    fvectors_train = np.array(model["fvectors_train"])
    labels_train = np.array(model["labels_train"])

    # Call the classify function.
    labels = classify(fvectors_train, labels_train, fvectors_test)

    return labels


def classify_boards(fvectors_test: np.ndarray, model: dict) -> List[str]:
    """Run classifier on a array of image feature vectors presented in 'board order'.

    The feature vectors for each square are guaranteed to be in 'board order', i.e.
    you can infer the position on the board from the position of the feature vector
    in the feature vector array.

    In the dummy code below, we just re-use the simple classify_squares function,
    i.e. we ignore the ordering.

    Args:
        fvectors_test (np.ndarray): An array in which feature vectors are stored as rows.
        model (dict): A dictionary storing the model data.

    Returns:
        list[str]: A list of one-character strings representing the labels for each square.
    """

    return classify_squares(fvectors_test, model)

def forward_feature_selection(data, labels, num_features): 
    """
    forward feature selection (ffs) algorithm
    
    parameters:
    - data: input data.
    - labels: target variable.
    - num_features: number of features to select.

    returns:
    - selected_indices: list of indices of selected features.
    """
   
    n_total_features = data.shape[1]
    
    ## initialise the set of remaining features to be all features and selected features to be empty
    remaining_features_set = set(range(n_total_features))
    selected_feature_indices = []

    ## continueing selecting features until reaching the desried numbers
    while len(selected_feature_indices) < num_features:
        best_feature_index = None
        best_score = -np.inf

        # iteratethrough remaining features
        for feature_index in remaining_features_set:
            current_selected_indices = selected_feature_indices + [feature_index]
            current_data = data[:, current_selected_indices].reshape(-1, len(current_selected_indices))

            ## applying a linear regression
            beta, _, _, _ = np.linalg.lstsq(current_data, labels, rcond=None)
            residuals = labels - np.dot(current_data, beta)
            
            ##calculating t-statistic for the coefficent of the feature
            t_statistic = beta[-1] / (np.std(residuals) / np.sqrt(np.sum(current_data[:, 0] ** 2)))
            
            ##using absolute value of the t-statistic for ranking
            score = np.abs(t_statistic)
            
            if score > best_score:
                best_score = score
                best_feature_index = feature_index

        ##remove the selected feature from the remaining features
        remaining_features_set.remove(best_feature_index)

        #  append the index of the best feature to the selected feature indices
        selected_feature_indices.append(best_feature_index)

    ##return the np array of hte selected feature indices
    return np.array(selected_feature_indices)


def univariate_feature_selection(data, labels, num_features):
    """
    univariate_feature_selection (ufs) algorithm
    
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.f_oneway.html
    
    parameters:
    - data: input data.
    - labels: target variable.
    - n_features_to_select: number of features to select.

    returns:
    - selected_indices: list of indices of selected features.
    """
    ##ANOVA F-value For Feature Selection
    #compute the F-values and p-values for each feature
    f_values, p_values = f_oneway(*[data[labels == i] for i in np.unique(labels)])

    #sort the features by their p-values
    sorted_indices = np.argsort(p_values)

    #return the indices of the top num_features features
    return sorted_indices[:num_features]


##KNN classifier
class KNN:
    ##initialising the k value 
    def __init__(self, k=3):
        self.k = k
        self.train = None
        self.train_labels = None

    ##fitting the training data and labels
    def fit(self, train: np.ndarray, train_labels: np.ndarray) -> None:
        self.train = train
        self.train_labels = train_labels
    
    ##predicting the test data
    def predict(self, test: np.ndarray) -> List[str]:
        ##calculating the distances between the test and train data using eculedian distance
        distances = np.sqrt(np.sum((self.train - test[:, np.newaxis])**2, axis=2))
        
        ##sorting the distances and retriving the k nearest neighbours
        nearest_neighbors = np.argsort(distances, axis=1)[:, :self.k]
        nearest_labels = self.train_labels[nearest_neighbors]
        predictions = [np.bincount(labels).argmax() for labels in nearest_labels]
        
        ##mapping the labels to the original labels (int to cgaracter)
        reverse_label_mapping = {
            0: '.', 1: 'p', 2: 'r', 3: 'n', 4: 'b', 5: 'q', 6: 'k',
            7: 'P', 8: 'R', 9: 'N', 10: 'B', 11: 'Q', 12: 'K'
        }
        
        original_predictions = np.vectorize(reverse_label_mapping.get)(predictions) 
        
        ##returning the predictions
        return original_predictions.tolist()
