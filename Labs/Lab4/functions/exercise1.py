import numpy as np
import matplotlib.pyplot as plt

def KNN(x_train, class_train, x_test, N_test, K):
    """
    Perform K-Nearest Neighbors classification

    Parameters :
    x_train : np.ndarray
        Training data containing feature vectors.
    class_train : np.ndarray
        Class labels corresponding to each training sample.
    x_test : np.ndarray
        Test data for which class labels are to be predicted.
    N_test : int
        Number of test samples to classify.
    K : int
        Number of nearest neighbors to consider.

    Returns :
    np.ndarray
        Predicted class labels for each test sample.

    Notes :
    np.argpartition is used for efficiency to select the K smallest
      distances without fully sorting the array.
    - In case of ties during count, the smallest class label
      is chosen by np.bincount().argmax().
      """

    class_pred = []

    for i in range(N_test):
        #Compute the distance between x_test[i] and each point of training set
        distance_xi = np.linalg.norm(x_test[i] - x_train, axis = 1)
        #Sort and get K nearest ids
        id = np.argpartition(distance_xi, K)[:K]
        #Get corresponding labels
        labels = class_train[id].astype(np.int64)
        #selection final class
        class_pred.append(np.bincount(labels).argmax())

    return np.array(class_pred)


def get_error_rate(class_pred, class_test):
    """
    Compute the classification error rate (proportion of incorrectly
    predicted labels compared to the true class labels).

    Parameters :
    class_pred : np.ndarray
        Predicted class labels.
    class_test : np.ndarray
        True class labels.

    Returns :
    float
        The error rate
    """
    return np.mean(np.abs(class_pred - class_test))


def predict_and_plot(x_train, class_train, x_test, class_test, N_test, K):
    """
    Performs KNN classification on the test set, identifies misclassified points, and plots both the predicted labels and misclassifications along with the training data.
    """
    class_pred_K = KNN(x_train, class_train, x_test, N_test, K)

    diff =  np.abs(class_pred_K - class_test)

    x_test_bad_pred = x_test[np.where(diff)]

    plt.scatter(x_test[:,0], x_test[:,1], c = class_pred_K, marker = 'x')
    plt.scatter(x_test_bad_pred[:,0], x_test_bad_pred[:,1], c = 'r', marker = 'x', label = "false_prediction")
    plt.scatter(x_train[:,0], x_train[:,1], c = class_train, alpha = 0.3)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Predicted class compared to train set (K = {K})")
    plt.grid()
    plt.legend()
    plt.savefig(f'images/plot_prediction_K_{K}.png')
    plt.show()
