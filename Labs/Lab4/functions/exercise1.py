import numpy as np
import matplotlib.pyplot as plt

def KNN(x_train, class_train, x_test, N_test, K):

    class_pred = []

    for i in range(N_test):

        distance_xi = np.linalg.norm(x_test[i] - x_train, axis = 1)
        id = np.argpartition(distance_xi, K)[:K]
        labels = class_train[id].astype(np.int64)
        class_pred.append(np.bincount(labels).argmax())

    return np.array(class_pred)


def get_error_rate(class_pred, class_test):
    return np.mean(np.abs(class_pred - class_test))


def predict_and_plot(x_train, class_train, x_test, class_test, N_test, K):
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
    plt.show()
