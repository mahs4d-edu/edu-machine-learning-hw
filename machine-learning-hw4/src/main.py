from os import path

import math
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

ALPHA = 0.000001
THRESHOLD = 0.1
EPSILON = 0.000001


def load_dataset(file_name, t):
    """
    loads dataset and returns it
    :param file_name:
    :param t: type of processing on dataset (s: standard, l: log, b: binary)
    :return:
    """
    file_location = path.join(path.abspath(path.dirname(__file__)), '../data', file_name)
    dataset = np.loadtxt(file_location, dtype=np.float128)

    if t == 's':
        return standardize_dataset(dataset)
    elif t == 'l':
        return logplus_dataset(dataset)
    elif t == 'b':
        return binarize_dataset(dataset)


def standardize_dataset(dataset):
    """
    standardizes a dataset
    :param dataset:
    :return:
    """
    x = dataset[:, :-1]
    dataset[:, :-1] = (x - x.mean(axis=0)) / np.std(x, axis=0)
    return dataset


def logplus_dataset(dataset):
    """
    computes log(a + 0.1) on all elements of input
    :param dataset:
    :return:
    """
    x = dataset[:, :-1]
    dataset[:, :-1] = np.log(x + 0.1)
    return dataset


def binarize_dataset(dataset):
    """
    if a > 0 => that element will become 1 else => 0
    :param dataset:
    :return:
    """
    x = dataset[:, :-1]
    binary_x = x.copy()
    binary_x[binary_x > 0] = 1
    binary_x[binary_x <= 0] = 0
    dataset[:, :-1] = binary_x
    return dataset


def generate_training_test_datasets(dataset):
    """
    splites data to training and test datasets
    :param dataset:
    :return: training, test
    """
    np.random.shuffle(dataset)
    split_boundary = math.floor(80 * dataset.shape[0] / 100)
    training_dataset, test_dataset = dataset[:split_boundary], dataset[split_boundary:]
    return training_dataset, test_dataset


def h(x, beta):
    """
    hypothesis function (sigmoid of input x matrix with beta parameters)
    :param x: i*j matrix
    :param beta: j*1 vector
    :return: i*1 vector
    """
    return 1 / (1 + np.exp(-x.dot(beta)))


def compute_descent_size(x, y, beta):
    """
    computes the amount of descent for each parameter
    :param x: i*j matrix
    :param y: i*1 vector
    :param beta: j*1 vector
    :return: j*1 vector
    """
    return ((h(x, beta) - y).T.dot(x)).T


def gradient_descent_step(x, y, beta):
    """
    discends beta parameters for a single step and returns the result
    :param x: i*j matrix
    :param y: i*1 vector
    :param beta: j*1 vector
    :return: j*1 vector (new beta values)
    """

    return beta - (ALPHA * compute_descent_size(x, y, beta))


def cost(x, y, beta):
    """
    computes cost of logistic model based on current beta values
    :param x: i*j matrix
    :param y: i*1 vector
    :param beta: j*1 vector
    :return: number
    """

    h_value = h(x, beta)

    s1 = - (y.T.dot(np.log(h_value + EPSILON)))
    s2 = - ((1 - y).T.dot(np.log(1 - h_value + EPSILON)))

    return (s1 + s2)[0, 0]


def gradient_descent(training_dataset):
    """
    computes beta using gradient descent algorithm and computes list of costs
    :param training_dataset:
    :return: tuple containing beta (j*1 vector), costs list (list of costs in each iteration)
    """

    # attach a column of 1s to the beginning of x
    x = training_dataset[:, :-1]
    x = np.hstack((np.matrix(np.ones(x.shape[0])).T, x))

    # save targets in separate variables
    y = training_dataset[:, -1].reshape((x.shape[0], 1))

    beta = np.zeros((x.shape[1], 1))

    costs_list = []
    last_cost = math.inf
    current_cost = 0

    while abs(last_cost - current_cost) > THRESHOLD:
        beta = gradient_descent_step(x, y, beta)

        last_cost = current_cost
        current_cost = cost(x, y, beta)

        costs_list.append(current_cost)

    return beta, costs_list


def draw_costs_plot(costs_list):
    sns.lineplot(range(0, len(costs_list)), costs_list)
    plt.show()


def h_single(x, beta, threshold=0.5):
    """
    hypothesis function ran on single row
    :param x: a row of features (j * 1 vector)
    :param beta: j*1 vector
    :param threshold:
    :return: single value predicted for y
    """
    p = 1 / 1 + np.exp(1 / (1 + np.exp(-x.dot(beta))))[0, 0]

    if p > threshold:
        return 1
    else:
        return 0


def compute_error_rate(test_dataset, beta):
    """
    computes error rate of model with specified beta using test dataset
    :param test_dataset:
    :param beta:
    :return:
    """
    errors_count = 0
    for row in test_dataset:
        real_y = row[-1]
        x = np.insert(row[:-1], 0, 1)
        predicted_value = h_single(x, beta)

        if predicted_value != real_y:
            errors_count += 1

    return errors_count / len(test_dataset)


def start():
    """
    starting point of program
    :return: None
    """
    processing_type = input('Please Enter Processing Type (s: standard, l: log, b: binary) > ')

    dataset = load_dataset('spam.data', processing_type)

    training_dataset, test_dataset = generate_training_test_datasets(dataset)

    # get model information
    beta, costs_list = gradient_descent(training_dataset)

    # draw cost plot
    draw_costs_plot(costs_list)

    # compute error rate
    error_rate = compute_error_rate(test_dataset, beta)
    print('Error Rate is: {0}'.format(error_rate))


if __name__ == '__main__':
    start()
