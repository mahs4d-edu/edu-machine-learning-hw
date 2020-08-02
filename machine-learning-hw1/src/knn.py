import math
import pandas
import plotly.express as px


def _compute_distance(d1, d2):
    """
    computes the euclidean distance between d1 and d2
    :param d1:
    :param d2:
    :return: euclidean distance
    """

    d = 0

    for i in range(4):
        d += (d2[i] - d1[i]) ** 2

    return math.sqrt(d)


def _add_if_less(k, l, d):
    """
    if d[0] is less than all other items in the list, adds it to the list and removes the maximum if the size is
    more than k
    :param k:
    :param l:
    :param d:
    :return:
    """

    for i, item in enumerate(l):
        if item[0] > d[0]:
            l.insert(i, d)
            break
    else:
        l.append(d)

    if len(l) > k:
        del l[len(l) - 1]


def classify(k, training_dataset, test_data):
    """
    returns classification of input test_data (single row) based on nearest k neighbors in training dataset
    :param k:
    :param training_dataset:
    :param test_data:
    :return: classification
    """

    neighbors = []

    # compute distance of test_data to each data in training dataset and add them to a min-list (the list only contains
    # k data with minimum distances
    for data in training_dataset:
        distance = _compute_distance(test_data, data)
        _add_if_less(k, neighbors, (distance, data))

    # compute count of classes in k nearest neighbors
    poll_result = {
        'Iris-setosa': 0,
        'Iris-versicolor': 0,
        'Iris-virginica': 0,
    }

    for neighbor in neighbors:
        # add to the property of dictionary with same name as the classification
        poll_result[neighbor[1][4]] += 1

    # return the class with maximum items
    return max(poll_result.keys(), key=(lambda key: poll_result[key]))


def get_error_rate(k, training_dataset, test_dataset):
    """
    returns the error_rate of knn with "k" argument on test_dataset using training dataset
    :param k:
    :param training_dataset:
    :param test_dataset:
    :return: error rate
    """

    errors_count = 0
    for test_data in test_dataset:
        # classify each data in test dataset
        predicted_class = classify(k, training_dataset, test_data)

        # if the predicted class is not equal to actual test data class, then add to errors count
        if predicted_class != test_data[4]:
            errors_count += 1

    # return errors_count / test_dataset_count as error rate
    return errors_count / len(test_dataset)


def find_best_k(training_dataset, test_dataset, start_k=3, end_k=15):
    """
    computes error rates of all ks between start_k and end_k
    :param training_dataset:
    :param test_dataset:
    :param start_k:
    :param end_k:
    :return: (best_k, all error rates with their respective k)
    """

    error_rates = []
    # get error rate of all ks and add it to an array
    for k in range(start_k, end_k + 1):
        error_rate = get_error_rate(k, training_dataset, test_dataset)
        error_rates.append((k, error_rate))

    return min(error_rates, key=(lambda i: i[1]))[0], error_rates


def draw_error_rates(error_rates):
    """
    draws error rates in respect of their k on chart
    :param error_rates:
    :return: None
    """

    # preparing data to feed to pandas datafram
    chart_data = []
    for er in error_rates:
        chart_data.append({'K': er[0], 'Error Rate': er[1]})

    dataframe = pandas.DataFrame(chart_data)
    fig = px.line(dataframe, x='K', y='Error Rate')
    fig.show()


def get_precisions_and_recalls(k, training_dataset, test_dataset):
    """
    calculates precisions and recalls using confusion matrix
    :param k:
    :param training_dataset:
    :param test_dataset:
    :return: precisions, recalls
    """

    # generating confusion matrix from prediction and expected values
    confusion_matrix = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

    classes = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
    for test_data in test_dataset:
        predicted_class = classify(k, training_dataset, test_data)
        i = classes.index(predicted_class)
        j = classes.index(test_data[4])
        confusion_matrix[i][j] += 1

    # add rows of confusion matrix to get precisions
    precisions = [0, 0, 0]
    for i in range(3):
        precision = 0
        for j in range(3):
            precision += confusion_matrix[j][i]

        precisions[i] = confusion_matrix[i][i] / precision

    # add columns of confusion matrix to get recalls
    recalls = [0, 0, 0]
    for i in range(3):
        recall = 0
        for j in range(3):
            recall += confusion_matrix[i][j]

        recalls[i] = confusion_matrix[i][i] / recall

    precisions = {
        'Iris-setosa': precisions[0],
        'Iris-versicolor': precisions[1],
        'Iris-virginica': precisions[2],
    }

    recalls = {
        'Iris-setosa': recalls[0],
        'Iris-versicolor': recalls[1],
        'Iris-virginica': recalls[2],
    }

    return precisions, recalls
