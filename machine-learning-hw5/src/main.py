from os import path

import math
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler


def load_dataset(filename, names):
    p = path.join(path.abspath(path.dirname(__file__)), '../data/', filename)

    if names is None:
        return pd.read_csv(p)
    else:
        return pd.read_csv(p, header=None, names=names)


def create_lda_model(training_dataset, classes_list, last_vals_index, class_column):
    model = {}
    cov = np.zeros(shape=(training_dataset.shape[1] - 1, training_dataset.shape[1] - 1))
    for cls in classes_list:
        filtered_data = training_dataset[training_dataset.loc[:, class_column] == cls].iloc[:, :last_vals_index]
        pi = filtered_data.shape[0] / training_dataset.shape[0]
        mean = filtered_data.mean().values

        pi_key = 'pi_{0}'.format(cls)
        mean_key = 'u_{0}'.format(cls)

        cov += (filtered_data.cov() * (filtered_data.shape[0] - 1)) / (
                filtered_data.shape[0] - training_dataset.shape[1] - 1)

        model[pi_key] = pi
        model[mean_key] = mean

    model['covinv'] = np.linalg.pinv(cov)
    return model


def lda_classify(sample, model, classes_list):
    s_list = []

    for cls in classes_list:
        pi_key = 'pi_{0}'.format(cls)
        mean_key = 'u_{0}'.format(cls)

        x = sample
        u = model[mean_key]
        e = model['covinv']  # np.eye(u.shape[0])
        pi = math.log(model[pi_key])

        s = np.matmul(np.matmul(x.T, e), u) - 0.5 * np.matmul(np.matmul(u.T, e), u) + pi

        s_list.append((cls, s))

    return max(s_list, key=lambda k: k[1])[0]


def test_lda_model(model, test_dataset, last_vals_index, classes_list, class_column):
    correct_answers = 0

    for index, sample in test_dataset.iterrows():
        x = sample.iloc[0:last_vals_index].values
        actual_class = sample.loc[class_column]
        predicted_class = lda_classify(x, model, classes_list)

        if actual_class == predicted_class:
            correct_answers += 1

    return correct_answers / test_dataset.shape[0]


def pca(dataset, last_vals_index, class_column, features_count=2):
    """
    dimentionality reduction using pca algorithm
    :param dataset:
    :param last_vals_index:
    :param classes_list:
    :param class_column:
    :return:
    """
    y = dataset.loc[:, class_column]
    x = dataset.iloc[:, :last_vals_index].values
    x_std = StandardScaler().fit_transform(x)

    # compute covariance matrix
    cov_matrix = np.cov(x_std.T)

    # eigenvectors and eigenvalues for the from the covariance matrix
    eig_vals, eig_vecs = np.linalg.eig(cov_matrix)

    # Make a list of (eigenvalue, eigenvector) tuples
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:, i]) for i in range(len(eig_vals))]

    # Sort the (eigenvalue, eigenvector) tuples from high to low
    eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)

    # select top k eigen vectors to generate the V vector of transformation
    e_list = []
    for i in range(features_count):
        e_list.append(eig_pairs[i][1].reshape(last_vals_index, 1))

    v = np.hstack(tuple(e_list))

    # transform x
    x_ld = x_std.dot(v)

    # create dataframe and return it
    columns = []
    for t in range(features_count):
        columns.append('prop{0}'.format(t))

    df = pd.DataFrame(x_ld, columns=columns)
    df[class_column] = dataset[class_column]

    return v, df


def fda(dataset, last_vals_index, classes_list, class_column, features_count=2):
    """
    dimentionality reduction using fda algorithm
    :param dataset:
    :param last_vals_index:
    :param classes_list:
    :param class_column:
    :return:
    """
    y = dataset[class_column].values
    x = dataset.iloc[:, :last_vals_index].values

    # compute mean vector
    counts = {}
    means_vectors = {
        'all': dataset.iloc[:, :last_vals_index].mean().values.reshape((last_vals_index, 1))
    }

    for cls in classes_list:
        a = dataset[dataset[class_column] == cls].iloc[:, :last_vals_index]
        counts[cls] = a.shape[0]
        means_vectors[cls] = a.mean().values.reshape((last_vals_index, 1))

    # compute within class scatter
    s_w = np.zeros((last_vals_index, last_vals_index))
    for cls in classes_list:
        p1 = x[y == cls] - means_vectors[cls].T
        p2 = p1.T
        s_w += p2.dot(p1)

    # compute between class scatter
    s_b = np.zeros((last_vals_index, last_vals_index))
    for cls in classes_list:
        n = counts[cls]
        p1 = means_vectors[cls] - means_vectors['all']
        p2 = p1.T
        s_b += n * p1.dot(p2)

    # compute eigenvalues and vectors
    eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(s_w).dot(s_b))

    # Make a list of (eigenvalue, eigenvector) tuples
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:, i]) for i in range(len(eig_vals))]

    # Sort the (eigenvalue, eigenvector) tuples from high to low
    eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)

    # select top k eigen vectors to generate the V vector of transformation
    e_list = []
    for i in range(features_count):
        e_list.append(eig_pairs[i][1].reshape(last_vals_index, 1))

    v = np.hstack(tuple(e_list))

    # transform x
    x_ld = x.dot(v)

    # create dataframe and return it
    columns = []
    for t in range(features_count):
        columns.append('prop{0}'.format(t))

    df = pd.DataFrame(x_ld, columns=columns)
    df[class_column] = dataset[class_column]

    return v, df


def transform(dataset, v, last_vals_index, class_column, features_count=2):
    x = dataset.iloc[:, :last_vals_index].values

    # transform x
    x_ld = x.dot(v)

    # create dataframe and return it
    columns = []
    for t in range(features_count):
        columns.append('prop{0}'.format(t))

    df = pd.DataFrame(x_ld, columns=columns)
    df[class_column] = dataset[class_column]

    return df


def question2_a():
    """
    pass iris dataset directly to lda for classification (without any preprocessing)
    :return:
    """
    # load iris dataset and a all one column at the beginning
    dataset = load_dataset('iris.csv', ['sl', 'sw', 'pl', 'pw', 'cls'])
    dataset.insert(0, 'one', value=1)

    # split to training and test datasets
    training_dataset = dataset.sample(frac=0.8, random_state=200)
    test_dataset = dataset.drop(training_dataset.index)

    # create lda model for raw iris data
    model = create_lda_model(training_dataset, ['Iris-setosa', 'Iris-virginica', 'Iris-versicolor'], 5, 'cls')
    accuracy = test_lda_model(model, test_dataset, 5, ['Iris-setosa', 'Iris-virginica', 'Iris-versicolor'], 'cls')

    print('Accuracy for UnProcessed Iris Dataset: {0}'.format(accuracy))


def question2_b():
    """
        first apply fda on iris dataset and then pass it to lda
        :return:
    """

    # load iris dataset
    dataset = load_dataset('iris.csv', ['sl', 'sw', 'pl', 'pw', 'cls'])

    # pass it to pca
    v, dataset = pca(dataset, 4, 'cls', 2)

    # split to training and test datasets
    dataset.insert(0, 'one', value=1)

    training_dataset = dataset.sample(frac=0.8, random_state=200)
    test_dataset = dataset.drop(training_dataset.index)

    # create lda model for raw iris data
    model = create_lda_model(training_dataset, ['Iris-setosa', 'Iris-virginica', 'Iris-versicolor'], 3, 'cls')
    accuracy = test_lda_model(model, test_dataset, 3, ['Iris-setosa', 'Iris-virginica', 'Iris-versicolor'], 'cls')

    print('Accuracy for Iris Dataset with PCA applied: {0}'.format(accuracy))


def question2_c():
    """
    first apply fda on iris dataset and then pass it to lda
    :return:
    """

    # load iris dataset
    dataset = load_dataset('iris.csv', ['sl', 'sw', 'pl', 'pw', 'cls'])

    # pass it to fda
    v, dataset = fda(dataset, 4, ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'], 'cls')

    # split to training and test datasets
    dataset.insert(0, 'one', value=1)

    training_dataset = dataset.sample(frac=0.8, random_state=200)
    test_dataset = dataset.drop(training_dataset.index)

    # create lda model for raw iris data
    model = create_lda_model(training_dataset, ['Iris-setosa', 'Iris-virginica', 'Iris-versicolor'], 3, 'cls')
    accuracy = test_lda_model(model, test_dataset, 3, ['Iris-setosa', 'Iris-virginica', 'Iris-versicolor'], 'cls')

    print('Accuracy for Iris Dataset with FDA applied: {0}'.format(accuracy))


def question2_e():
    # load vowel datasets
    training_dataset = load_dataset('vowel_train.csv', None)[[
        'x.1', 'x.2', 'x.3', 'x.4', 'x.5', 'x.6', 'x.7', 'x.8', 'x.9', 'x.10', 'y']]
    test_dataset = load_dataset('vowel_test.csv', None)[[
        'x.1', 'x.2', 'x.3', 'x.4', 'x.5', 'x.6', 'x.7', 'x.8', 'x.9', 'x.10', 'y']]

    # create the full dataset
    dataset = training_dataset.append(test_dataset).reset_index()

    # apply fda and pca transforms
    pca_k = 2
    fda_k = 2

    v_pca, _ = pca(training_dataset, 10, 'y', pca_k)
    v_fda, _ = fda(training_dataset, 10, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], 'y', fda_k)

    training_dataset_pca = transform(training_dataset, v_pca, 10, 'y', pca_k)
    test_dataset_pca = transform(test_dataset, v_pca, 10, 'y', pca_k)

    training_dataset_fda = transform(training_dataset, v_fda, 10, 'y', fda_k)
    test_dataset_fda = transform(test_dataset, v_fda, 10, 'y', fda_k)

    dataset.insert(0, 'one', value=1)
    training_dataset.insert(0, 'one', value=1)
    test_dataset.insert(0, 'one', value=1)
    training_dataset_pca.insert(0, 'one', value=1)
    test_dataset_pca.insert(0, 'one', value=1)
    training_dataset_fda.insert(0, 'one', value=1)
    test_dataset_fda.insert(0, 'one', value=1)

    # raw --------------
    model = create_lda_model(training_dataset, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], 11, 'y')
    accuracy = test_lda_model(model, test_dataset, 11, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], 'y')

    print('Accuracy for UnProcessed Vowel Dataset: {0}'.format(accuracy))

    # pca --------------
    model = create_lda_model(training_dataset_pca, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], pca_k + 1, 'y')
    accuracy = test_lda_model(model, test_dataset_pca, pca_k + 1, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], 'y')

    print('Accuracy for Vowel Dataset with PCA applied: {0}'.format(accuracy))

    # fda --------------
    model = create_lda_model(training_dataset_fda, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], fda_k + 1, 'y')
    accuracy = test_lda_model(model, test_dataset_fda, fda_k + 1, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], 'y')

    print('Accuracy for Vowel Dataset with FDA applied: {0}'.format(accuracy))


question2_a()
question2_b()
question2_c()
print('-' * 40)
question2_e()
