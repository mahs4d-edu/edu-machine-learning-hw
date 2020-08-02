import random

import math
import pandas
import plotly.express as px
from plotly import figure_factory
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def load_dataset_from_file(file_location):
    """
    loads a dataset from input file and saves it in an array
    :param file_location:
    :return: dataset array
    """

    dataset = []

    # open file (with closes the file when block finishes)
    with open(file_location, 'r') as file:
        # read first line for while initial condition (rstrip removes \n in the end)
        line = file.readline().rstrip('\n')
        while line:
            # split line string to array of data
            raw_data = line.split(',')

            # convert first 4 features to float from string
            for i in range(4):
                raw_data[i] = float(raw_data[i])

            # add data to array
            dataset.append(tuple(raw_data))

            # read next line
            line = file.readline().rstrip('\n')

    return dataset


def write_dataset_to_file(file_location, dataset):
    """
    writes input dataset to an output file
    :param file_location:
    :param dataset:
    :return: None
    """

    with open(file_location, 'w') as file:
        for data in dataset:
            file.write('{0},{1},{2},{3},{4}\n'.format(*data))


def generate_test_and_training_datasets(dataset, test_proportion=20):
    """
    partitions input dataset to two separate datasets. the proportion is 20% by default for test data
    and 80% for training
    :param dataset:
    :param test_proportion:
    :return: training_dataset, test_dataset
    """

    # compute test data length
    dataset_length = len(dataset)
    test_count = int((test_proportion * dataset_length) / 100)

    # copy dataset to another variable so that the main dataset remain intact
    training_dataset = dataset.copy()
    test_dataset = []

    for i in range(test_count):
        # generate a random number between 0 and length of remaining training data
        t = random.randrange(0, len(training_dataset))
        # move the data from training dataset to test dataset
        test_dataset.append(training_dataset[t])
        del training_dataset[t]

    return training_dataset, test_dataset


def draw_dataset_distribution(dataset):
    """
    draws the distribution chart of data
    :param dataset:
    :return: None
    """
    # preparing data for plot.ly figure_factory distplot
    # the data should be an array of 4 each containing arrays of dataset features
    chart_data = [[], [], [], []]
    for i, er in enumerate(dataset):
        chart_data[0].append(dataset[i][0])
        chart_data[1].append(dataset[i][1])
        chart_data[2].append(dataset[i][2])
        chart_data[3].append(dataset[i][3])

    # data labels
    labels = ['sepal length', 'sepal width', 'petal length', 'petal width']

    # generate the figure
    fig = figure_factory.create_distplot(chart_data, labels, bin_size=0.2)
    fig.show()


def get_data_statistics(dataset):
    """
    computes feature average, variance and skewness
    :param dataset:
    :return: averages, variances, skewness
    """

    # average calculations
    averages = [0, 0, 0, 0]
    for d in dataset:
        averages[0] += d[0]
        averages[1] += d[1]
        averages[2] += d[2]
        averages[3] += d[3]

    averages[0] = averages[0] / len(dataset)
    averages[1] = averages[1] / len(dataset)
    averages[2] = averages[2] / len(dataset)
    averages[3] = averages[3] / len(dataset)

    # variance calculations
    variances = [0, 0, 0, 0]
    for d in dataset:
        variances[0] += (d[0] - averages[0]) ** 2
        variances[1] += (d[1] - averages[1]) ** 2
        variances[2] += (d[2] - averages[2]) ** 2
        variances[3] += (d[3] - averages[3]) ** 2

    variances[0] = variances[0] / len(dataset)
    variances[1] = variances[1] / len(dataset)
    variances[2] = variances[2] / len(dataset)
    variances[3] = variances[3] / len(dataset)

    # median computation
    temp_dataset = dataset.copy()
    medians = []

    for i in range(4):
        temp_dataset = sorted(temp_dataset, key=lambda x: x[i])
        medians.append(temp_dataset[math.floor(len(temp_dataset) / 2)][i])

    # skewness computations
    skewness = [0, 0, 0, 0]
    for d in dataset:
        skewness[0] += (d[0] - medians[0]) ** 3
        skewness[1] += (d[1] - medians[1]) ** 3
        skewness[2] += (d[2] - medians[2]) ** 3
        skewness[3] += (d[3] - medians[3]) ** 3

    skewness[0] = skewness[0] / ((len(dataset) + 1) * (variances[0] ** 3))
    skewness[1] = skewness[1] / ((len(dataset) + 1) * (variances[1] ** 3))
    skewness[2] = skewness[2] / ((len(dataset) + 1) * (variances[2] ** 3))
    skewness[3] = skewness[3] / ((len(dataset) + 1) * (variances[3] ** 3))

    return averages, variances, skewness


def draw_pca(dataset):
    chart_data = []
    for i, data in enumerate(dataset):
        chart_data.append(
            {
                'sepal_length': data[0],
                'sepal_width': data[1],
                'petal_length': data[2],
                'petal_width': data[3],
                'target': data[4],
            }
        )

    features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    dataframe = pandas.DataFrame(chart_data)

    x = dataframe.loc[:, features].values
    y = dataframe.loc[:, ['target']].values

    x = StandardScaler().fit_transform(x)

    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(x)
    principal_df = pandas.DataFrame(data=principal_components,
                                    columns=['principal component 1', 'principal component 2'])
    final_df = pandas.concat([principal_df, dataframe[['target']]], axis=1)

    fig = px.scatter(final_df, x="principal component 1", y="principal component 2", color='target')
    fig.show()
