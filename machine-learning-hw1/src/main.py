import dataset
import knn


def main():
    # load dataset from iris.data
    print('>> Loading Main Dataset...')
    all_dataset = dataset.load_dataset_from_file('../data/iris.data')
    training_dataset = dataset.load_dataset_from_file('../data/training.data')
    test_dataset = dataset.load_dataset_from_file('../data/test.data')

    # show menu and get input
    print('1. Single KNN')
    print('2. Draw Data Distributions and Show Data Statistics')
    print('3. Generate New Test and Training Data')
    print('4. Find Best K Between 3 and 15 and Get Precision and Recall of It')
    print('5. Apply PCA to Data and Draw its 2D Projection')

    choice = input('What Do You Want? ')
    if choice == '1':
        k = int(input('Enter Your K: '))

        data = input('Enter Data (4 features separated by comma): ').split(',')

        for i in range(4):
            data[i] = float(data[i])

        cl = knn.classify(k, training_dataset, data)

        print('Your Record Class Is: "{0}"'.format(cl))
    elif choice == '2':
        averages, variances, skewness = dataset.get_data_statistics(all_dataset)
        print('Averages: {0}'.format(averages))
        print('Variances: {0}'.format(variances))
        print('Skewness: {0}'.format(skewness))

        dataset.draw_dataset_distribution(all_dataset)
    elif choice == '3':
        gen_training_dataset, gen_test_dataset = dataset.generate_test_and_training_datasets(all_dataset, 20)

        dataset.write_dataset_to_file('../data/gen_test.data', gen_training_dataset)
        dataset.write_dataset_to_file('../data/gen_test.data', gen_test_dataset)

        print('>> gen_test.data and gen_training.data Files were Generated in "data" Folder')
    elif choice == '4':
        best_k, error_rates = knn.find_best_k(training_dataset, test_dataset, 3, 15)
        print('Best K is {0}'.format(best_k))

        precisions, recalls = knn.get_precisions_and_recalls(best_k, training_dataset, test_dataset)

        for i in precisions:
            print('Precision[{0}]: {1}'.format(i, precisions[i]))

        for i in recalls:
            print('Recall[{0}]: {1}'.format(i, recalls[i]))

        knn.draw_error_rates(error_rates)
    elif choice == '5':
        dataset.draw_pca(all_dataset)


main()
