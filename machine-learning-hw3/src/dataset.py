import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import seaborn as sns


class Dataset:
    def __init__(self, dataset_file_location):
        self.dataset_file_location = dataset_file_location
        self._load_dataset_from_file()
        self._clean()
        self._generate_training_and_test()

    def _load_dataset_from_file(self):
        self.main_dataset = pd.read_csv(self.dataset_file_location, header=None, names=[
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca',
            'thai', 'num'], index_col=False)

    def _clean(self):
        self.main_dataset = self.main_dataset[self.main_dataset['num'] < 2]
        self.main_dataset.fillna(self.main_dataset.mean(), inplace=True)

    def draw_distribution(self, column):
        t0_dataset = self.main_dataset.loc[self.main_dataset['num'] == 0].loc[:, column]
        t1_dataset = self.main_dataset.loc[self.main_dataset['num'] == 1].loc[:, column]

        sns.distplot(a=t0_dataset, hist=True, rug=True, label='0')
        sns.distplot(a=t1_dataset, hist=True, rug=True, label='1')
        plt.show()

    def _generate_training_and_test(self):
        mask = np.random.rand(len(self.main_dataset)) < 0.8
        self.training_dataset = self.main_dataset[mask]
        self.test_dataset = self.main_dataset[~mask]
        self.validation_folds = KFold(n_splits=5, shuffle=True)

    def get_main_dataset(self):
        return self.main_dataset

    def get_training_dataset(self):
        return self.training_dataset

    def get_test_dataset(self):
        return self.test_dataset

    def get_cross_validation_indexes(self):
        return self.validation_folds.split(self.training_dataset)
