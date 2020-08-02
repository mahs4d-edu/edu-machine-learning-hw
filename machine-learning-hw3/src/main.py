from os import path

import math

import decisiontree as dt
from dataset import Dataset

print('loading dataset ...')
file_location = path.join(path.abspath(
    path.dirname(__file__)), '../data', 'cleveland.data')
dataset = Dataset(file_location)

dataset.draw_distribution('age')
dataset.draw_distribution('sex')
dataset.draw_distribution('cp')
dataset.draw_distribution('chol')
dataset.draw_distribution('trestbps')
dataset.draw_distribution('fbs')
dataset.draw_distribution('restecg')
dataset.draw_distribution('thalach')
dataset.draw_distribution('exang')
dataset.draw_distribution('oldpeak')
dataset.draw_distribution('slope')
dataset.draw_distribution('ca')
dataset.draw_distribution('thai')

attributes_age = dt.AttributeFactory.create_continuous('age', dataset.get_training_dataset(), 'age', 'num')

attribute_sex = dt.AttributeFactory.create_discrete('sex', [0, 1], lambda x: x['sex'])
attribute_cp = dt.AttributeFactory.create_discrete('cp', [1, 2, 3, 4], lambda x: x['cp'])

attributes_chol = dt.AttributeFactory.create_continuous('chol', dataset.get_training_dataset(), 'chol', 'num')
attributes_trestbps = dt.AttributeFactory.create_continuous('trestbps', dataset.get_training_dataset(), 'trestbps',
                                                            'num')

attribute_fbs = dt.AttributeFactory.create_discrete('fbs', [0, 1], lambda x: x['fbs'])
attribute_restecg = dt.AttributeFactory.create_discrete('restecg', [0, 2], lambda x: x['restecg'])

attributes_thalach = dt.AttributeFactory.create_continuous('thalach', dataset.get_training_dataset(), 'thalach', 'num')

attribute_exang = dt.AttributeFactory.create_discrete('exang', [0, 1], lambda x: x['exang'])

attributes_oldpeak = dt.AttributeFactory.create_continuous('oldpeak', dataset.get_training_dataset(), 'oldpeak', 'num')

attribute_slope = dt.AttributeFactory.create_discrete('slope', [1, 2, 3, 4], lambda x: x['slope'])
attribute_ca = dt.AttributeFactory.create_discrete('ca', [0, 1, 2, 3], lambda x: x['ca'])
attribute_thai = dt.AttributeFactory.create_discrete('thai', [3, 4, 5, 6, 7], lambda x: x['thai'])

attribute_num = dt.AttributeFactory.create_discrete('num', [0, 1], lambda x: x['num'])

print('generating decision tree and pruning (using kfold) ...')
attribute_selector = dt.GiniIndexAttributeSelector()
attributes_list = [attribute_sex, attribute_cp, attribute_fbs, attribute_restecg,
                   attribute_exang, attribute_slope, attribute_ca, attribute_thai]
attributes_list.extend(attributes_age)
attributes_list.extend(attributes_chol)
attributes_list.extend(attributes_trestbps)
attributes_list.extend(attributes_thalach)
attributes_list.extend(attributes_oldpeak)

min_error_tree = (math.inf, None)
for training_indices, validation_indices in dataset.get_cross_validation_indexes():
    dtree = dt.DecisionTreeFactory.create(dataset.get_training_dataset().iloc[training_indices, :],
                                          attributes_list,
                                          attribute_num, attribute_selector)
    dtree.prune(dataset.get_training_dataset().iloc[validation_indices, :])
    error_rate = dtree.evaluate(dataset.get_training_dataset().iloc[validation_indices, :])
    if error_rate <= min_error_tree[0]:
        min_error_tree = (error_rate, dtree)

error_rate = dtree.evaluate(dataset.test_dataset)
print('error rate is {0}'.format(error_rate))

min_error_tree[1].draw('Cleveland Heart Diseases')
