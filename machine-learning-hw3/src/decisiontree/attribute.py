from abc import ABC, abstractmethod

import math


class Attribute(ABC):
    def __init__(self, name, all_classes):
        self.name = name
        self.all_classes = all_classes

    def get_all_classes(self):
        return self.all_classes

    @abstractmethod
    def get_class(self, sample):
        """
        get class of the input sample for the attribute
        :param sample:
        :return: class (int)
        """
        pass

    def filter(self, samples, cls):
        mask = [(self.get_class(x) == cls) for i, x in samples.iterrows()]
        return samples.loc[mask, :]


class AttributeFactory:
    @staticmethod
    def create_discrete(name, all_classes, get_class_fn):
        class DiscreteAttributeImpl(Attribute):
            def __init__(self):
                super().__init__(name, all_classes)

            def get_class(self, sample):
                return get_class_fn(sample)

        return DiscreteAttributeImpl()

    @staticmethod
    def create_continuous(attrname, samples, source_column, target_column):
        class ContinuousAttributeImpl(Attribute):
            def __init__(self, threshold):
                self.threshold = threshold
                name = '{0} < {1}'.format(attrname, threshold)
                super().__init__(name, ['Yes', 'No'])

            def get_class(self, sample):
                if sample[source_column] < self.threshold:
                    return 'Yes'

                return 'No'

        sorted_samples = samples.sort_values(source_column)
        attributes = []

        last_target = sorted_samples.iloc[0][target_column]
        last_source = sorted_samples.iloc[0][source_column]

        last_min = -math.inf

        for _, sample in sorted_samples.iterrows():
            if sample[target_column] != last_target:
                threshold = (sample[source_column] + last_source) / 2
                if threshold > last_min:
                    attribute = ContinuousAttributeImpl(threshold)
                    attributes.append(attribute)
                    last_min = threshold

            last_target = sample[target_column]
            last_source = sample[source_column]

        return attributes


class AttributeSelector(ABC):
    @abstractmethod
    def select(self, samples, attributes, target_attribute):
        pass


class InformationGainAttributeSelector(AttributeSelector):
    def _compute_entropy(self, samples, target_attribute):
        entropy = 0

        if len(samples) == 0:
            return entropy

        for cls in target_attribute.get_all_classes():
            p = len(target_attribute.filter(samples, cls)) / len(samples)
            if p != 0:
                entropy += -1 * p * math.log(p, 2)

        return entropy

    def select(self, samples, attributes, target_attribute):
        current_entropy = self._compute_entropy(samples, target_attribute)
        max_gain = (None, -math.inf)
        for attribute in attributes:
            total_entropy = 0
            for cls in attribute.get_all_classes():
                filtered_samples = attribute.filter(samples, cls)
                factor = len(filtered_samples) / len(samples)
                total_entropy += factor * self._compute_entropy(filtered_samples, target_attribute)

            total_entropy = current_entropy - total_entropy
            if total_entropy > max_gain[1]:
                max_gain = (attribute, total_entropy)

        return max_gain[0]


class GiniIndexAttributeSelector(AttributeSelector):
    def _compute_gindex(self, samples, target_attribute):
        gindex = 0

        if len(samples) == 0:
            return gindex

        for cls in target_attribute.get_all_classes():
            p = len(target_attribute.filter(samples, cls)) / len(samples)
            gindex += p ** 2

        return 1 - gindex

    def select(self, samples, attributes, target_attribute):
        current_gindex = self._compute_gindex(samples, target_attribute)
        max_gain = (None, -math.inf)
        for attribute in attributes:
            total_gindex = 0
            for cls in attribute.get_all_classes():
                filtered_samples = attribute.filter(samples, cls)
                factor = len(filtered_samples) / len(samples)
                total_gindex += factor * self._compute_gindex(filtered_samples, target_attribute)

            total_gindex = current_gindex - total_gindex
            if total_gindex > max_gain[1]:
                max_gain = (attribute, total_gindex)

        return max_gain[0]
