import random
from abc import ABC, abstractmethod
from os import path

import graphviz


class DTreeNode(ABC):
    @abstractmethod
    def classify(self, sample):
        pass


class DTreeLink:
    def __init__(self, attribute_class, node):
        self.attribute_class = attribute_class
        self.node = node


class DTreeDecisionNode(DTreeNode):
    def __init__(self, attribute, links, predominant_target_value):
        self.attribute = attribute
        self.links = links
        self.predominant_target_value = predominant_target_value
        self.is_active = True

    def classify(self, sample):
        """
        checks all the link values with the sample input and then
        :param sample:
        :return:
        """
        if not self.is_active:
            return self.predominant_target_value

        sample_attribute_class = self.attribute.get_class(sample)

        for link in self.links:
            if link.attribute_class == sample_attribute_class:
                return link.node.classify(sample)

        raise ValueError('the sample {0} does not belong to any class of {1}'.format(str(sample), self.attribute.name))

    def deactivate(self):
        self.is_active = False

    def activate(self):
        self.is_active = True


class DTreeTerminalNode(DTreeNode):
    def __init__(self, target_value):
        self.target_value = target_value

    def classify(self, sample):
        """
        simply returns the target value for all samples as this is a leaf node
        :param sample:
        :return: target value
        """
        return self.target_value


class DecisionTree:
    def __init__(self, root_node, target_attribute):
        self.root_node = root_node
        self.target_attribute = target_attribute

    def classify(self, sample):
        """
        classifies input sample with this decision tree
        :param sample:
        :return: predicted target attribute value
        """
        return self.root_node.classify(sample)

    def evaluate(self, test_samples):
        errors_count = 0
        for _, sample in test_samples.iterrows():
            try:
                predicted_value = self.classify(sample)
                if predicted_value != self.target_attribute.get_class(sample):
                    errors_count += 1
            except ValueError:
                errors_count += 1

        return errors_count / len(test_samples)

    def _draw_node(self, graph, node):
        nname = str(random.randrange(1, 1000000))
        if isinstance(node, DTreeDecisionNode):
            if node.is_active:
                graph.attr('node', shape='box')
                graph.node(name=nname, label=node.attribute.name)
                for link in node.links:
                    child_lbl = self._draw_node(graph, link.node)
                    graph.edge(nname, child_lbl, str(link.attribute_class))
            else:
                graph.attr('node', shape='diamond')
                graph.node(name=nname, label=str(node.predominant_target_value))
        else:
            graph.attr('node', shape='ellipse')
            graph.node(name=nname, label=str(node.target_value))

        return nname

    def draw(self, name):
        file_path = path.join(path.abspath(path.dirname(__file__)), '../../docs', 'dtree.gv')
        graph = graphviz.Graph(name, filename=file_path)

        self._draw_node(graph, self.root_node)

        graph.view()

    def _get_all_nodes(self, n, parent_node, include_decision=True, include_terminal=False,
                       include_inactive=False):
        nodes = []
        if isinstance(parent_node, DTreeDecisionNode):
            if not parent_node.is_active and not include_inactive:
                return nodes

            if include_decision:
                nodes.append((n, parent_node))

            for link in parent_node.links:
                nodes.extend(self._get_all_nodes(n + 1, link.node, include_decision, include_terminal,
                                                 include_inactive))

        elif include_terminal:
            nodes.append((n, parent_node))

        return nodes

    def prune(self, validation_samples):
        all_nodes = self._get_all_nodes(0, self.root_node, True, False, False)
        all_nodes = sorted(all_nodes, key=lambda x: x[0], reverse=True)
        error_rate = self.evaluate(validation_samples)

        for _, node in all_nodes:
            node.deactivate()
            pruned_error_rate = self.evaluate(validation_samples)

            if pruned_error_rate > error_rate:
                node.activate()
            else:
                error_rate = pruned_error_rate


class DecisionTreeFactory:
    @staticmethod
    def _create_node(samples, attributes, target_attribute, attribute_selector):
        if len(samples) == 0:
            return DTreeTerminalNode('X')

        tattr = samples.loc[:, target_attribute.name].mode().iloc[0]

        # if there is no more attributes -> return terminal with most abundant target value
        if len(attributes) == 0:
            return DTreeTerminalNode(tattr)

        # if all the samples have same target_attribute -> return terminal node with target value
        if samples.loc[:, target_attribute.name].nunique() == 1:
            return DTreeTerminalNode(tattr)

        # select best attribute
        selected_attribute = attribute_selector.select(samples, attributes, target_attribute)
        attributes.remove(selected_attribute)

        # create sub nodes
        links = []
        for attribute_class in selected_attribute.get_all_classes():
            child_node = DecisionTreeFactory._create_node(selected_attribute.filter(samples, attribute_class),
                                                          attributes[:], target_attribute, attribute_selector)
            link = DTreeLink(attribute_class, child_node)
            links.append(link)

        return DTreeDecisionNode(selected_attribute, links, tattr)

    @staticmethod
    def create(samples, attributes, target_attribute, attribute_selector):
        root_node = DecisionTreeFactory._create_node(samples, attributes, target_attribute, attribute_selector)
        return DecisionTree(root_node, target_attribute)
