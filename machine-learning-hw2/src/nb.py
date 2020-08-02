from __future__ import division

import os
import math
from collections import defaultdict

import tokenizer

# region general constants
POS_LABEL = 'pos'
NEG_LABEL = 'neg'
# endregion

# region dataset constants
PATH_TO_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data')
TRAIN_DIR = os.path.join(PATH_TO_DATA, "train")
TEST_DIR = os.path.join(PATH_TO_DATA, "test")


# endregion

def tokenize_doc(doc):
    mappings = dict()

    for token in tokenizer.tokenize(text=doc):
        kind, txt, val = token

        if kind == tokenizer.TOK.WORD:
            txt = txt.lower()
            if txt in mappings:
                mappings[txt] += 1
            else:
                mappings[txt] = 1

    return mappings


class NaiveBayes:
    """A Naive Bayes model for text classification."""

    def __init__(self):
        # Vocabulary is a set that stores every word seen in the training data
        self.vocab = set()

        # class_total_doc_counts is a dictionary that maps a class (i.e., pos/neg) to
        # the number of documents in the trainning set of that class
        self.class_total_doc_counts = {POS_LABEL: 0.0,
                                       NEG_LABEL: 0.0}

        # class_total_word_counts is a dictionary that maps a class (i.e., pos/neg) to
        # the number of words in the training set in documents of that class
        self.class_total_word_counts = {POS_LABEL: 0.0,
                                        NEG_LABEL: 0.0}

        # class_word_counts is a dictionary of dictionaries. It maps a class (i.e.,
        # pos/neg) to a dictionary of word counts. For example:
        #    self.class_word_counts[POS_LABEL]['awesome']
        # stores the number of times the word 'awesome' appears in documents
        # of the positive class in the training documents.
        self.class_word_counts = {POS_LABEL: defaultdict(float),
                                  NEG_LABEL: defaultdict(float)}

    def train_model(self, num_docs=None):
        """
        This function processes the entire training set using the global PATH
        variable above.  It makes use of the tokenize_doc and update_model
        functions you will implement.

        num_docs: set this to e.g. 10 to train on only 10 docs from each category.
        """

        if num_docs is not None:
            print("Limiting to only %s docs per clas" % num_docs)

        pos_path = os.path.join(TRAIN_DIR, POS_LABEL)
        neg_path = os.path.join(TRAIN_DIR, NEG_LABEL)
        print("Starting training with paths %s and %s" % (pos_path, neg_path))
        for (p, label) in [(pos_path, POS_LABEL), (neg_path, NEG_LABEL)]:
            filenames = os.listdir(p)

            if num_docs is not None:
                filenames = filenames[:num_docs]

            for f in filenames:
                with open(os.path.join(p, f), 'r') as doc:
                    content = doc.read()
                    self.tokenize_and_update_model(content, label)

        self.report_statistics_after_training()

    def report_statistics_after_training(self):
        """
        Report a number of statistics after training.
        """

        print("REPORTING CORPUS STATISTICS")
        print("NUMBER OF DOCUMENTS IN POSITIVE CLASS:", self.class_total_doc_counts[POS_LABEL])
        print("NUMBER OF DOCUMENTS IN NEGATIVE CLASS:", self.class_total_doc_counts[NEG_LABEL])
        print("NUMBER OF TOKENS IN POSITIVE CLASS:", self.class_total_word_counts[POS_LABEL])
        print("NUMBER OF TOKENS IN NEGATIVE CLASS:", self.class_total_word_counts[NEG_LABEL])
        print("VOCABULARY SIZE: NUMBER OF UNIQUE WORDTYPES IN TRAINING CORPUS:", len(self.vocab))

    def update_model(self, bow, label):
        # increment number of docs for specific label by 1 for each time this function is called
        self.class_total_doc_counts[label] += 1

        # increment number of words in specific label by the size of input bag of words
        self.class_total_word_counts[label] += len(bow.keys())

        # add each word to the vocab set and add it's count to class_word_counts[label]
        for word in bow.keys():
            self.vocab.add(word)
            if word in self.class_word_counts[label]:
                self.class_word_counts[label][word] += bow[word]
            else:
                self.class_word_counts[label][word] = bow[word]

    def tokenize_and_update_model(self, doc, label):
        """
        Tokenizes a document doc and updates internal count statistics.
        doc - a string representing a document.
        label - the sentiment of the document (either postive or negative)
        stop_word - a boolean flag indicating whether to stop word or not

        Make sure when tokenizing to lower case all of the tokens!
        """

        bow = tokenize_doc(doc)
        self.update_model(bow, label)

    def top_n(self, label, n):
        # first sort class_word_counts of input label
        sorted_values = sorted(self.class_word_counts[label].items(), key=lambda kv: kv[1], reverse=True)

        most_frequent_words = []
        for i in range(n):
            most_frequent_words.append(sorted_values[i])

        return most_frequent_words

    def p_word_given_label(self, word, label):
        # divide count of word in that class to the total number of words in that class
        total_class_word_count = sum(self.class_word_counts[label].values())
        return self.class_word_counts[label][word] / total_class_word_count

    def p_word_given_label_and_psuedocount(self, word, label, alpha):
        total_class_word_count = sum(self.class_word_counts[label].values())

        # add alpha to numerator
        numerator = self.class_word_counts[label][word] + alpha
        # add alpha * number of words in that class to denominator
        denominator = total_class_word_count + (alpha * len(self.class_word_counts[label].keys()))

        return numerator / denominator

    def log_likelihood(self, bow, label, alpha):
        log_lh = 0
        for word in bow.keys():
            p = self.p_word_given_label_and_psuedocount(word, label, alpha)
            log_lh += math.log(p)

        return log_lh

    def log_prior(self, label):
        # divide number of documents with this label to total documents count
        p = self.class_total_doc_counts[label] / sum(self.class_total_doc_counts.values())
        # return logarithm of probability
        return math.log(p)

    def unnormalized_log_posterior(self, bow, label, alpha):
        return self.log_prior(label) + self.log_likelihood(bow, label, alpha)

    def classify(self, bow, alpha):
        positive_unnormalized_lp = self.unnormalized_log_posterior(bow, POS_LABEL, alpha)
        negative_unnormalized_lp = self.unnormalized_log_posterior(bow, NEG_LABEL, alpha)

        if positive_unnormalized_lp >= negative_unnormalized_lp:
            return POS_LABEL
        else:
            return NEG_LABEL

    def likelihood_ratio(self, word, alpha):
        numerator = self.p_word_given_label_and_psuedocount(word, POS_LABEL, alpha)
        denominator = self.p_word_given_label_and_psuedocount(word, NEG_LABEL, alpha)

        return numerator / denominator

    def evaluate_classifier_accuracy(self, alpha):
        # compute pos and neg paths for test data
        pos_path = os.path.join(TEST_DIR, POS_LABEL)
        neg_path = os.path.join(TEST_DIR, NEG_LABEL)

        correct_predictions = 0
        total_predictions = 0

        for (p, label) in [(pos_path, POS_LABEL), (neg_path, NEG_LABEL)]:
            filenames = os.listdir(p)

            for f in filenames:
                with open(os.path.join(p, f), 'r') as doc:
                    content = doc.read()

                    # tokenize
                    bow = tokenize_doc(content)

                    # classify
                    prediction = self.classify(bow, alpha)

                    total_predictions += 1

                    # add to correct predictions if it was right
                    if prediction == label:
                        correct_predictions += 1

        return correct_predictions / total_predictions


def plot_psuedocount_vs_accuracy(psuedocounts, accuracies):
    """
    A function to plot psuedocounts vs. accuracies. You may want to modify this function
    to enhance your plot.
    """

    import matplotlib.pyplot as plt

    plt.plot(psuedocounts, accuracies)
    plt.xlabel('Psuedocount Parameter')
    plt.ylabel('Accuracy (%)')
    plt.title('Psuedocount Parameter vs. Accuracy Experiment')
    plt.show()


def produce_hw2_results():
    psuedocounts = [0.01, 0.1, 1, 10, 100]
    accuracies = []
    for alpha in psuedocounts:
        print('Evaluating for Alpha = {0}'.format(alpha))
        accuracy = nb.evaluate_classifier_accuracy(alpha)
        print('Accuracy for Alpha = {0} is {1}'.format(alpha, accuracy))
        accuracies.append(accuracy)

    plot_psuedocount_vs_accuracy(psuedocounts, accuracies)


if __name__ == '__main__':
    nb = NaiveBayes()

    print('Training Model...')
    nb.train_model()
    produce_hw2_results()
