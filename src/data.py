# MIT License
#
# Copyright (c) 2017 Jonatan Almén, Alexander Håkansson, Jesper Jaxing, Gmal
# Tchaefa, Maxim Goretskyy, Axel Olivecrona
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ==============================================================================
"""
A module for handling training, validation and test data for the ANN model
"""

import src.helper as helper
import logging
from src.csv_reader import CsvReader, Dataenum

class Data(object):
    def __init__(self, networkconfig):
        self.netcfg = networkconfig
        self._current_train_index = 0
        self._current_valid_index = 0
        self._current_test_index = 0
        self.completed_training_epochs = 0
        self.percent_of_epoch = 0.0
        self.title_length = networkconfig['max_title_length']
        self.user_count = networkconfig['users_to_select']
        self.batch_size = self.netcfg['batch_size']
        self.reader = CsvReader(networkconfig)
        self._read_data()
        self._build_dict()

    def _read_data(self):
        """ Reads all the data from specified path """
        logging.debug("Reading training data...")

        self.train_data, self.train_labels = self.reader.get_data(Dataenum.TRAINING)
        self.train_size = len(self.train_data)

        logging.debug("Reading validation data...")

        self.validation_data, self.valid_labels = self.reader.get_data(Dataenum.VALIDATION)
        self.validation_size = len(self.validation_data)

        logging.debug("Reading testing data...")

        self.test_data, self.test_labels = self.reader.get_data(Dataenum.TESTING)
        self.test_size = len(self.test_data)

    def _build_dict(self):
        """ Builds dictionaries using given data """
        logging.debug("Building dictionaries...")

        vocab = " ".join(self.train_data).split()
        users = " ".join(self.train_labels).split()

        _, _, self.word_dict, self.rev_dict = \
            helper.build_dataset(vocab, vocabulary_size=self.netcfg['vocabulary_size'])
        _, _, self.users_dict, self.rev_users_dict = \
            helper.build_dataset(users, vocabulary_size=self.netcfg['user_count'])
        print(self.users_dict)

        # TODO Next train batch är i princip identisk med
        # next_valid_batch

    def next_train_batch(self, batch_size=None):
        batch_size = batch_size or self.batch_size
        """ Get the next batch of training data """
        batch_x = []
        batch_y = []
        for _ in range(0, batch_size):
            sentence = self.train_data[self._current_train_index]
            label = self.train_labels[self._current_train_index]
            self._current_train_index += 1
            # Support multiple epochs
            if self._current_train_index >= self.train_size:
                self._current_train_index = 0
                self.completed_training_epochs += 1
                self.percent_of_epoch = 0.0
            # TODO Detta ska inte ligga i funktionen som generar ny data

            # Turn sentences and labels into vector representations
            sentence_vec = helper.get_indicies(sentence,
                                               self.word_dict,
                                               self.netcfg['max_title_length'])

            label_vec = helper.label_vector(label.split(), self.users_dict,
                                            self.netcfg['user_count'])
            batch_x.append(sentence_vec)
            batch_y.append(label_vec)

        self.percent_of_epoch = self._current_train_index / self.train_size
        return batch_x, batch_y

    def get_validation(self):
        """ Get the whole validation set in a vectorized form """
        old_ind = self._current_valid_index
        self._current_valid_index = 0
        batch_x, batch_y = self.next_valid_batch()
        self._current_valid_index = old_ind
        return batch_x, batch_y

    def next_valid_batch(self, batch_size=None):
        batch_size = batch_size or self.batch_size
        """ Get the next batch of validation data """
        batch_x = []
        batch_y = []
        for _ in range(0, self.netcfg['batch_size']):
            sentence = self.validation_data[self._current_valid_index]
            label = self.valid_labels[self._current_valid_index]

            self._current_valid_index += 1

            # Support multiple epochs
            if self._current_valid_index >= self.validation_size:
                self._current_valid_index = 0

            # Turn sentences and labels into vectors
            sentence_vec = helper.get_indicies(sentence,
                                               self.word_dict,
                                               self.netcfg['max_title_length'])

            label_vec = helper.label_vector(label.split(), self.users_dict,
                                            self.netcfg['user_count'])
            batch_x.append(sentence_vec)
            batch_y.append(label_vec)
        return batch_x, batch_y

    def get_testing(self):
        """ Get the whole validation set in a vectorized form """
        old_ind = self._current_test_index
        self._current_test_index = 0
        batch_x, batch_y = self.next_test_batch()
        self._current_test_index = old_ind
        return batch_x, batch_y

    def next_test_batch(self):
        """ Get the next batch of validation data """
        batch_x = []
        batch_y = []
        for _ in range(0, self.netcfg['batch_size']):
            sentence = self.test_data[self._current_test_index]
            label = self.test_labels[self._current_test_index]

            self._current_test_index += 1

            # Support multiple epochs
            if self._current_test_index >= self.test_size:
                self._current_test_index = 0

            # Turn sentences and labels into vectors
            sentence_vec = helper.get_indicies(sentence,
                                               self.word_dict,
                                               self.netcfg['max_title_length'])

            label_vec = helper.label_vector(label.split(), self.users_dict,
                                            self.netcfg['user_count'])
            batch_x.append(sentence_vec)
            batch_y.append(label_vec)
        return batch_x, batch_y

    def for_n_train_epochs(self, num_epochs=1, batch_size=25):
        # TODO Ta bort parameterar
        """ Calculates how many training iterations to do for num_epochs
        number of epochs with a batch size of batch_size """
        return range((self.train_size * num_epochs) // batch_size)

    def get_training(self):
        """ Get the whole training set in a vectorized form """
        old_ind = self._current_train_index
        old_epoch = self.completed_training_epochs
        self._current_train_index = 0
        batch_x, batch_y = self.next_train_batch(self.train_size)
        self._current_train_index = old_ind
        self.completed_training_epochs = old_epoch
        return batch_x, batch_y

    def get_testing(self):
        """ Get the whole validation set in a vectorized form """
        old_ind = self._current_test_index
        self._current_test_index = 0
        batch_x, batch_y = self.next_testing_batch(self.title_length, self.user_count,
                                                 self.test_size)
        self._current_test_index = old_ind
        return batch_x, batch_y

    def next_testing_batch(self, title_length, user_count,
                         batch_size):
        """ Get the next batch of validation data """
        batch_x = []
        batch_y = []
        for _ in range(0, batch_size):
            sentence = self.test_data[self._current_test_index]
            label = self.test_labels[self._current_test_index]

            self._current_test_index += 1

            # Support multiple epochs
            if self._current_test_index >= self.test_size:
                self._current_test_index = 0

            # Turn sentences and labels into vectors
            sentence_vec = helper.get_indicies(sentence,
                                               self.word_dict, title_length)
            label_vec = helper.label_vector(label.split(), self.users_dict,
                                            user_count)
            batch_x.append(sentence_vec)
            batch_y.append(label_vec)
        return batch_x, batch_y
