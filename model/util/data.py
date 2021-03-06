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

import logging
from . import helper
from .csv_reader import CsvReader, Dataenum

class Data(object):
    def __init__(self, networkconfig):
        self.netcfg = networkconfig
        self._current_train_index = 0
        self._current_valid_index = 0
        self._current_test_index = 0
        self._current_pre_train_index = 0
        self.completed_training_epochs = 0
        self.percent_of_epoch = 0.0
        self.subreddit_count = 0
        self.title_length = networkconfig['max_title_length']
        self.batch_size = self.netcfg['batch_size']
        self.reader = CsvReader(networkconfig)
        self.use_pretrained = self.netcfg['use_pretrained']
        self.embedding_size = self.netcfg['embedding_size']
        self.pre_trained_matrix = self.netcfg['pre_trained_matrix']
        self.vocabulary_size = self.netcfg['vocabulary_size']
        self.user_count = self.netcfg['user_count']
        self.max_title_length = self.netcfg['max_title_length']
        self._read_data()
        self.embedding_matrix = None
        self._build_dict()
        self.train_absent = 0
        self.train_present = 0
        self.valid_absent = 0
        self.valid_present = 0

    def _read_data(self):
        """ Reads all the data from specified path """

        logging.debug("Reading training data...")

        self.train_data, self.train_subreddits, self.train_labels = \
            self.reader.get_data(Dataenum.TRAINING)
        self.train_size = len(self.train_data)

        logging.debug("Reading validation data...")

        self.validation_data, self.valid_subreddits, self.valid_labels = \
            self.reader.get_data(Dataenum.VALIDATION)
        self.validation_size = len(self.validation_data)

        logging.debug("Reading testing data...")

        self.test_data, self.test_subreddits, self.test_labels = \
            self.reader.get_data(Dataenum.TESTING)
        self.test_size = len(self.test_data)

    def _build_dict(self):
        """ Builds dictionaries using given data """
        logging.debug("Building dictionaries...")
        if not self.use_pretrained:
            vocab = " ".join(self.train_data).split()
            _, _, self.word_dict, self.rev_dict = \
                helper.build_dataset(vocab, vocabulary_size=self.vocabulary_size)
        else:
            self.word_dict, self.embedding_matrix = \
                self.reader.load_pretrained_embeddings(
                    self.pre_trained_matrix,
                    self.embedding_size)
        users = " ".join(self.train_labels).split()
        _, _, self.users_dict, self.rev_users_dict = \
            helper.build_dataset(users, vocabulary_size=self.user_count)

        subreddits = " ".join(self.train_subreddits).split()
        self.subreddit_dict = helper.build_subreddit_dict(subreddits)
        self.subreddit_count = len(self.subreddit_dict)

    def next_train_batch(self, batch_size=None):
        """ Get the next batch of training data """
        batch_size = batch_size or self.batch_size
        batch_x = []
        batch_sub = []
        batch_y = []
        for _ in range(0, batch_size):
            sentence = self.train_data[self._current_train_index]
            subreddit = self.train_subreddits[self._current_train_index]
            label = self.train_labels[self._current_train_index]
            self._current_train_index += 1
            # Support multiple epochs
            if self._current_train_index >= self.train_size:
                self._current_train_index = 0
                self.completed_training_epochs += 1
                self.percent_of_epoch = 0.0
            # TODO Detta ska inte ligga i funktionen som generar ny data

            # Turn sentences and labels into vector representations
            sentence_vec, present, absent = \
                helper.get_indicies(sentence,
                                    self.word_dict,
                                    self.max_title_length)
            self.train_present += present
            self.train_absent += absent

            subreddit_vec = helper.label_vector(subreddit,
                                                self.subreddit_dict,
                                                self.subreddit_count)

            label_vec = helper.label_vector(label.split(), self.users_dict,
                                            self.user_count)
            batch_x.append(sentence_vec)
            batch_sub.append(subreddit_vec)
            batch_y.append(label_vec)

        self.percent_of_epoch = self._current_train_index / self.train_size
        return batch_x, batch_sub, batch_y

    def next_pre_train_batch(self, batch_size=None):
        """ Get the next batch of training data """
        batch_size = batch_size or self.batch_size
        batch_x = []
        batch_y = []
        batch_sub = []

        for _ in range(0, batch_size):
            sentence = self.train_data[self._current_pre_train_index]
            subreddit = self.train_subreddits[self._current_pre_train_index]
            label = self.train_labels[self._current_pre_train_index]
            self._current_pre_train_index += 1
            # Support multiple epochs
            if self._current_pre_train_index >= self.train_size:
                self._current_pre_train_index = 0

            # Turn sentences and labels into vector representations
            sentence_vec, present, absent = \
                helper.get_indicies(sentence,
                                    self.word_dict,
                                    self.max_title_length)
            self.train_present += present
            self.train_absent += absent

            subreddit_vec = helper.label_vector(subreddit,
                                                self.subreddit_dict,
                                                self.subreddit_count)
            batch_x.append(sentence_vec)
            batch_y.append(subreddit_vec)
            batch_sub.append(subreddit_vec)


        return batch_x, batch_sub, batch_y

    def get_validation(self):
        """ Get the whole validation set in a vectorized form """
        old_ind = self._current_valid_index
        self._current_valid_index = 0
        batch_x, batch_sub, batch_y = self.next_valid_batch()
        self._current_valid_index = old_ind
        return batch_x, batch_sub, batch_y

    def next_valid_batch(self, batch_size=None):
        """ Get the next batch of validation data """
        batch_size = batch_size or self.batch_size
        batch_x = []
        batch_sub = []
        batch_y = []
        for _ in range(0, batch_size):
            sentence = self.validation_data[self._current_valid_index]
            subreddit = self.valid_subreddits[self._current_valid_index]
            label = self.valid_labels[self._current_valid_index]

            self._current_valid_index += 1

            # Support multiple epochs
            if self._current_valid_index >= self.validation_size:
                self._current_valid_index = 0

            # Turn sentences and labels into vectors
            sentence_vec, pres, absent = \
                helper.get_indicies(sentence,
                                    self.word_dict,
                                    self.max_title_length)

            self.valid_present += pres
            self.valid_absent += absent

            subreddit_vec = helper.label_vector(subreddit,
                                                self.subreddit_dict,
                                                self.subreddit_count)

            label_vec = helper.label_vector(label.split(), self.users_dict,
                                            self.user_count)
            batch_x.append(sentence_vec)
            batch_sub.append(subreddit_vec)
            batch_y.append(label_vec)
        return batch_x, batch_sub, batch_y

    def get_testing(self):
        """ Get the whole validation set in a vectorized form """
        old_ind = self._current_test_index
        self._current_test_index = 0
        batch_x, batch_sub, batch_y = self.next_test_batch(self.test_size)
        self._current_test_index = old_ind
        return batch_x, batch_sub, batch_y

    def next_test_batch(self, batch_size=None):
        """ Get the next batch of validation data """
        batch_size = batch_size or self.batch_size
        batch_x = []
        batch_sub = []
        batch_y = []
        for _ in range(0, batch_size):
            sentence = self.test_data[self._current_test_index]
            subreddit = self.test_subreddits[self._current_test_index]
            label = self.test_labels[self._current_test_index]

            self._current_test_index += 1

            # Support multiple epochs
            if self._current_test_index >= self.test_size:
                self._current_test_index = 0

            # Turn sentences and labels into vectors
            sentence_vec = helper.get_indicies(sentence,
                                               self.word_dict,
                                               self.max_title_length)

            subreddit_vec = helper.label_vector(subreddit,
                                                self.subreddit_dict,
                                                self.subreddit_count)

            label_vec = helper.label_vector(label.split(), self.users_dict,
                                            self.user_count)
            batch_x.append(sentence_vec)
            batch_sub.append(subreddit_vec)
            batch_y.append(label_vec)
        return batch_x, batch_sub, batch_y

    def for_n_train_epochs(self, num_epochs=1, batch_size=25):
        # TODO Ta bort parameterar
        """ Calculates how many training iterations to do for num_epochs
        number of epochs with a batch size of batch_size"""
        return range((self.train_size * num_epochs) // batch_size)

    def get_training(self):
        """ Get the whole training set in a vectorized form """
        old_ind = self._current_train_index
        old_epoch = self.completed_training_epochs
        self._current_train_index = 0
        batch_x, batch_sub, batch_y = self.next_train_batch(self.train_size)
        self._current_train_index = old_ind
        self.completed_training_epochs = old_epoch
        return batch_x, batch_sub, batch_y

    def get_stats(self):
        """ Returns statistics about embedding matrix """
        return self.train_present, self.train_absent, self.valid_present, self.valid_absent
