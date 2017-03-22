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
#==============================================================================
"""
A module for handling training, validation and test data for the ANN model
"""
import csv_reader
import helper

#FILE_TRAINING    = "arti.csv"
#FILE_VALIDATION = "arti_val.csv"

FILE_TRAINING = "training_data_top_n.csv"
FILE_VALIDATION = "validation_data_top_n.csv"
FILE_TESTING = "testing_data_top_n.csv"
DEFAULT_BATCH_SIZE = 1
DEFAULT_VOCAB_SIZE = 50000

class Data(object):
    """ A class for getting handling data """
    def __init__(self,user_count, data_path="./data/", verbose=False,
                 vocab_size=DEFAULT_VOCAB_SIZE):
        self.users_count = user_count
        self._data_path = data_path
        self._vocab_size = vocab_size
        self._verbose = verbose
        self._current_train_index = 0
        self._current_valid_index = 0
        self._current_test_index = 0
        self.completed_training_epochs = 0
        self.percent_of_epoch = 0.0
        self._read_data()
        self._build_dict()

    def _read_data(self):
        """ Reads all the data from specified path """
        if self._verbose:
            print("Reading training data...")
        self.train_data, self.train_labels = \
            csv_reader.read(self._data_path + FILE_TRAINING, [0], 1)
        self.train_size = len(self.train_data)
        if self._verbose:
            print("Reading validation data...")
        self.valid_data, self.valid_labels = \
            csv_reader.read(self._data_path + FILE_VALIDATION, [0], 1)
        self.valid_size = len(self.valid_data)
        if self._verbose:
            print("Reading testing data...")
        self.test_data, self.test_labels = \
            csv_reader.read(self._data_path + FILE_TESTING, [0], 1)
        self.test_size = len(self.test_data)

    def _build_dict(self):
        """ Builds dictionaries using given data """
        if self._verbose:
            print("Building dictionaries...")
        vocab = " ".join(self.train_data).split()
        users = " ".join(self.train_labels).split()

        _, _, self.word_dict, self.rev_dict = \
            helper.build_dataset(vocab, vocabulary_size=self._vocab_size)
        _, _, self.users_dict, self.rev_users_dict = \
        helper.build_dataset(users, vocabulary_size=self.users_count)

    def next_train_batch(self, title_length, user_count,
                         batch_size=DEFAULT_BATCH_SIZE):
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

            # Turn sentences and labels into vector representations
            sentence_vec = helper.get_indicies(sentence,
                                               self.word_dict, title_length)
            label_vec = helper.label_vector(label.split(), self.users_dict,
                                            user_count)
            batch_x.append(sentence_vec)
            batch_y.append(label_vec)

        self.percent_of_epoch = self._current_train_index / self.train_size
        return batch_x, batch_y

    def get_training(self, title_length, user_count):
        """ Get the whole training set in a vectorized form """
        old_ind = self._current_train_index
        old_epoch = self.completed_training_epochs
        self._current_train_index = 0
        batch_x, batch_y = self.next_train_batch(title_length, user_count,
                                                 self.train_size)
        self._current_train_index = old_ind
        self.completed_training_epochs = old_epoch
        return batch_x, batch_y

    def get_validation(self, title_length, user_count):
        """ Get the whole validation set in a vectorized form """
        old_ind = self._current_valid_index
        self._current_valid_index = 0
        batch_x, batch_y = self.next_valid_batch(title_length, user_count,
                                                 self.valid_size)
        self._current_valid_index = old_ind
        return batch_x, batch_y

    def next_valid_batch(self, title_length, user_count,
                         batch_size=DEFAULT_BATCH_SIZE):
        """ Get the next batch of validation data """
        batch_x = []
        batch_y = []
        for _ in range(0, batch_size):
            sentence = self.valid_data[self._current_valid_index]
            label = self.valid_labels[self._current_valid_index]

            self._current_valid_index += 1

            # Support multiple epochs
            if self._current_valid_index >= self.valid_size:
                self._current_valid_index = 0

            # Turn sentences and labels into vectors
            sentence_vec = helper.get_indicies(sentence,
                                               self.word_dict, title_length)
            label_vec = helper.label_vector(label.split(), self.users_dict,
                                            user_count)
            batch_x.append(sentence_vec)
            batch_y.append(label_vec)
        return batch_x, batch_y

    def for_n_train_epochs(self, num_epochs=1, batch_size=DEFAULT_BATCH_SIZE):
        """ Calculates how many training iterations to do for num_epochs
        number of epochs with a batch size of batch_size """
        return range((self.train_size*num_epochs)//batch_size)

    def get_testing(self, title_length, user_count):
        """ Get the whole validation set in a vectorized form """
        old_ind = self._current_test_index
        self._current_test_index = 0
        batch_x, batch_y = self.next_testing_batch(title_length, user_count,
                                                 self.test_size)
        self._current_test_index = old_ind
        return batch_x, batch_y

    def next_testing_batch(self, title_length, user_count,
                         batch_size=DEFAULT_BATCH_SIZE):
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
