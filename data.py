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

FILE_TRAINING = "training_data.csv"
FILE_VALIDATION = "validation_data.csv"
FILE_TESTING = "testing_data.csv"
DEFAULT_BATCH_SIZE = 1

class Data(object):
    """ A class for getting handling data """
    def __init__(self, data_path="./data/", verbose=False):
        self._data_path = data_path
        self._verbose = verbose
        self._current_train_index = 0
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
        if self._verbose:
            print("Reading validation data...")
        self.valid_data, self.valid_labels = \
            csv_reader.read(self._data_path + FILE_VALIDATION, [0], 1)
        if self._verbose:
            print("Reading testing data...")
        self.test_data, self.test_labels = \
            csv_reader.read(self._data_path + FILE_TESTING, [0], 1)

    def _build_dict(self):
        """ Builds dictionaries using given data """
        if self._verbose:
            print("Building dictionaries...")
        vocab = " ".join(self.train_data).split()
        users = " ".join(self.train_labels).split()

        _, _, self.word_dict, self.rev_dict = helper.build_dataset(vocab)
        _, _, self.users_dict, self.rev_users_dict = helper.build_dataset(users)

    def next_train_batch(self, title_length, user_count,
                         batch_size=DEFAULT_BATCH_SIZE):
        """ Get the next batch of training data """
        batch_x = []
        batch_y = []
        for _ in range(0, batch_size):
            t_i = self._current_train_index
            sentence = self.train_data[t_i]
            label = self.train_labels[t_i]

            # Support multiple epochs
            if t_i >= len(self.train_data):
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
            self._current_train_index += 1

        self.percent_of_epoch = self._current_train_index / len(self.train_data)
        return batch_x, batch_y



