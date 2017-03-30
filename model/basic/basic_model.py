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
""" A RNN model for predicting which people would like a text

The model is created as part of the bachelor's thesis
"Automatic mentions and highlights" at Chalmers University of
Technology and the University of Gothenburg.
"""
import glob
import os.path
import tensorflow as tf
from definitions import CHECKPOINTS_DIR, TENSOR_DIR_VALID, TENSOR_DIR_TRAIN
from ..util import data as data
from ..util.folder_builder import build_structure
from ..util.writer import log_config


# TODO Separera checkpoints ut ur modell klassen
class Model(object):
    def __init__(self, config, session):
        self._session = session
        self.output_layer = None
        self.latest_layer = None
        self.output_weights = None
        self.output_bias = None
        self.l2_term = tf.constant(0, dtype=tf.float64)

        self.vocabulary_size = config['vocabulary_size']
        self.user_count = config['user_count']
        self.learning_rate = config['learning_rate']
        self.embedding_size = config['embedding_size']
        self.max_title_length = config['max_title_length']
        self.lstm_neurons = config['lstm_neurons']
        self.batch_size = config['batch_size']
        self.training_epochs = config['training_epochs']
        self.users_to_select = config['users_to_select']
        self.use_l2_loss = config['use_l2_loss']
        self.l2_factor = config['l2_factor']
        self.use_dropout = config['use_dropout']
        self.dropout_prob = config['dropout_prob'] # Only used for train op
        self.hidden_layers = config['hidden_layers']
        self.hidden_neurons = config['hidden_neurons']


        # Will be set in build_graph
        self._input = None
        self._target = None
        self.sigmoid = None
        self.train_op = None
        self.error = None
        self._init_op = None
        self.saver = None
        self.epoch = None
        self._keep_prob = None

        self.logging_dir = build_structure(config)
        self.checkpoints_dir = self.logging_dir + '/' + CHECKPOINTS_DIR + '/' + "models.ckpt"
        log_config(config) #Discuss if we should do this after, and somehow take "highest" precision from validation?


        with tf.device("/cpu:0"):
            self.data = data.Data(config)


    def load_checkpoint(self):
        """ Loads any exisiting trained model """
        checkpoint_files = glob.glob(self.checkpoints_dir + "*")
        if all([os.path.isfile(file) for file in checkpoint_files]) \
                and checkpoint_files:
            self.saver.restore(self._session, self.checkpoints_dir)
            self._session.run(tf.local_variables_initializer())
        else:
            self._session.run(self._init_op)

    def save_checkpoint(self, path=None):
        """ Saves the model to a file """
        path = path or self.checkpoints_dir
        self.saver.save(self._session, path)

    def validate(self):
        """ Validates the model and returns the final precision """
        print("Starting validation...")
        # Evaluate epoch
        epoch = self.epoch.eval(self._session)

        # Compute validation error
        val_data, val_labels = self.data.get_validation()
        val_prec, val_err = self._session.run([self.prec_sum_validation,
                                               self.error_sum],
                                              {self._input: val_data,
                                               self._target: val_labels,
                                               self._keep_prob: 1.0})

        self.valid_writer.add_summary(val_prec, epoch)
        self.valid_writer.add_summary(val_err, epoch)

        # Compute training error
        train_data, train_labels = self.data.get_training()
        train_prec, train_err = self._session.run([self.prec_sum_training,
                                                   self.error_sum],
                                                  {self._input: train_data,
                                                   self._target: train_labels,
                                                   self._keep_prob: 1.0})
        self.train_writer.add_summary(train_prec, epoch)
        self.train_writer.add_summary(train_err, epoch)
        return None

    def validate_batch(self):
        """ Validates a batch of data and returns cross entropy error """
        with tf.device("/cpu:0"):
            data_batch, label_batch = self.data.next_valid_batch()

        return self._session.run(self.error,
                                 feed_dict={self._input: data_batch,
                                            self._target: label_batch})

    # TODO funktionen gör alldeles för mycket,
    # dela upp utskrift, beräkning och träning
    def train(self):
        """ Trains the model on the dataset """
        print("Starting training...")

        self.train_writer = tf.summary.FileWriter(self.logging_dir + '/' + TENSOR_DIR_TRAIN, self._session.graph)
        self.valid_writer = tf.summary.FileWriter(self.logging_dir + '/' + TENSOR_DIR_VALID)

        error_sum = 0
        val_error_sum = 0
        old_epoch = 0

        if self.epoch.eval(self._session) == 0:
            self.validate()
        # Train for a specified amount of epochs

        for i in self.data.for_n_train_epochs(self.training_epochs,
                                              self.batch_size):
            # Debug print out
            epoch = self.data.completed_training_epochs
            self.train_batch()
            # validation_error = self.validate_batch()

            # error_sum += error
            # val_error_sum += validation_error

            # Don't validate so often
            #            if i % (self.data.train_size // self.config['batch_size'] // 10) == 0 and i:
            #                 avg_val_err = val_error_sum / i
            #                 avg_trn_err = error_sum / i
            #                 print("Training... Epoch: {:d}, Done: {:%}" \
            #                       .format(epoch, done))
            #                 print("Validation error: {:f} ({:f}), Training error {:f} ({:f})" \
            #                       .format(validation_error, avg_val_err, error, avg_trn_err))
            #

            # Do a full evaluation once an epoch is complete
            if epoch != old_epoch:
                self._session.run(self.epoch.assign_add(1))
                print("Epoch complete...old ", old_epoch)
                self.save_checkpoint()
                self.validate()
            old_epoch = epoch

            # Save model when done training
            # self.save_checkpoint()

    def train_batch(self):
        """ Trains for one batch and returns cross entropy error """
        with tf.device("/cpu:0"):
            batch_input, batch_label = self.data.next_train_batch()

        self._session.run(self.train_op,
                          {self._input: batch_input,
                           self._target: batch_label})

        return self._session.run(self.error,
                                 feed_dict={self._input: batch_input,
                                            self._target: batch_label})
    def close_writers(self):
        self.train_writer.close()
        self.valid_writer.close()

