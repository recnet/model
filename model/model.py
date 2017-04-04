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
from definitions import *
from .util import data as data
from .util.folder_builder import build_structure
from .util.writer import log_samefile
from .util.helper import get_val_summary_tensor


# TODO Separera checkpoints ut ur modell klassen
class Model(object):
    def __init__(self, config, session):
        self.config = config
        self._session = session
        self.output_layer = None
        self.latest_layer = None
        self.output_weights = None
        self.output_bias = None
        self.l2_term = tf.constant(0, dtype=tf.float64)

        self.vocabulary_size = config[VOC_SIZE]
        self.user_count = config[USER_COUNT]
        self.learning_rate = config[LEARN_RATE]
        self.embedding_size = config[EMBEDD_SIZE]
        self.max_title_length = config[MAX_TITLE_LENGTH]
        self.lstm_neurons = config[LSTM_NEURONS]
        self.batch_size = config[BATCH_SIZE]
        self.training_epochs = config[TRAINING_EPOCHS]
        self.use_l2_loss = config[USE_L2_LOSS]
        self.l2_factor = config[L2_FACTOR]
        self.use_dropout = config[USE_DROPOUT]
        self.dropout_prob = config[DROPOUT_PROB] # Only used for train op
        self.hidden_layers = config[HIDDEN_LAYERS]
        self.hidden_neurons = config[HIDDEN_NEURONS]
        self.is_trainable_matrix = config[TRAINABLE_MATRIX]
        self.use_pretrained = config[USE_PRETRAINED]
        self.use_constant_limit = config[USE_CONSTANT_LIMIT]
        self.constant_prediction_limit = config[CONSTANT_PREDICTION_LIMIT]
        self.use_concat_input = config[USE_CONCAT_INPUT]
        self.use_pretrained_net = config[USE_PRETRAINED_NET]
        self.subreddit_count = 0

        # Will be set in build_graph
        self.input = None
        self.subreddit_input = None
        self.target = None
        self.sec_target = None
        self.sigmoid = None
        self.train_op = None
        self.pre_train_op = None
        self.error = None
        self.init_op = None
        self.saver = None
        self.epoch = None
        self.keep_prob = None
        self.embedding_placeholder = None
        self.embedding_init = None
        self.train_writer = None
        self.valid_writer = None

        # variables for tensorboard
        self.prec_sum_training = None
        self.error_sum = None
        self.recall_sum_training = None
        self.f1_sum_training = None
        self.prec_sum_validation = None
        self.error_sum = None
        self.recall_sum_validation = None
        self.f1_sum_validation = None

        self.logging_dir = build_structure(config)
        self.checkpoints_dir = self.logging_dir + '/' + CHECKPOINTS_DIR + '/' + "models.ckpt"

        self.f1_score_train = 0
        self.f1_score_valid = 0
        self.epoch_top, self.prec_valid, self.prec_train, self.recall_valid, self.recall_train = 0, 0, 0, 0, 0



        with tf.device("/cpu:0"):
            self.data = data.Data(config)
            self.subreddit_count = self.data.subreddit_count
            if self.use_pretrained:
                self.vocabulary_size = len(self.data.embedding_matrix)


    def load_checkpoint(self):
        """ Loads any exisiting trained model """
        checkpoint_files = glob.glob(self.checkpoints_dir + "*")
        if all([os.path.isfile(file) for file in checkpoint_files]) \
                and checkpoint_files:
            self.saver.restore(self._session, self.checkpoints_dir)
            self._session.run(tf.local_variables_initializer())
        else:
            self._session.run(self.init_op)

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
        val_data, val_sub, val_labels = self.data.get_validation()
        val_prec, val_err, val_recall, val_f1 = \
            self._session.run([self.prec_sum_validation,
                               self.error_sum,
                               self.recall_sum_validation,
                               self.f1_sum_validation],
                              {self.input: val_data,
                               self.subreddit_input: val_sub,
                               self.target: val_labels,
                               self.keep_prob: 1.0})

        # Write results to TensorBoard
        self.valid_writer.add_summary(val_prec, epoch)
        self.valid_writer.add_summary(val_err, epoch)
        self.valid_writer.add_summary(val_recall, epoch)
        self.valid_writer.add_summary(val_f1, epoch)

        # Compute training error
        train_data, train_sub, train_labels = self.data.get_training()
        train_prec, train_err, train_recall, train_f1 = \
            self._session.run([self.prec_sum_training,
                               self.error_sum,
                               self.recall_sum_training,
                               self.f1_sum_training],
                              {self.input: train_data,
                               self.subreddit_input: train_sub,
                               self.target: train_labels,
                               self.keep_prob: 1.0})
        # Write results to Tensorboard
        self.train_writer.add_summary(train_prec, epoch)
        self.train_writer.add_summary(train_err, epoch)
        self.train_writer.add_summary(train_recall, epoch)
        self.train_writer.add_summary(train_f1, epoch)

        if self.f1_score_valid < get_val_summary_tensor(val_f1):
            self.f1_score_valid = get_val_summary_tensor(val_f1)
            self.f1_score_train = get_val_summary_tensor(train_f1)
            self.epoch_top, self.prec_valid, self.prec_train, self.recall_valid, self.recall_train = \
                epoch, get_val_summary_tensor(val_prec), get_val_summary_tensor(train_prec), \
                get_val_summary_tensor(val_recall), get_val_summary_tensor(train_recall)

    def validate_batch(self):
        """ Validates a batch of data and returns cross entropy error """
        with tf.device("/cpu:0"):
            batch_input, batch_sub, batch_label = self.data.next_valid_batch()

        return self._session.run(self.error,
                                 feed_dict={self.input: batch_input,
                                            self.subreddit_input: batch_sub,
                                            self.target: batch_label})

    # TODO funktionen gör alldeles för mycket,
    # dela upp utskrift, beräkning och träning
    def train(self, use_pretrained_net=False):
        """ Trains the model on the dataset """
        print("Starting training...")

        if self.use_pretrained and \
                (self.use_pretrained_net and use_pretrained_net) or \
                (not self.use_pretrained_net and not use_pretrained_net):
            self._session.run(self.embedding_init,
                              feed_dict={self.embedding_placeholder:
                                         self.data.embedding_matrix})

        old_epoch = 0

        if self.epoch.eval(self._session) == 0 and not use_pretrained_net:
            self.validate()

        # Train for a specified amount of epochs
        for i in self.data.for_n_train_epochs(self.training_epochs, self.batch_size):
            # Debug print out
            epoch = self.data.completed_training_epochs

            if not use_pretrained_net:
                training_error = self.train_batch()
                validation_error = self.validate_batch()

                # Don't validate so often
                if i % (self.data.train_size // self.batch_size // 10) == 0 and i:
                    done = self.data.percent_of_epoch
                    print("Validation error: {:f} | Training error: {:f} | Done: {:.0%}"
                          .format(validation_error, training_error, done))
            else:
                self.train_batch(True)

            # Do a full evaluation once an epoch is complete
            if epoch != old_epoch:
                self._session.run(self.epoch.assign_add(1))
                print("Epoch complete...old ", old_epoch)
                self.save_checkpoint()
                if not self.use_pretrained_net:
                    self.validate()
            old_epoch = epoch

        # Save model when done training
        self.save_checkpoint()
        log_samefile(config=self.config, f1_score_valid=self.f1_score_valid, f1_score_train=self.f1_score_train,
                     epoch_top=self.epoch_top, prec_valid=self.prec_valid, prec_train=self.prec_train,
                     recall_valid=self.recall_valid, recall_train=self.recall_train)

    def train_batch(self, pre_train_net=False):
        """ Trains for one batch and returns cross entropy error """
        with tf.device("/cpu:0"):
            if not pre_train_net:
                batch_input, batch_sub, batch_label = \
                    self.data.next_train_batch()
            else:
                batch_input, batch_label = \
                    self.data.next_pre_train_batch()

        if pre_train_net:
            self._session.run(self.pre_train_op,
                              {self.input: batch_input,
                               self.sec_target: batch_label})
        else:
            self._session.run(self.train_op,
                              {self.input: batch_input,
                               self.subreddit_input: batch_sub,
                               self.target: batch_label})

            return self._session.run(self.error,
                                     feed_dict={self.input: batch_input,
                                                self.subreddit_input: batch_sub,
                                                self.target: batch_label})
    def close_writers(self):
        """ Close tensorboard writers """
        self.train_writer.close()
        self.valid_writer.close()


