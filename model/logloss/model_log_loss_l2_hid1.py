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

class Model(object):
    """ A model representing our neural network """

    def __init__(self, config, session):
        self._session = session
        self.vocabulary_size = config['vocabulary_size']
        self.user_count = config['user_count']
        self.learning_rate = config['learning_rate']
        self.embedding_size = config['embedding_size']
        self.max_title_length = config['max_title_length']
        self.lstm_neurons = config['lstm_neurons']
        self.batch_size = config['batch_size']
        self.training_epochs = config['training_epochs']
        self.users_to_select = config['users_to_select']
        self.hidden_neurons = config['hidden_neurons']
        self.use_l2_loss = config['use_l2_loss']
        self.l2_factor = config['l2_factor']
        self.use_dropout = config['use_dropout']
        self.dropout_prob = config['dropout_prob'] # Only used for train op

        # Will be set in build_graph
        self._input = None
        self._target = None
        self.sigmoid = None
        self.train_op = None
        self.error = None
        self._init_op = None
        self.saver = None
        self.epoch = None

        self.logging_dir = build_structure(config)
        self.checkpoints_dir = self.logging_dir + '/' + CHECKPOINTS_DIR + '/' + "models.ckpt"
        log_config(config)  # Discuss if we should do this after, and somehow take "highest" precision from validation?
        self.train_writer = tf.summary.FileWriter(self.logging_dir + '/' + TENSOR_DIR_TRAIN)
        self.valid_writer = tf.summary.FileWriter(self.logging_dir + '/' + TENSOR_DIR_VALID)
        self.build_graph()

        with tf.device("/cpu:0"):
            self.data = data.Data(config)
            self.load_checkpoint()

    def build_graph(self):
        """ Builds the computational graph """
        self.epoch = tf.Variable(0, dtype=tf.int32, name="train_epoch")

        self._input = tf.placeholder(tf.int32,
                                     [None, self.max_title_length],
                                     name="input")

        self._target = tf.placeholder(tf.float64,
                                      [None, self.user_count],
                                      name="target")
        self._keep_prob = tf.placeholder(tf.float64, name="keep_prob")

        # LSTM Layer
        lstm_layer = tf.contrib.rnn.LSTMCell(self.lstm_neurons,
                                             state_is_tuple=True)

        # Embedding matrix for the words
        embedding_matrix = tf.Variable(
            tf.random_uniform(
                [self.vocabulary_size, self.embedding_size],
                -1.0, 1.0, dtype=tf.float64),
            name="embeddings")

        embedded_input = tf.nn.embedding_lookup(embedding_matrix,
                                                self._input)
        # Run the LSTM layer with the embedded input
        lstm_outputs, _ = tf.nn.dynamic_rnn(lstm_layer, embedded_input,
                                            dtype=tf.float64)

        lstm_outputs = tf.transpose(lstm_outputs, [1, 0, 2])
        lstm_output = lstm_outputs[-1]
        if self.use_dropout:
            lstm_output = tf.nn.dropout(lstm_outputs[-1], self._keep_prob)

        # Hidden layer 1
        hid1_weights = tf.Variable(tf.random_normal(
            [self.lstm_neurons, self.hidden_neurons],
            stddev=0.35,
            dtype=tf.float64),
                                   name="hid1_weights")

        # Small positive initial bias to avoid "dead" neurons for ReLU
        # (according to Deep MNIST tutorial from TF)
        hid1_bias = tf.Variable(tf.constant(0.1,
                                            shape=[self.hidden_neurons],
                                            dtype=tf.float64),
                                name="hid1_bias")

        hid1_logits = tf.matmul(lstm_output, hid1_weights) + hid1_bias
        hid1_relu = tf.nn.relu(hid1_logits)
        hid1_output = hid1_relu
        if self.use_dropout:
            hid1_output = tf.nn.dropout(hid1_relu, self._keep_prob)

        # Output layer
        # Feed the output of the previous layer to a sigmoid layer
        sigmoid_weights = tf.Variable(tf.random_normal(
            [self.hidden_neurons, self.user_count],
            stddev=0.35,
            dtype=tf.float64),
                                      name="weights")

        sigmoid_bias = tf.Variable(tf.random_normal([self.user_count],
                                                    stddev=0.35,
                                                    dtype=tf.float64),
                                   name="biases")

        logits = tf.matmul(hid1_output, sigmoid_weights) + sigmoid_bias
        self.sigmoid = tf.nn.sigmoid(logits)

        # Training

        # Defne error function
        error = tf.nn.sigmoid_cross_entropy_with_logits(labels=self._target,
                                                        logits=logits)

        # Regularization(L2)
        # If more layers are added these should be added as a l2_loss term in
        # the regularization function (both weight and bias).
        cross_entropy = None
        if self.use_l2_loss:
            cross_entropy = tf.reduce_mean(error \
                + self.l2_factor * tf.nn.l2_loss(sigmoid_weights) \
                + self.l2_factor * tf.nn.l2_loss(sigmoid_bias) \
                + self.l2_factor * tf.nn.l2_loss(hid1_weights) \
                + self.l2_factor * tf.nn.l2_loss(hid1_bias))
        else:
            cross_entropy = tf.reduce_mean(error)

        self.error = cross_entropy
        self.train_op = tf.train.AdamOptimizer(
            self.learning_rate).minimize(cross_entropy)

        # Cast a tensor to booleans, x above mean are True, else False
        greater_than_avg = lambda x: tf.greater_equal(
            x, tf.reduce_mean(x))

        # Convert all probibalistic predictions to discrete predictions
        self.predictions = tf.map_fn(greater_than_avg, self.sigmoid, dtype=tf.bool)

        # Calculate precision
        # Need to create two different precisions, because they have
        # internal memory of old values
        _, self.precision_validation = tf.metrics.precision(self._target,
                                                            self.predictions)
        _, self.precision_training = tf.metrics.precision(self._target,
                                                          self.predictions)
        # Calculate recall
        _, self.recall_validation = tf.metrics.recall(self._target,
                                                      self.predictions)
        _, self.recall_training = tf.metrics.recall(self._target,
                                                    self.predictions)

        # Calculate F1-score: 2 * (prec * recall) / (prec + recall)
        self.f1_score_validation = tf.multiply(2.0, tf.truediv( \
            tf.multiply(self.precision_validation, self.recall_validation), \
            tf.add(self.precision_validation, self.recall_validation)))
        self.f1_score_training = tf.multiply(2.0, tf.truediv( \
            tf.multiply(self.precision_training, self.recall_training), \
            tf.add(self.precision_training, self.recall_training)))

        # Last step
        self._init_op = tf.group(tf.global_variables_initializer(),
                                 tf.local_variables_initializer())

        self.error_sum = tf.summary.scalar('cross_entropy', self.error)

        # Need to create two different, because they have internal memory
        # of old values
        self.prec_sum_validation = \
            tf.summary.scalar('precision_validation', self.precision_validation)
        self.prec_sum_training = \
            tf.summary.scalar('precision_training', self.precision_training)

        self.recall_sum_validation = \
            tf.summary.scalar('recall_validation', self.recall_validation)
        self.recall_sum_training = \
            tf.summary.scalar('recall_training', self.recall_training)

        self.f1_sum_validation = \
            tf.summary.scalar('f1_score_validation', self.f1_score_validation)
        self.f1_sum_training = \
            tf.summary.scalar('f1_score_training', self.f1_score_training)

        self.saver = tf.train.Saver()

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
        val_prec, val_err, val_recall, val_f1 = \
                    self._session.run([self.prec_sum_validation,
                                       self.error_sum,
                                       self.recall_sum_validation,
                                       self.f1_sum_validation],
                                      {self._input: val_data,
                                       self._target: val_labels,
                                       self._keep_prob: 1.0})

        # Write results to TensorBoard
        self.valid_writer.add_summary(val_prec, epoch)
        self.valid_writer.add_summary(val_err, epoch)
        self.valid_writer.add_summary(val_recall, epoch)
        self.valid_writer.add_summary(val_f1, epoch)

        # Compute training error
        train_data, train_labels = self.data.get_training()
        train_prec, train_err, train_recall, train_f1 = \
                    self._session.run([self.prec_sum_training,
                                       self.error_sum,
                                       self.recall_sum_training,
                                       self.f1_sum_training],
                                      {self._input: train_data,
                                       self._target: train_labels,
                                       self._keep_prob: 1.0})
        # Write results to Tensorboard
        self.train_writer.add_summary(train_prec, epoch)
        self.train_writer.add_summary(train_err, epoch)
        self.train_writer.add_summary(train_recall, epoch)
        self.train_writer.add_summary(train_f1, epoch)

    def validate_batch(self):
        """ Validates a batch of data and returns cross entropy error """
        with tf.device("/cpu:0"):
            data_batch, label_batch = self.data.next_valid_batch()

        return self._session.run(self.error,
                                 feed_dict={self._input: data_batch,
                                            self._target: label_batch,
                                            self._keep_prob: 1.0})

    def train(self):
        """ Trains the model on the dataset """
        print("Starting training...")
        old_epoch = 0
        # Do initial validation if first time running
        if self.epoch.eval(self._session) == 0:
            self.validate()
        # Train for a specified amount of epochs
        for i in self.data.for_n_train_epochs(self.training_epochs,
                                              self.batch_size):
            # Debug print out
            epoch = self.data.completed_training_epochs
            training_error = self.train_batch()
            validation_error = self.validate_batch()

            # error_sum += error
            # val_error_sum += validation_error

            # Don't validate so often
            if i % (self.data.train_size//self.batch_size//10) == 0 and i:
                done = self.data.percent_of_epoch
                print("Validation error: {:f} | Training error: {:f} | Done: {:.0%}" \
                    .format(validation_error, training_error, done))

            # Do a full evaluation once an epoch is complete
            if epoch != old_epoch:
                self._session.run(self.epoch.assign_add(1))
                print("Epoch complete...old ", old_epoch)
                self.save_checkpoint()
                self.validate()
            old_epoch = epoch

        # Save model when done training
        self.save_checkpoint()

    def train_batch(self):
        """ Trains for one batch and returns cross entropy error """
        with tf.device("/cpu:0"):
            batch_input, batch_label = self.data.next_train_batch()
        self._session.run(self.train_op,
                          {self._input: batch_input,
                           self._target: batch_label,
                           self._keep_prob: self.dropout_prob})

        # self.save_checkpoint()
        return self._session.run(self.error,
                                 feed_dict={self._input: batch_input,
                                            self._target: batch_label,
                                            self._keep_prob: 1.0})

    def close_writers(self):
        """ Closes tensorboard writers """
        self.train_writer.close()
        self.valid_writer.close()