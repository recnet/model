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
        self.l2_term = 0

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
        self.train_writer = tf.summary.FileWriter(self.logging_dir + '/' + TENSOR_DIR_TRAIN)
        self.valid_writer = tf.summary.FileWriter(self.logging_dir + '/' + TENSOR_DIR_VALID)

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


class ModelBuilder(object):
    """A class following the builder pattern to create a model"""

    def __init__(self, config, session):
        self._model = Model(config, session)
        self.added_layers = False
        self.number_of_layers = 0

    def add_input_layer(self):
        """Adds a input layer to the graph, no other layers can be added before this has been added"""
        self._model.epoch = tf.Variable(0, dtype=tf.int32, name="train_epoch")

        self._model._input = tf.placeholder(tf.int32,
                                     [None, self._model.max_title_length],
                                     name="input")

        self._model._target = tf.placeholder(tf.float64,
                                      [None, self._model.user_count],
                                      name="target")

        self._model._keep_prob = tf.placeholder(tf.float64, name="keep_prob")

        lstm_layer = tf.contrib.rnn.LSTMCell(self._model.lstm_neurons, state_is_tuple=True)

        # Embedding matrix for the words
        embedding_matrix = tf.Variable(
            tf.random_uniform(
                [self._model.vocabulary_size, self._model.embedding_size],
                - 1.0, 1.0, dtype=tf.float64),
            name="embeddings")

        embedded_input = tf.nn.embedding_lookup(embedding_matrix,
                                                self._model._input)
        # Run the LSTM layer with the embedded input
        outputs, _ = tf.nn.dynamic_rnn(lstm_layer, embedded_input,
                                       dtype=tf.float64)

        outputs = tf.transpose(outputs, [1, 0, 2])
        output = outputs[-1]

        self._model.latest_layer = output

    def add_layer(self, number_of_neurons):
        """Adds a layer between latest added layer and the output layer"""

        self.number_of_layers += 1

        if not self.added_layers:
            self.added_layers = True
            weights = tf.Variable(tf.random_normal(
                [self._model.lstm_neurons, number_of_neurons],
                stddev=0.35,
                dtype=tf.float64),
                                    name="weights" + str(self.number_of_layers))
            bias = tf.Variable(tf.random_normal([number_of_neurons],
                                                    stddev=0.35,
                                                    dtype=tf.float64),
                                   name="biases" + str(self.number_of_layers))

        else:
            weights = tf.Variable(tf.random_normal(
                [self._model.latest_layer.get_shape()[1].value, number_of_neurons],
                stddev=0.35,
                dtype=tf.float64),
                                name="weights" + str(self.number_of_layers))
            bias = tf.Variable(tf.random_normal([number_of_neurons],
                                                 stddev=0.35,
                                                 dtype=tf.float64),
                                name="biases" + str(self.number_of_layers))

        logits = tf.matmul(self._model.latest_layer, weights) + bias
        self._model.latest_layer = tf.nn.relu(logits)

        if self._model.use_l2_loss:
            self._model.l2_term += tf.nn.l2_loss(weights) + tf.nn.l2_loss(bias)
        if self._model.use_dropout:
            self._model.latest_layer = tf.nn.dropout(self._model.latest_layer,
                                                    self._model.dropout_prob)
        return self

    def add_output_layer(self):
        """Adds an output layer, including error and optimisation functions.
           After this method no new layers should be added."""

        # Output layer
        # Feed the output of the previous layer to a sigmoid layer
        sigmoid_weights = tf.Variable(tf.random_normal(
            [self._model.latest_layer.get_shape()[1].value, self._model.user_count],
            stddev=0.35,
            dtype=tf.float64),
            name="weights")

        sigmoid_bias = tf.Variable(tf.random_normal([self._model.user_count],
                                                    stddev=0.35,
                                                    dtype=tf.float64),
                                   name="biases")

        logits = tf.matmul(self._model.latest_layer, sigmoid_weights) + sigmoid_bias
        self._model.sigmoid = tf.nn.sigmoid(logits)

        # Training

        # Defne error function
        error = tf.nn.sigmoid_cross_entropy_with_logits(labels=self._model._target,
                                                        logits=logits)

        if self._model.use_l2_loss:
            cross_entropy = tf.reduce_mean(error
                                           + self._model.l2_factor * tf.nn.l2_loss(sigmoid_weights)
                                           + self._model.l2_factor * tf.nn.l2_loss(sigmoid_bias)
                                           + self._model.l2_factor * self._model.l2_term)
        else:
            cross_entropy = tf.reduce_mean(error)

        self._model.error = cross_entropy
        self._model.train_op = tf.train.AdamOptimizer(
            self._model.learning_rate).minimize(cross_entropy)

        return self

    def add_precision_operations(self):
        """Adds precision operation and tensorboard operations"""
        # Cast a tensor to booleans, where top k are True, else False
        top_k_to_bool = lambda x: tf.greater_equal(
            x, tf.reduce_min(
                tf.nn.top_k(x, k=self._model.users_to_select)[0]))

        # Convert all probibalistic predictions to discrete predictions
        self._model.predictions = tf.map_fn(top_k_to_bool, self._model.sigmoid, dtype=tf.bool)
        # Need to create two different precisions, because they have
        # internal memory of old values
        self._model.precision_validation = tf.metrics.precision(self._model._target,
                                                         self._model.predictions)
        self._model.precision_training = tf.metrics.precision(self._model._target,
                                                       self._model.predictions)
        self._model.error_sum = tf.summary.scalar('cross_entropy', self._model.error)

        # Need to create two different, because they have internal memory
        # of old values
        self._model.prec_sum_validation = \
            tf.summary.scalar('precision', self._model.precision_validation[1])
        self._model.prec_sum_training = \
            tf.summary.scalar('precision', self._model.precision_training[1])

        return self

    def build(self):
        """Adds saver and init operation and returns the model"""
        self.add_input_layer()

        # Add a number of hidden layers
        for i in range(self._model.hidden_layers):
            self.add_layer(self._model.hidden_neurons)

        self.add_output_layer()

        self.add_precision_operations()

        self._model._init_op = tf.group(tf.global_variables_initializer(),
                                        tf.local_variables_initializer())
        self._model.saver = tf.train.Saver()
        self._model.load_checkpoint()
        return self._model
