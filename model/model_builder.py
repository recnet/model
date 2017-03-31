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

import tensorflow as tf
from model.model import Model

class ModelBuilder(object):
    """A class following the builder pattern to create a model"""

    def __init__(self, config, session):
        self._model = Model(config, session)
        self.added_layers = False
        self.number_of_layers = 0

    def add_input_layer(self):
        """
        Adds a input layer to the graph, no other layers can
        be added before this has been added
        """
        self._model.epoch = tf.Variable(0, dtype=tf.int32, name="train_epoch")

        self._model.input = \
            tf.placeholder(tf.int32,
                           [None, self._model.max_title_length],
                           name="input")

        self._model.target = \
            tf.placeholder(tf.float64,
                           [None, self._model.user_count],
                           name="target")

        self._model.keep_prob = tf.placeholder(tf.float64, name="keep_prob")

        lstm_layer = tf.contrib.rnn.LSTMCell(self._model.lstm_neurons, state_is_tuple=True)

        # Embedding matrix for the words
        embedding_matrix = tf.Variable(
            tf.constant(0.0,
                        shape=[self._model.vocabulary_size,
                               self._model.embedding_size],
                        dtype=tf.float64),
            trainable=self._model.is_trainable_matrix,
            name="embedding_matrix",
            dtype=tf.float64)

        self._model.embedding_placeholder = \
            tf.placeholder(tf.float64,
                           [self._model.vocabulary_size, self._model.embedding_size])
        self._model.embedding_init = \
            embedding_matrix.assign(self._model.embedding_placeholder)

        embedded_input = tf.nn.embedding_lookup(embedding_matrix,
                                                self._model.input)
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

        logits = tf.add(tf.matmul(self._model.latest_layer, weights), bias)
        self._model.latest_layer = tf.nn.relu(
            logits, name="hidden_layer-" + str(self.number_of_layers))

        if self._model.use_l2_loss:
            self._model.l2_term = tf.add(
                tf.add(self._model.l2_term, tf.nn.l2_loss(weights)),
                tf.nn.l2_loss(bias))
        if self._model.use_dropout:
            self._model.latest_layer = \
                tf.nn.dropout(self._model.latest_layer,
                              self._model.dropout_prob,
                              name="hidden_layer" + str(self.number_of_layers) + "dropout")

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
                                      name="output_weights")

        sigmoid_bias = tf.Variable(tf.random_normal([self._model.user_count],
                                                    stddev=0.35,
                                                    dtype=tf.float64),
                                   name="output_biases")

        logits = tf.add(tf.matmul(self._model.latest_layer, sigmoid_weights), sigmoid_bias)
        self._model.sigmoid = tf.nn.sigmoid(logits)

        # Training

        # Defne error function
        error = tf.nn.sigmoid_cross_entropy_with_logits(labels=self._model.target,
                                                        logits=logits)

        if self._model.use_l2_loss:
            cross_entropy = \
                tf.reduce_mean(tf.add(
                    tf.add(error,
                           tf.multiply(self._model.l2_factor,
                                       tf.nn.l2_loss(sigmoid_weights))),
                    tf.add(tf.multiply(self._model.l2_factor,
                                       tf.nn.l2_loss(sigmoid_bias)),
                           tf.multiply(self._model.l2_factor,
                                       self._model.l2_term))))
        else:
            cross_entropy = tf.reduce_mean(error)

        self._model.error = cross_entropy
        self._model.train_op = tf.train.AdamOptimizer(
            self._model.learning_rate).minimize(cross_entropy)

        return self

    def add_precision_operations(self):
        """Adds precision operation and tensorboard operations"""
        # Determine which prediction function to use. Casts a tensor to
        # booleans.
        if self._model.use_constant_limit:
            # x above limit are True, else False
            prediction_func = lambda x: tf.greater_equal(
                x, self._model.constant_prediction_limit)
        else:
            # x above mean are True, else False
            prediction_func = lambda x: tf.greater_equal(
                x, tf.reduce_mean(x))

        # Convert all probibalistic predictions to discrete predictions
        self._model.predictions = \
            tf.map_fn(prediction_func, self._model.sigmoid, dtype=tf.bool)

        # Calculate precision
        # Need to create two different precisions, because they have
        # internal memory of old values
        _, self._model.precision_validation = \
            tf.metrics.precision(self._model.target,
                                 self._model.predictions)
        _, self._model.precision_training = \
            tf.metrics.precision(self._model.target,
                                 self._model.predictions)
        # Calculate recall
        _, self._model.recall_validation = \
            tf.metrics.recall(self._model.target,
                              self._model.predictions)
        _, self._model.recall_training = \
            tf.metrics.recall(self._model.target,
                              self._model.predictions)

        # Calculate F1-score: 2 * (prec * recall) / (prec + recall)
        self._model.f1_score_validation = \
            tf.multiply(2.0,
                        tf.truediv(tf.multiply(self._model.precision_validation,
                                               self._model.recall_validation),
                                   tf.add(self._model.precision_validation,
                                          self._model.recall_validation)))
        # Convert to 0 if f1 score is NaN
        self._model.f1_score_validation = \
            tf.where(tf.is_nan(self._model.f1_score_validation),
                     tf.zeros_like(self._model.f1_score_validation),
                     self._model.f1_score_validation)

        self._model.f1_score_training = \
            tf.multiply(2.0,
                        tf.truediv(tf.multiply(self._model.precision_training,
                                               self._model.recall_training),
                                   tf.add(self._model.precision_training,
                                          self._model.recall_training)))
        # Convert to 0 if f1 score is NaN
        self._model.f1_score_training = \
            tf.where(tf.is_nan(self._model.f1_score_training),
                     tf.zeros_like(self._model.f1_score_training),
                     self._model.f1_score_training)

        self._model.error_sum = \
            tf.summary.scalar('cross_entropy', self._model.error)

        # Need to create two different, because they have internal memory
        # of old values
        self._model.prec_sum_validation = \
            tf.summary.scalar('precision_validation',
                              self._model.precision_validation)
        self._model.prec_sum_training = \
            tf.summary.scalar('precision_training',
                              self._model.precision_training)

        self._model.recall_sum_validation = \
            tf.summary.scalar('recall_validation',
                              self._model.recall_validation)
        self._model.recall_sum_training = \
            tf.summary.scalar('recall_training',
                              self._model.recall_training)

        self._model.f1_sum_validation = \
            tf.summary.scalar('f1_score_validation',
                              self._model.f1_score_validation)
        self._model.f1_sum_training = \
            tf.summary.scalar('f1_score_training',
                              self._model.f1_score_training)

        return self

    def build(self):
        """Adds saver and init operation and returns the model"""
        self.add_input_layer()

        # Add a number of hidden layers
        for _ in range(self._model.hidden_layers):
            self.add_layer(self._model.hidden_neurons)

        self.add_output_layer()

        self.add_precision_operations()

        self._model.init_op = tf.group(tf.global_variables_initializer(),
                                       tf.local_variables_initializer())
        self._model.saver = tf.train.Saver()
        self._model.load_checkpoint()
        return self._model