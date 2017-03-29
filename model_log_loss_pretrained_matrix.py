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
import data
from networkconfig import yamlconfig as networkconfig
from folder_builder import build_structure
from writer import log_config
from definitions import CHECKPOINTS_DIR, TENSOR_DIR_VALID, TENSOR_DIR_TRAIN

class Model(object):
    """ A model representing our neural network """

    def __init__(self, config, session):
        self.config = config
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
        with tf.device("/cpu:0"):
            self.data = data.Data(config)
            if 'usePretrained' in self.config:
                self.vocabulary_size = len(self.data.embedding_matrix)
        self.build_graph()
        self.load_checkpoint()
        self._session.run(self.embedding_init, feed_dict={self.embedding_placeholder: self.data.embedding_matrix})

    def build_graph(self):
        """ Builds the computational graph """
        self.epoch = tf.Variable(0, dtype=tf.int32, name="train_epoch")

        self._input = tf.placeholder(tf.int32,
                                     [None, self.max_title_length],
                                     name="input")

        self._target = tf.placeholder(tf.float64,
                                      [None, self.user_count],
                                      name="target")

        lstm_layer = tf.contrib.rnn.LSTMCell(self.lstm_neurons, state_is_tuple=True)

        # Embedding matrix for the words
        # embedding_matrix = tf.Variable(
        #     tf.random_uniform(
        #         [self.vocabulary_size, self.embedding_size],
        #         -1.0, 1.0, dtype=tf.float64),
        #     name="embeddings")
        embedding_matrix = tf.Variable(
            tf.constant(0.0, shape=[self.vocabulary_size, self.config['dimensions']], dtype=tf.float64),
            trainable=False, name="embedding_matrix", dtype=tf.float64)

        self.embedding_placeholder = tf.placeholder(tf.float64, [self.vocabulary_size, self.config['dimensions']])
        self.embedding_init = embedding_matrix.assign(self.embedding_placeholder)

        embedded_input = tf.nn.embedding_lookup(embedding_matrix,
                                                self._input)
        # Run the LSTM layer with the embedded input
        outputs, _ = tf.nn.dynamic_rnn(lstm_layer, embedded_input,
                                       dtype=tf.float64)

        outputs = tf.transpose(outputs, [1, 0, 2])
        output = outputs[-1]



        # Training

        # Feed the output of the LSTM layer to a softmax layer
        sigmoid_weights = tf.Variable(tf.random_normal(
            [self.lstm_neurons, self.user_count],
            stddev=0.35,
            dtype=tf.float64),
                                      name="weights")

        sigmoid_bias = tf.Variable(tf.random_normal([self.user_count],
                                                    stddev=0.35,
                                                    dtype=tf.float64),
                                   name="biases")

        logits = tf.matmul(output, sigmoid_weights) + sigmoid_bias
        self.sigmoid = tf.nn.sigmoid(logits)

        # Defne error function
        error = tf.nn.sigmoid_cross_entropy_with_logits(labels=self._target, logits=logits)
        #error = tf.losses.log_loss(labels=self._target,
        #                           predictions=self.sigmoid)

        cross_entropy = tf.reduce_mean(error)
        self.error = cross_entropy

        self.train_op = tf.train.AdamOptimizer(
            self.learning_rate).minimize(cross_entropy)

        # Cast a tensor to booleans, where top k are True, else False
        top_k_to_bool = lambda x: tf.greater_equal(
            x, tf.reduce_min(
                tf.nn.top_k(x, k=self.users_to_select)[0]))

        # Convert all probibalistic predictions to discrete predictions
        self.predictions = tf.map_fn(top_k_to_bool, self.sigmoid, dtype=tf.bool)
        # Need to create two different precisions, because they have
        # internal memory of old values
        self.precision_validation = tf.metrics.precision(self._target,
                                                         self.predictions)
        self.precision_training = tf.metrics.precision(self._target,
                                                       self.predictions)

        # Last step
        self._init_op = tf.group(tf.global_variables_initializer(),
                                 tf.local_variables_initializer())

        self.error_sum = tf.summary.scalar('cross_entropy', self.error)

        # Need to create two different, because they have internal memory
        # of old values
        self.prec_sum_validation = \
            tf.summary.scalar('precision', self.precision_validation[1])
        self.prec_sum_training = \
            tf.summary.scalar('precision', self.precision_training[1])

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
        validation_summary = self._session.run(self.prec_sum_validation,
                                               {self._input: val_data,
                                                self._target: val_labels})

        self.valid_writer.add_summary(validation_summary, epoch)
        validation_err_summary = self._session.run(self.error_sum,
                                                   {self._input: val_data,
                                                    self._target: val_labels})
        self.valid_writer.add_summary(validation_err_summary, epoch)

        # Compute training error
        train_data, train_labels = self.data.get_training()
        training_summary = self._session.run(self.prec_sum_training,
                                             {self._input: train_data,
                                              self._target: train_labels})
        self.train_writer.add_summary(training_summary, epoch)
        training_err_summary = self._session.run(self.error_sum,
                                                 {self._input: train_data,
                                                  self._target: train_labels})
        self.train_writer.add_summary(training_err_summary, epoch)
        return None

    def validate_batch(self):
        """ Validates a batch of data and returns cross entropy error """
        with tf.device("/cpu:0"):
            data_batch, label_batch = self.data.next_valid_batch()

        return self._session.run(self.error,
                                 feed_dict={self._input: data_batch,
                                            self._target: label_batch})

    def train(self):
        """ Trains the model on the dataset """
        print("Starting training...")
        error_sum = 0
        val_error_sum = 0
        old_epoch = 0
        # Do initial validation if first time running
        if self.epoch.eval(self._session) == 0:
            self.validate()
        # Train for a specified amount of epochs
        for i in self.data.for_n_train_epochs(self.training_epochs,
                                              self.batch_size):
            # Debug print out
            epoch = self.data.completed_training_epochs
            self.train_batch()
            validation_error = self.validate_batch()

            # error_sum += error
            # val_error_sum += validation_error

            # Don't validate so often
            if i % (self.data.train_size//self.batch_size//10) == 0 and i:
                avg_val_err = val_error_sum/i
                avg_trn_err = error_sum/i
                # print("Training... Epoch: {:d}, Done: {:%}" \
                #     .format(epoch, done))
                # print("Training error {:f} ({:f})" \
                #       .format( error, avg_trn_err))

                # print("Validation error: {:f} ({:f}))" \
                #     .format(validation_error, avg_val_err))

            # Do a full evaluation once an epoch is complete
            if epoch != old_epoch:
                self._session.run(self.epoch.assign_add(1))
                # print("Epoch complete...old ", old_epoch)
                # self.save_checkpoint()
                self.validate()
            old_epoch = epoch

        # Save model when done training
        # self.save_checkpoint()
        trainPres, trainAbs, validPres, validAbs = self.data.get_stats()
        print("STATISTICS ", trainPres, trainAbs, validPres, validAbs)
        print("Percentage present in train ", (trainPres)/(trainPres + trainAbs))
        print("Percentage present in valid ", (validPres)/(validPres + validAbs))

    def train_batch(self):
        """ Trains for one batch and returns cross entropy error """
        with tf.device("/cpu:0"):
            batch_input, batch_label = self.data.next_train_batch()
        self._session.run(self.train_op,
                          {self._input: batch_input,
                           self._target: batch_label})

        # self.save_checkpoint()
        return self._session.run(self.error,
                                 feed_dict={self._input: batch_input,
                                            self._target: batch_label})

    def close_writers(self):
        self.train_writer.close()
        self.valid_writer.close()

def main():
    """ A main method that creates the model and starts training it """
    with tf.Session() as sess:
        config = 5
        model = Model(networkconfig[config], sess)
        model.train()
        model.close_writers()

    tf.reset_default_graph()
    with tf.Session() as sess:
        config = 3
        model = Model(networkconfig[config], sess)
        model.train()
        model.close_writers()

if __name__ == "__main__":
    main()
