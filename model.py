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
""" A RNN model for predicting which people would like a text

The model is created as part of the bachelor's thesis
"Automatic mentions and highlights" at Chalmers University of
Technology and the University of Gothenburg.
"""
import glob
import os.path
import tensorflow as tf
import helper
import numpy as np
import data

CKPT_PATH = "checkpoints/model.ckpt"

class Model(object):
    """ A model representing our neural network """
    def __init__(self, session):
        self._session = session
        self.vocabulary_size = 50000
        self.learning_rate = 0.5
        self.embedding_size = 150
        self.max_title_length = 30
        self.lstm_neurons = 500
        self.user_count = 13000
        self.batch_size = 100
        self.training_epochs = 10
        self.users_to_select = 15
        # Will be set in build_graph
        self._input = None
        self._target = None
        self.softmax = None
        self.train_op = None
        self.error = None
        self._init_op = None
        self.saver = None

        self.build_graph()
        self.data = data.Data(data_path="./data/top5/", verbose=True)
        self.load_checkpoint()

    def build_graph(self):
        """ Builds the model in tensorflow """

        print("Building graph...")
        # Placeholders for input and output
        self._input = tf.placeholder(tf.int32,
                                     [None, self.max_title_length],
                                     name="input")
        self._target = tf.placeholder(tf.int32,
                                      [None, self.user_count],
                                      name="target")

        # This is the first, and input, layer of our network
        lstm_layer = tf.nn.rnn_cell.LSTMCell(self.lstm_neurons,
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
        outputs, state = tf.nn.dynamic_rnn(lstm_layer, embedded_input,
                                           dtype=tf.float64)
        outputs = tf.transpose(outputs, [1, 0, 2])
        output = outputs[-1]

        # Feed the output of the LSTM layer to a softmax layer
        softmax_weights = tf.Variable(tf.random_normal(
            [self.lstm_neurons, self.user_count],
            stddev=0.35,
            dtype=tf.float64),
                                      name="weights")

        softmax_bias = tf.Variable(tf.random_normal([self.user_count],
                                                    stddev=0.35,
                                                    dtype=tf.float64),
                                   name="biases")

        logits = tf.matmul(output, softmax_weights) + softmax_bias
        self.softmax = tf.nn.softmax(logits)

        # Defne error function
        error = tf.nn.softmax_cross_entropy_with_logits(labels=self._target,
                                                        logits=logits)
        cross_entropy = tf.reduce_mean(error)
        self.train_op = tf.train.GradientDescentOptimizer(
            self.learning_rate).minimize(cross_entropy)
        self.error = cross_entropy

        # Last step
        self._init_op = tf.global_variables_initializer()
        self.saver = tf.train.Saver()

    def load_checkpoint(self):
        """ Loads any exisiting trained model """
        checkpoint_files = glob.glob(CKPT_PATH + "*")
        if all([os.path.isfile(file) for file in checkpoint_files]) \
           and checkpoint_files:
            self.saver.restore(self._session, CKPT_PATH)
        else:
            self._session.run(self._init_op)

    def save_checkpoint(self, path=CKPT_PATH):
        """ Saves the model to a file """
        self.saver.save(self._session, path)

    def validate(self):
        """ Validates the model and returns the final precision """
        print("Starting validation...")
        true_positives = 0
        false_positives = 0

        for i, sentence in enumerate(self.data.valid_data):
            label = self.data.valid_labels[i]
            sentence_vec = helper.get_indicies(sentence,
                                               self.data.word_dict,
                                               self.max_title_length)
            label_vec = helper.label_vector(label.split(),
                                            self.data.users_dict,
                                            self.user_count)

            res = self._session.run(self.softmax,
                                    {self._input: [sentence_vec],
                                     self._target: [label_vec]})
            res = res[0]

            # Get the top k probable users from the softmax
            to_select = self.users_to_select
            if len(label.split()) < to_select:
                # If fewer users than k actually liked the post we
                # don't want incorrect error numbers
                to_select = len(label.split())
            indicies = np.argsort(res)[-to_select:]

            for j in indicies:
                lab = label_vec[j]
                if lab == 0:
                    false_positives += 1
                else:
                    true_positives += 1

        precision = true_positives / (true_positives + false_positives)
        print("Final validation... Precision: {:%}".format(precision))
        return precision

    def validate_batch(self):
        """ Validates a batch of data and returns cross entropy error """
        data_batch, label_batch = self.data.next_valid_batch \
            (self.max_title_length, self.user_count, self.batch_size)

        return self._session.run(self.error,
                                 feed_dict={self._input: data_batch,
                                            self._target: label_batch})

    def train(self):
        """ Trains the model on the dataset """
        print("Starting training...")


        iters_diverge = 0
        error_sum = 0
        val_error_sum = 0
        old_avg_err = 0
        old_val_avg_err = 0
        # Train for a specified amount of epochs
        for i in self.data.for_n_train_epochs(self.training_epochs,
                                              self.batch_size):
            # Debug print out
            epoch = self.data.completed_training_epochs
            done = self.data.percent_of_epoch
            error = self.train_batch()
            validation_error = self.validate_batch()

            error_sum += error
            val_error_sum += validation_error

            if i % 10 == 0:
                new_avg_err = error_sum / 10
                new_val_avg_err = val_error_sum / 10
                error_sum = 0
                val_error_sum = 0
                if np.sign(new_val_avg_err - old_val_avg_err) > np.sign(new_avg_err - old_avg_err):
                    iters_diverge += 1
                    print("Diff")
                else:
                    print("Reset")
                    iters_diverge = 0
                old_avg_err = new_avg_err
                old_val_avg_err = new_val_avg_err
                print("Validation error: {:f}, Training error {:f}".format(new_val_avg_err, new_avg_err))

            # Print completion every 10%
            if done % 0.1 <= 0.03:
                print("Training... Epoch: {:d}, Done: {:%}" \
                      .format(epoch, done))

        # Save model when done training
        self.save_checkpoint()

    def train_batch(self):
        """ Trains for one batch and returns cross entropy error """
        batch_input, batch_label = self.data.next_train_batch \
            (self.max_title_length, self.user_count, self.batch_size)

        self._session.run(self.train_op,
                          {self._input: batch_input,
                           self._target: batch_label})

        self.save_checkpoint()
        return self._session.run(self.error,
                                 feed_dict={self._input: batch_input,
                                            self._target: batch_label})

def main():
    """ A main method that creates the model and starts training it """
    model = Model(tf.InteractiveSession())
    model.train()
    # model.validate()

if __name__ == "__main__":
    main()
