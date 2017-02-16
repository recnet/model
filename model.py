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
import csv_reader
import helper

CKPT_PATH = "checkpoints/model.ckpt"

class Model(object):
    """ A model representing our neural network """
    def __init__(self, session):
        self._session = session
        self.vocabulary_size = 50000
        self.embedding_size = 128
        self.max_title_length = 30
        self.lstm_neurons = 500
        self.user_count = 100
        self.batch_size = 1
        self.cost = 40 #Don't know what this should be initialised as
        self.build_graph()
        self.load_checkpoint()

    def build_graph(self):
        """ Builds the model in tensorflow """

        # Placeholders for input and output
        self._input = tf.placeholder(tf.int32, [1, self.max_title_length])
        self._target = tf.placeholder(tf.int32, [1, self.user_count])

        # This is the first, and input, layer of our network
        self.lstm_layer = tf.nn.rnn_cell.LSTMCell(self.max_title_length,
                                                  state_is_tuple=True)
        # Embedding matrix for the words
        self._embedding_matrix = tf.Variable(
            tf.random_uniform(
                [self.vocabulary_size, self.embedding_size], -1.0, 1.0, dtype=tf.float64),
            name="embeddings")
        embedded_input = tf.nn.embedding_lookup(self._embedding_matrix,
                                                self._input)
        embedded_input = tf.unstack(embedded_input, num=self.max_title_length,
                                    axis=1)

        # Run the LSTM layer with the embedded input

        initial_state = self.lstm_layer.zero_state(self.batch_size,
                                                   dtype=tf.float64)

        outputs, state = tf.nn.rnn(self.lstm_layer, embedded_input,
                                   initial_state=initial_state)
        self.lstm_final_state = state
        #output = tf.reshape(tf.concat_v2(outputs, 1), [-1, self.lstm_neurons])
        output = outputs[-1]
        # Feed the output of the LSTM layer to a softmax layer
        self.softmax_weights = tf.Variable(tf.random_normal([self.max_title_length,
                                                             self.user_count],
                                                            stddev=0.35,
                                                            dtype=tf.float64),
                                           name="weights")

        self.softmax_bias = tf.Variable(tf.random_normal([self.user_count],
                                                         stddev=0.35,
                                                         dtype=tf.float64),
                                        name="biases")

        logits = tf.matmul(output, self.softmax_weights) + self.softmax_bias
        self.softmax = tf.nn.softmax(logits)


        error = tf.nn.softmax_cross_entropy_with_logits(labels=self._target, logits=logits)
        cross_entropy = tf.reduce_mean(error)
        self.train_op = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

        # Last step
        self._init_op = tf.global_variables_initializer()
        self.saver = tf.train.Saver()

    def load_checkpoint(self):
        """ Loads any exisiting trained model """
        checkpoint_files = glob.glob(CKPT_PATH + "*")
        if all([os.path.isfile(file) for file in checkpoint_files]) and checkpoint_files:
            self.saver.restore(self._session, CKPT_PATH)
        else:
            self._session.run(self._init_op)

    def save_checkpoint(self):
        """ Saves the model to a file """
        self.saver.save(self._session, CKPT_PATH)

    def train(self):
        """ Trains the model on the dataset """
        data, labels = csv_reader.read("../data/training_data.csv", [0], 1)

        vocab = " ".join(data)
        users = " ".join(labels)

        _, _, dict, rev_dict = helper.build_dataset(vocab)
        _, _, users_dict, rev_users_dict = helper.build_dataset(users)

        self.user_count = len(users_dict)

        for i, sentence in enumerate(data):
            label = labels[i]

            sentence_vec = helper.getIndices(sentence, dict)
            label_vec = helper.label_vector(label.split(), users_dict)

            self._session.run(self.train_op,
                              {self._input: [sentence_vec],
                                self._target: [label_vec]})
            print("hej")

        # state = self.session.run(self.initial_state)
        #
        # fetches = {
        #    "cost": self.cost,
        #    "final_state": self.lstm_final_state
        # }
        # #Not sure if this will be needed.
        #
        # for step in range(self.batch_size): #Not sure if batch size
        #     feed_dict = {}
        #
        #     for i, (c, h) in enumerate(self.initial_state):
        #         feed_dict[c] = state[i].c
        #         feed_dict[h] = state[i].h
        #
        #     vals = self.session.run(fetches, feed_dict)
        #

        # Save model when done training
        self.save_checkpoint()

def main():
    model = Model(tf.InteractiveSession())
    model.train()

if(__name__=="__main__"):
    main()