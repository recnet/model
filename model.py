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
        self.embedding_size = 128
        self.max_title_length = 30
        self.lstm_neurons = 100
        self.user_count = 13000
        self.batch_size = 100
        self.build_graph()
        self.data = data.Data(verbose=True)
        self.load_checkpoint()

    def build_graph(self):
        """ Builds the model in tensorflow """

        print("Building graph...")
        # Placeholders for input and output
        self._input = tf.placeholder(tf.int32, [self.batch_size, self.max_title_length], name="input")
        self._target = tf.placeholder(tf.int32, [self.batch_size, self.user_count], name="target")

        # This is the first, and input, layer of our network
        self.lstm_layer = tf.nn.rnn_cell.LSTMCell(self.lstm_neurons,
                                                  state_is_tuple=True)
        # Embedding matrix for the words
        self._embedding_matrix = tf.Variable(
            tf.random_uniform(
                [self.vocabulary_size, self.embedding_size],
                -1.0, 1.0, dtype=tf.float64),
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
        output = outputs[-1]
        # Feed the output of the LSTM layer to a softmax layer
        self.softmax_weights = tf.Variable(tf.random_normal(
            [self.lstm_neurons, self.user_count],
            stddev=0.35,
            dtype=tf.float64),
                                           name="weights")

        self.softmax_bias = tf.Variable(tf.random_normal([self.user_count],
                                                         stddev=0.35,
                                                         dtype=tf.float64),
                                        name="biases")

        logits = tf.matmul(output, self.softmax_weights) + self.softmax_bias
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

    def save_checkpoint(self):
        """ Saves the model to a file """
        self.saver.save(self._session, CKPT_PATH)

    def validate(self):
        """ Validates the model """
        print("Starting validation...")
        errors = 0
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
            ind = np.argmax(res[0])
            val = res[0][ind]
            lab = label_vec[ind]

            if lab == 0:
                errors += 1

            if i % 1000 == 0 and i > 0:
                print("Error rate: ", errors/i)

        print("Errors: ", errors)
        print("Final error rate: ", errors/len(self.data.valid_data))

    def train(self):
        """ Trains the model on the dataset """
        print("Starting training...")
        for _ in range(1000):
            batch_input, batch_label = self.data.next_train_batch \
                (self.max_title_length, self.user_count, self.batch_size)

            self._session.run(self.train_op,
                              {self._input: batch_input,
                               self._target: batch_label})
            # Debug print out
            epoch = self.data.completed_training_epochs
            done = self.data.percent_of_epoch
            error = self._session.run(self.error,
                                      feed_dict={self._input: batch_input,
                                                 self._target: batch_label})

            print("Training... Epoch: {:d}, Done: {:%}, Error: {:f}" \
                .format(epoch, done, error))

        # Save model when done training
        self.save_checkpoint()

def main():
    """ A main method that creates the model and starts training it """
    model = Model(tf.InteractiveSession())
    model.train()
    model.validate()

if __name__ == "__main__":
    main()
