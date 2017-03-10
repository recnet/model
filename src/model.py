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
import src.data as data
from src.writer import Writer
from src.networkconfig import yamlconfig as networkconfig
from definitions import CHECKPOINTS_PATH


# TODO Separera checkpoints ut ur modell klassen
class Model(object):
    def __init__(self, config):
        self.config = config
        self._session = tf.InteractiveSession()
        self._input = None
        self._target = None
        self.softmax = None
        self.train_op = None
        self.error = None
        self._init_op = None
        self.saver = None

        self.vocabulary_size = config['vocabulary_size']
        self.user_count = config['user_count']
        self.learning_rate = config['learning_rate']
        self.embedding_size = config['embedding_size']
        self.max_title_length = config['max_title_length']
        self.lstm_neurons = config['lstm_neurons']
        self.batch_size = config['batch_size']
        self.training_epochs = config['training_epochs']
        self.users_to_select = config['users_to_select']



        self.build_graph()
        with tf.device("/cpu:0"):
            self.data = data.Data(config)
            self.load_checkpoint()

    def build_graph(self):
        self._input = tf.placeholder(tf.int32,
                                     [None, self.max_title_length],
                                     name="input")

        self._target = tf.placeholder(tf.int32,
                                      [None, self.user_count],
                                      name="target")

        lstm_layer = tf.contrib.rnn.LSTMCell(self.lstm_neurons, state_is_tuple=True)

        # Embedding matrix for the words
        embedding_matrix = tf.Variable(
            tf.random_uniform(
                [self.vocabulary_size, self.embedding_size],
                - 1.0, 1.0, dtype=tf.float64),
            name="embeddings")

        embedded_input = tf.nn.embedding_lookup(embedding_matrix,
                                                self._input)
        # Run the LSTM layer with the embedded input
        outputs, _ = tf.nn.dynamic_rnn(lstm_layer, embedded_input,
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

        self.train_op = tf \
            .train \
            .AdamOptimizer(self.learning_rate) \
            .minimize(cross_entropy)

        self.error = cross_entropy

        # Cast a tensor to booleans, where top k are True, else False
        top_k_to_bool = lambda x: tf.greater_equal(
            x, tf.reduce_min(tf.nn.top_k(x, k=self.users_to_select)[0]))

        # Convert all probibalistic predictions to discrete predictions
        self.predictions = tf.map_fn(top_k_to_bool, self.softmax, dtype=tf.bool)
        self.precision = tf.metrics.precision(self._target,
                                              self.predictions)  # Need to create two different, because they have internal memory of old values
        self.precision_copy = tf.metrics.precision(self._target, self.predictions)

        # Last step
        self._init_op = tf.group(tf.global_variables_initializer(),
                                 tf.local_variables_initializer())

        self.error_sum = tf.summary.scalar('cross_entropy', self.error)

        self.prec_sum = tf.summary.scalar('precision', self.precision[
            1])  # Need to create two different, because they have internal memory of old values
        self.prec_sum_copy = tf.summary.scalar('precision', self.precision_copy[1])

        # Last step
        log_dir = "./data"
        self.train_writer = tf.summary.FileWriter(log_dir + '/train')
        self.valid_writer = tf.summary.FileWriter(log_dir + '/valid')

        self.saver = tf.train.Saver()

    def load_checkpoint(self):
        """ Loads any exisiting trained model """
        checkpoint_files = glob.glob(os.path.join(CHECKPOINTS_PATH, "*"))
        if all([os.path.isfile(file) for file in checkpoint_files]) \
                and checkpoint_files:
            self.saver.restore(self._session, CHECKPOINTS_PATH)
            self._session.run(tf.local_variables_initializer())
        else:
            self._session.run(self._init_op)

    def save_checkpoint(self, path=os.path.join(CHECKPOINTS_PATH, "models.ckpt")):
        """ Saves the model to a file """
        self.saver.save(self._session, path)

    def validate(self, epoch):
        """ Validates the model and returns the final precision """
        print("Starting validation...")

        val_data, val_labels = self.data.get_validation()
        summary = self._session.run(self.prec_sum, {self._input: val_data, self._target: val_labels})
        self.valid_writer.add_summary(summary, epoch)
        errSum = self._session.run(self.error_sum, {self._input: val_data, self._target: val_labels})
        self.valid_writer.add_summary(errSum, epoch)

        train_data, train_labels = self.data.get_testing()  # HUGE NOTE: in the data class, change testing file to be the same as training file to compare validation vs training dataset :)
        sum = self._session.run(self.prec_sum_copy,
                                {self._input: train_data,
                                 self._target: train_labels})
        self.train_writer.add_summary(sum, epoch)
        errSum = self._session.run(self.error_sum, {self._input: train_data, self._target: train_labels})
        self.train_writer.add_summary(errSum, epoch)
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
                print("Epoch complete...old ", old_epoch)
                # self.save_checkpoint()
                self.validate(old_epoch)
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

        # self.save_checkpoint()
        return self._session.run(self.error,
                                 feed_dict={self._input: batch_input,
                                            self._target: batch_label})


def main():
    """ A main method that creates the model and starts training it """
    writer = Writer()
    writer.write(networkconfig)
    first_config = 0
    model = Model(networkconfig[first_config])
    model.train()
    model.train_writer.close()
    model.valid_writer.close()


if __name__ == "__main__":
    main()
