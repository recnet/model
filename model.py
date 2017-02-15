import glob
import os.path
import tensorflow as tf

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
        self.batch_size = 50
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
                [self.vocabulary_size, self.embedding_size], -1.0, 1.0),
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
        output = tf.reshape(tf.concat_v2(outputs, 1), [-1, self.lstm_neurons])

        # Feed the output of the LSTM layer to a softmax layer
        self.softmax_weights = tf.Variable(tf.random_normal([self.lstm_neurons,
                                                             self.user_count],
                                                            stddev=0.35,
                                                            dtype=tf.float64),
                                           name="weights")

        self.softmax_bias = tf.Variable(tf.random_normal([self.user_count],
                                                         stddev=0.35,
                                                         dtype=tf.float64),
                                        name="biases")

        self.softmax = tf.nn.softmax(tf.matmul(output, self.softmax_weights) +
                                     self.softmax_bias)
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

        # Save model when done training
        self.save_checkpoint()
