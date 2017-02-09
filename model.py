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
        self.batch_size = 50
        self.build_graph()
        self.load_checkpoint()

    def build_graph(self):
        """ Builds the model in tensorflow """

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
