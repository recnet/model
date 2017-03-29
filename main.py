import argparse
import tensorflow as tf

from model.logloss.model_log_loss_l2_hid12 import Model
from model.util.networkconfig import yamlconfig as networkconfig
from model.util.data import Data


def main():
    """ A main method that creates the model and starts training it """
    # # Parse arguments
    # parser = argparse.ArgumentParser(add_help=True)
    # parser.add_argument('configs', metavar='C', type=int, nargs='+',
    #                     help='Config number to use (can be multiple)')
    # args = parser.parse_args()

    # for conf in args.configs:
    #     config_file = networkconfig[conf]
    #     with tf.Session() as sess:
    #         network_model = get_model(config_file['type'], config_file, sess)
    #         network_model.train()
    #         network_model.close_writers()
    #    tf.reset_default_graph()

    cfg = networkconfig[0]

    target = tf.placeholder(tf.float64, [None, cfg['user_count']], name="target")
    data = tf.placeholder(tf.int32, [None, cfg['max_title_length']], name="data")
    db = Data(networkconfig[0])

    batch_input, batch_label = db.next_train_batch()

    print(type(batch_input))
    print(type(batch_label))

    m = Model(networkconfig[0], data, target)

    with tf.Session() as sess:
        sess.run(m.optimize, feed_dict={input: batch_input, target: target})


def train_batch(self):
    batch_input, batch_label = self.data.next_train_batch()
    self._session.run(self.train_op,
                      {self._input: batch_input,
                       self._target: batch_label,
                       self._keep_prob: self.dropout_prob})

    # self.save_checkpoint()
    return self._session.run(self.error, feed_dict={
        self._input: batch_input,
        self._target: batch_label,
        self._keep_prob: 1.0})


def get_model(model_type, config_file, session):
    """ Uses the config ID to return the correct model implementation """
    if model_type == "basic":
        from model.basic.basic_model import Model
        return Model(config_file, session)
    elif model_type == "logloss/basic":
        from model.logloss.basic_model_log_loss import Model
        return Model(config_file, session)
    elif model_type == "logloss/regularised/0-hidden":
        from model.logloss.model_log_loss_l2 import Model
        return Model(config_file, session)
    elif model_type == "logloss/regularised/1-hidden":
        from model.logloss.model_log_loss_l2_hid1 import Model
        return Model(config_file, session)
    else:
        from model.basic.basic_model import Model
        return Model(config_file, session)


if __name__ == "__main__":
    main()
