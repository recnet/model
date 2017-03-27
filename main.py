import argparse
import tensorflow as tf
from model.util.networkconfig import yamlconfig as networkconfig

def main():
    """ A main method that creates the model and starts training it """
    # Parse arguments
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('configs', metavar='C', type=int, nargs='+',
                        help='Config number to use (can be multiple)')
    args = parser.parse_args()

    for conf in args.configs:
        config_file = networkconfig[conf]
        with tf.Session() as sess:
            network_model = get_model(config_file['type'], config_file, sess)
            network_model.train()
            network_model.close_writers()
        tf.reset_default_graph()

def get_model(model_type, config_file, session):
    """ Uses the config ID to return the correct model implementation """
    if model_type == "basic":
        from model.basic.basic_model import SoftmaxModel
        return SoftmaxModel(config_file, session)
    elif model_type == "logloss/basic":
        from model.logloss.basic_model_log_loss import LogLossModel
        return LogLossModel(config_file, session)
    elif model_type == "logloss/regularised/0-hidden":
        from model.logloss.model_log_loss_l2 import LogLossRegularised
        return LogLossRegularised(config_file, session)
    elif model_type == "logloss/regularised/1-hidden":
        from model.logloss.model_log_loss_l2_hid1 import LogLossRegularisedHidden
        return LogLossRegularisedHidden(config_file, session)
    else:
        from model.basic.basic_model import SoftmaxModel
        return SoftmaxModel(config_file, session)

if __name__ == "__main__":
    main()
