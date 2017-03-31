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
import argparse
import tensorflow as tf
from model.util.networkconfig import yamlconfig as networkconfig
from model.model_builder import ModelBuilder

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
            builder = ModelBuilder(config_file, sess)
            network_model = builder.build()
            #network_model = get_model(config_file['type'], config_file, sess)
            network_model.train()
            network_model.close_writers()
        tf.reset_default_graph()

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
    elif model_type == 'logloss/pretrained-matrix':
        from model.logloss.model_log_loss_pretrained_matrix import Model
        return Model(config_file, session)
    else:
        from model.basic.basic_model import Model
        return Model(config_file, session)

if __name__ == "__main__":
    main()
