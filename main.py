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
import sys
import argparse
import tensorflow as tf
from definitions import *
from application import Application
from model.util.networkconfig import yamlconfig as networkconfig
from model.model_builder import ModelBuilder

def main():
    """ A main method that creates the model and starts training it """
    # Parse arguments
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('configs', metavar='C', type=int, nargs='*',
                        help='Config number to use (can be multiple)')
    parser.add_argument('--application', action='store_true')
    args = parser.parse_args()
    if args.application:
        conf_num = args.configs[0] if args.configs else 0
        serve_application(conf_num)
    else:
        for conf in args.configs if args.configs else range(len(networkconfig)):
            try:
                print("Starting config ", conf)
                config_file = networkconfig[conf]
                with tf.Session() as sess:
                    builder = ModelBuilder(config_file, sess)

                    network_model = builder.build()
                    if config_file[USE_PRETRAINED_NET]:
                        network_model.train(USE_PRETRAINED_NET)
                    network_model.train()
                    network_model.close_writers()
                tf.reset_default_graph()
            except Exception as e:
                print("Config ", networkconfig[conf]["name"], "failed to complete", file=sys.stderr)
                print(e, file=sys.stderr)
                tf.reset_default_graph()

def serve_application(config=0):
    """ Serves a simple API for making predictions on a specified model """
    config_file = networkconfig[config]
    with tf.Session() as sess:
        builder = ModelBuilder(config_file, sess)
        Application(builder.build(), sess)

if __name__ == "__main__":
    main()
