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
'''
The specified structure will look like this:

logs/type/name/
inside name folder there will be a tensor-folder with train and validation folders
inside name folder there will be a text file including parameters for the network

'''
import os
from definitions import TENSOR_DIR_TRAIN, TENSOR_DIR_VALID, LOGS_DIR, CHECKPOINTS_DIR

def build_structure(config):
    '''
    :param config: Dictionary of configurations
    :return: paths to logging directory
    '''
    type = config['type']
    name = config['name']
    dir_to_create = LOGS_DIR + '/' + type + '/' + name
    tensor_dir_train_to_create = dir_to_create + '/' + TENSOR_DIR_TRAIN
    tensor_dir_valid_to_create = dir_to_create + '/' + TENSOR_DIR_VALID
    checkpoints_dir_to_create = dir_to_create + '/' + CHECKPOINTS_DIR

    if not os.path.exists(tensor_dir_train_to_create):
        os.makedirs(tensor_dir_train_to_create)

    if not os.path.exists(tensor_dir_valid_to_create):
        os.makedirs(tensor_dir_valid_to_create)

    if not os.path.exists(checkpoints_dir_to_create):
        os.makedirs(checkpoints_dir_to_create)

    return dir_to_create
