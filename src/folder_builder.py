'''
The specified structure will look like this:

logs/type/name/
inside name folder there will be a tensor-folder with train and validation folders
inside name folder there will be a text file including parameters for the network

'''
import os
from definitions import TENSOR_DIR_TRAIN, TENSOR_DIR_VALID, LOGS_DIR

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
    print(tensor_dir_train_to_create)
    print(tensor_dir_valid_to_create)
    if not os.path.exists(tensor_dir_train_to_create):
        os.makedirs(tensor_dir_train_to_create)

    if not os.path.exists(tensor_dir_valid_to_create):
        os.makedirs(tensor_dir_valid_to_create)

    return dir_to_create