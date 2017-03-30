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

import os
import csv
import time
from definitions import LOGS_DIR, LOGGING_RESULTS_FILE

def log_config(config):
    filename = 'README.txt'
    networktype = config['type']
    name = config['name']
    dir_to_create = LOGS_DIR + '/' + networktype + '/' + name
    result = ''
    for key in config:
        result += "key: " + str(key) + "\tvalue: " + str(config[key]) + "\n"

    if not os.path.exists(dir_to_create):
        raise FileNotFoundError('Can not write because no directory is created')

    with open(dir_to_create+'/'+filename, "w+") as f:
        f.write(result)


def log_samefile(config, f1_score, epoch_top, prec_valid, prec_train, recall_valid, recall_train):
    NET_TYPE = 'type'
    NET_NAME = 'name'
    VOC_SIZE = 'vocabulary_size'
    USER_COUNT = 'user_count'
    LEARN_RATE = 'learning_rate'
    EMBEDD_SIZE = 'embedding_size'
    MAX_TITLE_LENGTH = 'max_title_length'
    LSTM_NEURONS = 'lstm_neurons'
    HIDDEN_NEURONS = 'hidden_neurons'
    BATCH_SIZE = 'batch_size'
    TRAINING_EPOCHS = 'training_epochs'
    USERS_TO_SELECT ='users_to_select'
    USE_L2_LOSS = 'use_l2_loss'
    L2_FACTOR = 'l2_factor'
    USE_DROPOUT = 'use_dropout'
    DROPOUT_PROB = 'dropout_prob'
    USE_CONSTANT_LIMIT = 'use_constant_limit'
    CONSTANT_PREDICTION_LIMIT = 'constant_prediction_limit'
    TRAINABLE_MATRIX = 'trainable_matrix'
    PRE_TRAINED_DIMENSION = 'pre_trained_dimension'
    PRE_TRAINED_MATRIX = 'pre_trained_matrix'
    USE_PRETRAINED = 'use_pretrained'
    VALIDATION_DATA = 'validation_data'
    TRAINING_DATA = 'training_data'
    TESTING_DATA = 'testing_data'

    F1_SCORE_TOP = 'F1 Score highest'
    EPOCH_WHEN_F1_TOP = 'Epoch for highest F1'
    PRECISION_VALIDATION = 'Precision validation'
    PRECISION_TRAINING = 'Precision training'
    RECALL_VALIDATION = 'Recall validation'
    RECALL_TRAINING = 'Recall training'
    DATE = 'Date of experiment'

    filename = LOGGING_RESULTS_FILE
    if not os.path.exists(LOGS_DIR):
        raise FileNotFoundError('Can not write because no directory is created')

    config_headers = [NET_TYPE, NET_NAME, VOC_SIZE, USER_COUNT, LEARN_RATE, EMBEDD_SIZE, MAX_TITLE_LENGTH, LSTM_NEURONS,
               HIDDEN_NEURONS, BATCH_SIZE, TRAINING_EPOCHS, USERS_TO_SELECT, USE_L2_LOSS, L2_FACTOR, USE_DROPOUT,
               DROPOUT_PROB, USE_CONSTANT_LIMIT, CONSTANT_PREDICTION_LIMIT, TRAINABLE_MATRIX, PRE_TRAINED_DIMENSION,
               PRE_TRAINED_MATRIX, USE_PRETRAINED, VALIDATION_DATA, TRAINING_DATA, TESTING_DATA]

    additional_headers = [F1_SCORE_TOP, EPOCH_WHEN_F1_TOP, PRECISION_VALIDATION, PRECISION_TRAINING,
                          RECALL_VALIDATION, RECALL_TRAINING, DATE]

    headers = config_headers + additional_headers

    if not os.path.isfile(filename):
        with open(filename, 'w+') as filedata:
            writer = csv.DictWriter(filedata, delimiter=',', fieldnames=headers)
            writer.writeheader()

    data = []
    for header in config_headers:
        data.append(config[header])

    time_logged = time.strftime("%c")
    data.append(str(f1_score))
    data.append(str(epoch_top))
    data.append(str(prec_valid))
    data.append(str(prec_train))
    data.append(str(recall_valid))
    data.append(str(recall_train))
    data.append(time_logged)

    with open(filename, 'a+', newline='') as fp:
        a = csv.writer(fp, delimiter=',')
        a.writerow(data)

log_samefile(None, None, None, None, None, None, None)
