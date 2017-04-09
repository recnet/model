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
from definitions import *


def log_samefile(config, f1_score_valid, f1_score_train, epoch_top, prec_valid, prec_train, recall_valid, recall_train):
    filename = LOGGING_RESULTS_FILE
    if not os.path.exists(LOGS_DIR):
        raise FileNotFoundError('Can not write because no directory is created')

    config_headers = [NET_TYPE, NET_NAME, VOC_SIZE, USER_COUNT, LEARN_RATE, EMBEDD_SIZE, MAX_TITLE_LENGTH, RNN_NEURONS, RNN_UNIT,
               HIDDEN_NEURONS, HIDDEN_LAYERS, USE_CONCAT_INPUT, BATCH_SIZE, TRAINING_EPOCHS, USE_L2_LOSS, L2_FACTOR, USE_DROPOUT,
               DROPOUT_PROB, USE_CONSTANT_LIMIT, CONSTANT_PREDICTION_LIMIT, TRAINABLE_MATRIX,
               PRE_TRAINED_MATRIX, USE_PRETRAINED, VALIDATION_DATA, TRAINING_DATA, TESTING_DATA]

    additional_headers = [F1_SCORE_TOP_VALID, F1_SCORE_TRAIN, EPOCH_WHEN_F1_TOP, PRECISION_VALIDATION, PRECISION_TRAINING,
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
    #Order of appends must match that of headers.
    data.append(str(f1_score_valid))
    data.append(str(f1_score_train))
    data.append(str(epoch_top))
    data.append(str(prec_valid))
    data.append(str(prec_train))
    data.append(str(recall_valid))
    data.append(str(recall_train))
    data.append(time_logged)

    with open(filename, 'a+', newline='') as fp:
        a = csv.writer(fp, delimiter=',')
        a.writerow(data)
