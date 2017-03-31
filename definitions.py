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

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

DATASETS_PATH = os.path.join(ROOT_DIR, "resources/datasets")
LOGS_DIR = os.path.join(ROOT_DIR, "logs")

TENSOR_DIR_TRAIN = 'tensorDir/train'
TENSOR_DIR_VALID = 'tensorDir/valid'
CHECKPOINTS_DIR = 'checkpoints'
LOGGING_RESULTS_FILE = os.path.join(LOGS_DIR, 'logs_results_all.csv')

NET_TYPE = 'type'
NET_NAME = 'name'
VOC_SIZE = 'vocabulary_size'
USER_COUNT = 'user_count'
LEARN_RATE = 'learning_rate'
EMBEDD_SIZE = 'embedding_size'
MAX_TITLE_LENGTH = 'max_title_length'
LSTM_NEURONS = 'lstm_neurons'
HIDDEN_NEURONS = 'hidden_neurons'
HIDDEN_LAYERS = 'hidden_layers'
BATCH_SIZE = 'batch_size'
TRAINING_EPOCHS = 'training_epochs'
USE_L2_LOSS = 'use_l2_loss'
L2_FACTOR = 'l2_factor'
USE_DROPOUT = 'use_dropout'
DROPOUT_PROB = 'dropout_prob'
USE_CONSTANT_LIMIT = 'use_constant_limit'
CONSTANT_PREDICTION_LIMIT = 'constant_prediction_limit'
TRAINABLE_MATRIX = 'trainable_matrix'
PRE_TRAINED_MATRIX = 'pre_trained_matrix'
USE_PRETRAINED = 'use_pretrained'
VALIDATION_DATA = 'validation_data'
TRAINING_DATA = 'training_data'
TESTING_DATA = 'testing_data'

F1_SCORE_TOP_VALID = 'F1 Score highest valid'
F1_SCORE_TRAIN = 'F1 Score training'
EPOCH_WHEN_F1_TOP = 'Epoch for highest F1'
PRECISION_VALIDATION = 'Precision validation'
PRECISION_TRAINING = 'Precision training'
RECALL_VALIDATION = 'Recall validation'
RECALL_TRAINING = 'Recall training'
DATE = 'Date of experiment'