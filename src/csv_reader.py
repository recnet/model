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
from enum import Enum

import re
from src.config import cfg
from definitions import DATASETS_PATH
import pandas as pd
import os


class CsvReader(Enum):
    TESTING = "testing_data"
    TRAINING = "training_data"
    VALIDATION = "validation_data"

    @staticmethod
    def readfile(datatype):
        filepath = os.path.join(DATASETS_PATH, cfg['csv'][datatype.value])
        training_data = pd.read_csv(filepath, sep=",", encoding=cfg['csv']['encoding'])
        return training_data['headlines'].tolist(), training_data['users'].tolist(),

    @staticmethod
    def get_data(datatype):
        data, label = CsvReader.readfile(datatype)
        processed = [CsvReader._process(text) for text in data]
        return processed, label

    @staticmethod
    def _process(text):
        # replaces each number with NUMTOKEN
        text_with_token = re.sub('\s+NUMTOKEN\s+', ' NUMTOKEN ',
                                 re.sub('\d+', ' NUMTOKEN ', text))

        for character in ['!', '?', '-', '_', '.', ',', '\'', '\"', ':', ';', '%']:
            text_without_special = text_with_token.replace(character, '')

        return text_without_special
