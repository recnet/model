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
from definitions import DATASETS_PATH
import pandas as pd
import os


class Dataenum(Enum):
    TESTING = "testing_data"
    TRAINING = "training_data"
    VALIDATION = "validation_data"


class CsvReader:
    def __init__(self, netcfg):
        self.netcfg = netcfg
        self.encoding = 'UTF-8'

    def readfile(self, datatype):
         filepath = os.path.join(DATASETS_PATH, self.netcfg[datatype.value])
         training_data = pd.read_csv(filepath, sep=",", encoding=self.encoding)
         return training_data['headlines'].tolist(), training_data['users'].tolist(),

    def get_data(self, datatype):
        data, label = self.readfile(datatype)
        processed = [self._process(text) for text in data]
        return processed, label

    def _process(self, text):
        # replaces each number with NUMTOKEN
        text_with_token = re.sub('\s+NUMTOKEN\s+', ' NUMTOKEN ',
                                 re.sub('\d+', ' NUMTOKEN ', text))

        for character in ['!', '?', '-', '_', '.', ',', '\'', '\"', ':', ';', '%']:
            text_without_special = text_with_token.replace(character, '')

        return text_without_special
