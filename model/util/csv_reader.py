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
import os
import csv
import numpy as np
from definitions import DATASETS_PATH


class Dataenum(Enum):
    TESTING = "testing_data"
    TRAINING = "training_data"
    VALIDATION = "validation_data"

class CsvReader:
    def __init__(self, netcfg):
        self.netcfg = netcfg
        self.encoding = 'UTF-8'

    def get_data(self, datatype, data_column=[0], sub_column=1, label_column=2):
        """ A function that reads the data and
        corresponding label from a CSV file """
        file_path = os.path.join(DATASETS_PATH, self.netcfg[datatype.value])
        with open(file_path, 'r', encoding=self.encoding) as csvfile:
            reader = csv.reader(csvfile)
            data_full = []
            subreddit_full = []
            label_full = []
            for row in reader:
                if len(row) < 3:
                    print(len(row))
                    print(row)
                    print(datatype)
                data = ""
                subreddit = row[sub_column]
                label = row[label_column]
                for elem in data_column:
                    col = row[elem]
                    # replaces each number with NUMTOKEN
                    col = re.sub('\d+', 'NUMTOKEN', col)
                    col = re.sub('\s+NUMTOKEN\s+', ' NUMTOKEN ', col)

                    for character in ['!', '?', '-', '_', '.', ',', '\'', '\"', ':', ';', '%', '(', ')']:
                        col = str(col)
                        col = col.replace(character, '')
                    if col:
                        data += col + ", "
                label = label.replace(',', '')
                data_full.append(data.strip(' ').strip(','))
                label_full.append(label)
                subreddit_full.append(subreddit)
            return data_full, subreddit_full, label_full

    def load_pretrained_embeddings(self, file_name, dimension_size=50):
        file_path = os.path.join(DATASETS_PATH, file_name)
        matrix = [np.random.rand(dimension_size).astype(np.float32)]
        word_dict = dict()
        word_dict['UNK'] = 0
        count = 1

        with open(file_path, 'r', encoding='UTF-8') as csvfile:
            reader = csv.reader(csvfile, delimiter=' ', quoting=csv.QUOTE_NONE)

            for row in reader:
                first_col = row[0]
                if first_col in ['!', '?', '-', '_', '.', ',', '\'', '\"', ':', ';', '%', '(', ')']:
                    continue

                # Some words are tokens for usernames like <user> or
                # <caps> etc, ignore them.
                if first_col[0] == '<':
                    continue

                word_dict[first_col] = count
                row_array = np.array(row[1:], dtype=np.float32)
                matrix.append(row_array)
                count += 1
        return word_dict, np.array(matrix)



