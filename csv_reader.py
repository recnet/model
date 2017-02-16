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
#==============================================================================
import csv

def read(file_path, data_column, label_column):
    """ A function that reads the data and
    corresponding label from a CSV file """
    with open(file_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        data_full = []
        label_full = []
        for row in reader:
            data = ""
            label = row[label_column]
            for elem in data_column:
                col = row[elem]
                for character in ['!', '?', '-', '_', '.', ',', '\'', '\"', ':', ';']:
                    col = str(col)
                    col = col.replace(character, '')
                if col:
                    data += col + ", "
            label = label.replace(',', '')
            data_full.append(data.strip(' ').strip(','))
            label_full.append(label)
        return [data_full, label_full]

[data, labels] = read("../data/training_data.csv",[0],1)

for i in range(len(data)):
     print("Data: ", data[i])
     print("Label: ", labels[i])
