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
import collections
import numpy as np

def test_with_custom_file(filename):
    """ Tests the build_dataset function """
    word_list = open(filename, 'r').read().split()
    print('Data size', len(word_list))
    data, count, dictionary, reverse_dictionary = build_dataset(word_list)

    print(str(count))
    print(str(data))
    print(str(dictionary))
    print(str(reverse_dictionary))

def build_dataset(words, vocabulary_size=50000):
    """ Builds a dictionary from given words """
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reverse_dictionary

def create_matrix(dictionary):
    """ Creates an identity matrix with size of dictionary """
    to_len = len(dictionary)
    matrix = np.identity(to_len)
    matrix[0][0] = 0 #this will be kinda "default vec" for 'UNK'
    return matrix

def get_indicies(sentence, dictionary, max_words):
    """ Turns a sentence into a vector of word IDs """
    # This assumes that the sentence is pre-processed as an array of words
    result = [0] * max_words
    for i, word in enumerate(sentence):
        if i > max_words-1:
            return result
        if word in dictionary:
            result[i] = dictionary[word]
        else:
            result[i] = 0
    return result

def label_vector(users, dic, max_users):
    """ Turns an array of some users into an array
    with ones on those users indicies """
    vector = [0]*max_users
    for user in users:
        if user in dic:
            vector[dic[user]] = 1
    return vector
