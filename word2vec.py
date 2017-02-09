import collections

filename = input("Enter filename, words needs to be seperated with spaces > ")
word_list = open(filename, 'r').read().split()
print('Data size', len(word_list))

vocabulary_size = 50000

def build_dataset(words):
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

data, count, dictionary, reverse_dictionary = build_dataset(word_list)

print(str(data))
print(str(dictionary))
print(str(reverse_dictionary))
