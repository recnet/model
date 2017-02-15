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
            data_full.append(data.strip(' ').strip(','))
            label_full.append(label)
        return [data_full, label_full]

# [data, labels] = read("../data/training_data.csv",[0],1)
#
# for i in range(len(data)):
#     print("Data: ", data[i])
#     print("Label: ", labels[i])
