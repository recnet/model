import tensorflow as tf
import csv

def read(filePath, dataCols, labelCol):
    with open(filePath, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        dataFull = []
        labelFull = []
        for row in reader:
            data = ""
            label = row[labelCol]
            for elem in dataCols:
                col = row[elem]
                for character in ['!', '?', '-', '_', '.', ',', '\'', '\"', ':', ';']:
                    col = str(col)
                    col = col.replace(character, '')
                if col:
                    data += col + ", "
            dataFull.append(data.strip(' ').strip(','))
            labelFull.append(label)
        return [dataFull,labelFull]

# [data, labels] = read("../data/training_data.csv",[0],1)
#
# for i in range(len(data)):
#     print("Data: ", data[i])
#     print("Label: ", labels[i])
