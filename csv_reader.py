import tensorflow as tf
import pandas as pd

def read(filePath,dataCols,labelCols):
    data = pd.read_csv(filePath, usecols = dataCols,header=None).values
    labels = pd.read_csv(filePath, usecols = labelCols,header=None).values

    for j in range(len(data)):
        row = data[j]
        for i in range(len(row)):
            col = row[i]
            for character in ['!','?','-','_','.',',','\'','\"',':',';']:
                col = col.replace(character, '')
            row[i] = col
        data[j] = row
    return [data,labels]

# [data,labels] = read("../csv/processeddatalatest.csv",[0],[1])

# for i in range(len(data)):
#     print("Data: ", data[i])
#     print("Label: ", labels[i])
