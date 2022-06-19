from sklearn.metrics import classification_report
import csv
from pretty_confusion_matrix import pp_matrix_from_data
import numpy as np
import pandas as pd

true_labels =[]
pred_labels =[]
targets =['clase 1','clase 2','clase 3','clase 4','clase 5','clase 6','clase 7','clase 8','clase 9','clase 10']
cont=0
with open('testing.csv') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',')
    for row in spamreader:
        if (cont!=0):
            true_labels.append(row[0])
            pred_labels.append(row[1])
        cont+=1

print(list(map(int, true_labels)))
print(list(map(int, pred_labels)))

print(classification_report(list(map(int, true_labels)), list(map(int, pred_labels)), target_names=targets))

pp_matrix_from_data(np.array(true_labels), np.array(pred_labels), cmap='PuRd')