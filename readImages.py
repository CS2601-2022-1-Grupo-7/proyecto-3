#!/usr/bin/env python3

import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
import pywt
import os
import sys
import pandas as pd

def transfHaar(path, x):  
  LL = cv2.imread(path)
  # if LL is None:
  #   return
  LL = cv2.resize(LL, (512, 512))
  LL = cv2.cvtColor(LL, cv2.COLOR_BGR2GRAY)
  for i in range(x):
    LL, (LH, HL, HH) = pywt.dwt2(LL, 'haar')
  LL = LL.flatten()
  return LL

def draw(arr):
  n = int((len(arr))**0.5)
  data = np.zeros((n, n))
  for i in range(n):
    for j in range(n):
      data[i][j] = arr[i*n+j]
  fig = plt.figure(figsize=(6, 1.5))
  for i, a in enumerate([data]):
      ax = fig.add_subplot(1, 4, i + 1)
      ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
      ax.set_xticks([])
      ax.set_yticks([])
  fig.tight_layout()
  plt.show()

def class_butterfly(file):
  if (file == "001"):
    class_assign = "1000000000"
  elif (file == "002"):
    class_assign = "0100000000"
  elif (file == "003"):
    class_assign = "0010000000"
  elif (file == "004"):
    class_assign = "0001000000"
  elif (file == "005"):
    class_assign = "0000100000"
  elif (file == "006"):
    class_assign = "0000010000"
  elif (file == "007"):
    class_assign = "0000001000"
  elif (file == "008"):
    class_assign = "0000000100"
  elif (file == "009"):
    class_assign = "0000000010"
  elif (file == "010"):
    class_assign = "0000000001"
  return class_assign

img_path = sys.argv[1]

files_names = os.listdir(img_path)
dataset_butterfly = []
dataset_butterfly_names = []

haar = 6

for file in files_names:
  result = transfHaar(img_path + '/' + file, haar)
  dataset_butterfly.append(result)
  dataset_butterfly_names.append(file[0]+file[1]+file[2])

df = pd.DataFrame(list(zip(dataset_butterfly, dataset_butterfly_names)), columns =['vect', 'class']) 
train, validate, test = np.split(df.sample(frac=1, random_state=42), [int(.8*len(df)), int(.9*len(df))])

trainX = []
trainY = []
for index, row in train.iterrows():
  key = row['vect']
  key = key.tolist()
  value = row['class']
  trainX.append(key)
  trainY.append(class_butterfly(value))

testX = []
testY = []
for index, row in test.iterrows():
  key = row['vect']
  key = key.tolist()
  value = row['class']
  testX.append(key)
  testY.append(class_butterfly(value))

validateX = []
validateY = []
for index, row in validate.iterrows():
  key = row['vect']
  key = key.tolist()
  value = row['class']
  validateX.append(key)
  validateY.append(class_butterfly(value))

dictF = {'trainX': trainX, 'trainY': trainY, 'testX': testX, 'testY': testY, 'validateX': validateX, 'validateY': validateY}
json_final = json.dumps(dictF)

with open('/dev/stdout', 'w') as outfile:
  outfile.write(json_final)
