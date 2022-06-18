#!/usr/bin/env python3

import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
import pywt
import os
import sys
import pandas as pd
from sklearn import preprocessing
import random

def transfHaar(path, mask_path, x):  
  LL = cv2.bitwise_and(cv2.imread(path), cv2.imread(mask_path))
  LL = cv2.cvtColor(LL, cv2.COLOR_RGB2GRAY)
  LL = cv2.resize(LL, (512, 512))
  for _ in range(x):
    LL, (LH, HL, HH) = pywt.dwt2(LL, 'haar')
  LL = LL.flatten()
  return LL

img_path = sys.argv[1]
seg_path = os.path.join(img_path, "segmentations")

files_names = os.listdir(img_path)
dataset_butterfly = []
dataset_butterfly_names = []

haar = 5

for file in os.scandir(os.path.join(img_path, "images")):
  name, ext = os.path.splitext(file.name)
  result = transfHaar(file.path, os.path.join(seg_path, f"{name}_seg0{ext}"), haar)
  dataset_butterfly.append(result)
  dataset_butterfly_names.append(file.name[:3])

dataset_butterfly = preprocessing.MinMaxScaler().fit_transform(dataset_butterfly)

df = pd.DataFrame(list(zip(dataset_butterfly, dataset_butterfly_names)), columns =['vect', 'class']) 
train, validate, test = np.split(df.sample(frac=1, random_state=random.randint(10, 50)), [int(.8*len(df)), int(.9*len(df))])

trainX = []
trainY = []
for index, row in train.iterrows():
  key = row['vect']
  key = key.tolist()
  value = row['class']
  trainX.append(key)
  trainY.append(value)

testX = []
testY = []
for index, row in test.iterrows():
  key = row['vect']
  key = key.tolist()
  value = row['class']
  testX.append(key)
  testY.append(value)

validateX = []
validateY = []
for index, row in validate.iterrows():
  key = row['vect']
  key = key.tolist()
  value = row['class']
  validateX.append(key)
  validateY.append(value)

dictF = {'trainX': trainX, 'trainY': trainY, 'testX': testX, 'testY': testY, 'validateX': validateX, 'validateY': validateY}
json_final = json.dumps(dictF)

with open('/dev/stdout', 'w') as outfile:
  outfile.write(json_final)
