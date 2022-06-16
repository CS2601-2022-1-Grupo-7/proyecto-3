import matplotlib.pyplot as plt
import numpy as np

error = []
errorTest = []
with open('error.txt') as f:
    error.append(f.readlines())

with open('errorTest.txt') as f:
    errorTest.append(f.readlines())

plt.plot(error[0])
plt.plot(errorTest[0])
# plt.gca().invert_yaxis()
plt.show()