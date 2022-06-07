import matplotlib.pyplot as plt
import numpy as np

error = []
with open('error.txt') as f:
    error.append(f.readlines())

plt.plot(error[0])
plt.gca().invert_yaxis()
plt.show()