# data analysis and wrangling

import numpy as np
# import pandas as pd

# data visualization

import matplotlib.pyplot as plt
# import seaborn as sns


# Reads the data
train_A1 = np.load('data/train/D-1.npy')
test_A1 = np.load('data/test/R-1.npy')
train_A1.shape

fig = plt.figure()
fig.add_axes()
ax = fig.add_subplot(111)  # row-col-num

# x = linspace(0, length)
y = train_A1[:, 0]
ax.plot(y)
