import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker
from matplotlib.pyplot import figure
import seaborn as sns
import h5py
from pandas import DataFrame


def show1():
    sns.set(font_scale=1.5)
    fpath = 'Data/TaxiBJ/BJ16_M32x32_T30_InOut.h5'
    with h5py.File(fpath) as f:
        flow_data = f['data'][()]
        time_data = f['date'][()]
    data = flow_data[20][0]
    cmap = sns.diverging_palette(80, 150, 90, 60, as_cmap=True)
    sns.heatmap(data=data, square=True, cmap='RdBu_r')
    plt.show()


def show2(j):
    fig, ax = plt.subplots(1, 1)
    fig.figsize=(25, 10)
    fpath = 'Data/TaxiBJ/BJ15_M32x32_T30_InOut.h5'
    with h5py.File(fpath) as f:
        flow_data = f['data'][()]
        time_data = f['date'][()]
    plt.xticks(rotation=45)
    front = 10 + 336 * (j - 1)
    back = 10 + 336 * j
    data = flow_data[front: back, 0, 5, 13]
    time_data = time_data[front: back]
    plt.plot(time_data, data)
    _xtick_labels = ['Mon' for i in range(48)] + ['Tue' for i in range(48)] + ['Wen' for i in range(48)] + \
                    ['Thu' for i in range(48)] + ['Fri' for i in range(48)] + ['Sat' for i in range(48)] + \
                    ['Sun' for i in range(48)]
    plt.xticks(time_data, _xtick_labels)
    tick_spacing = 48
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    plt.ylabel("Traffic Flow")
    plt.show()


for j in range(1, 20):
    show2(j)

