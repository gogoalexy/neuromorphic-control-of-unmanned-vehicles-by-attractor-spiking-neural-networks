import argparse

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

from ssc import SNNSequenceControl

if __name__ == '__main__':

    duration = 150
    results = []
    #for weight in np.logspace(0.1, 10, 10):
    for strength in range(50, 800, 50):
        row = []
        for weight in range(1, 10, 1):
            ssc = SNNSequenceControl(2, transitions=1, task_weights=[weight, weight], experiment_time=1000, repetition=100)
            ssc.setTransitionPeriod(500, duration, 500)
            ssc.generateTransitionStimuli('spike', 'AMPA', float(strength))
            ssc.spawnAttractor(100, 50, 'spike', 'AMPA', 400)
            ssc.startSimulation()
            x0, x1 = ssc.calculateRobustness()
            row.append(x1)
            #ssc.plotRaster()
        results.insert(0, row)

    fig = plt.figure()
    ax = fig.add_subplot()
    cbar = fig.colorbar(cm.ScalarMappable(norm=None, cmap='inferno'), ax=ax)
    cbar.ax.get_yaxis().labelpad = 15
    cbar.ax.set_ylabel('Robustness', rotation=270)
    im = ax.imshow(np.array(results), cmap='inferno', extent=[0.5,9.5,25,775], aspect='auto')
    plt.xlabel('Task weight')
    plt.ylabel("Stimulus strength to 'Status' (Hz)")
    plt.xticks([float(i) for i in range(1, 10, 1)])
    plt.yticks([i for i in range(50, 800, 50)])
    plt.show()
