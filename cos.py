import argparse

import matplotlib.pyplot as plt
import numpy as np

from ssm import SNNStateMachine

if __name__ == '__main__':

    for weight in np.logspace(0.1, 10, 10):
        results = []
        for strength in range(100, 1000, 100):
            row = []
            for duration in range(50, 500, 50):
                ssc = SNNStateMachine(2, transitions=1, task_weights=[weight, weight], experiment_time=1000, repetition=1)
                ssc.setTransitionPeriod(500, duration, 500)
                ssc.generateTransitionStimuli('spike', 'AMPA', strength)
                ssc.spawnAttractor(100, 50, 'spike', 'AMPA', 400)
                ssc.startSimulation()
                x0, x1 = ssc.calculateRobustness()
                row.append(x1)
                #ssc.plotRaster()
            results.insert(0, row)

        fig = plt.figure()
        ax = fig.add_subplot()
        im = ax.imshow(np.array(results))
        plt.show()
