import argparse

from matplotlib import colors
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

from ssc import SNNSequenceControl

if __name__ == '__main__':

    for duration in [50, 100, 150, 200, 300, 500]:
        results1 = []
        results2 = []
        #for weight in np.logspace(0.1, 10, 10):
        for strength in range(0, 900, 50):
            row1 = []
            row2 = []
            for weight in np.arange(0.0, 0.5, 0.01):
                ssc = SNNSequenceControl(3, transitions=2, task_weights=[weight, weight, weight], experiment_time=2000, repetition=10)
                ssc.setTransitionPeriod(300, duration, 700)
                ssc.generateTransitionStimuli('spike', 'AMPA', strength)
                ssc.spawnAttractor(100, 50, 'spike', 'AMPA', 400)
                ssc.startSimulation()
                x0, x1, x2 = ssc.calculateRobustness()
                row1.append(x1/10)
                row2.append(x2/10)
                #ssc.plotRaster()
            results1.insert(0, row1)
            results2.insert(0, row2)

        fig, axs = plt.subplots(1, 2)
        images = []
        images.append(axs[0].imshow(np.array(results1), cmap='inferno', extent=[-0.005,0.495,-25,875], aspect='auto'))
        axs[0].label_outer()
        images.append(axs[1].imshow(np.array(results2), cmap='inferno', extent=[-0.005,0.495,-25,875], aspect='auto'))
        axs[1].label_outer()
    
        vmin = min(image.get_array().min() for image in images)
        vmax = max(image.get_array().max() for image in images)
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        for im in images:
            im.set_norm(norm)
        cbar = fig.colorbar(images[0], ax=axs, orientation='vertical')
        #cbar = fig.colorbar(cm.ScalarMappable(norm=None, cmap='inferno'), ax=axs)
        cbar.ax.get_yaxis().labelpad = 15
        cbar.ax.set_ylabel('Robustness', rotation=270)
        #im1 = axs.imshow(np.array(results2), cmap='inferno', extent=[0.25,9.75,25,875], aspect='auto')
        axs[0].set_xlabel('Task weight')
        axs[1].set_xlabel('Task weight')
        axs[0].set_ylabel("Stimulus strength to 'Status' (Hz)")
        axs[0].set_xticks([i for i in np.arange(0, 0.5, 0.1)])
        axs[1].set_xticks([i for i in np.arange(0, 0.5, 0.1)])
        axs[0].set_yticks([i for i in range(0, 900, 100)])
        axs[1].set_yticks([i for i in range(0, 900, 100)])
        #im2 = axs[0, 1].imshow(np.array(results2), cmap='inferno', extent=[0.25,9.75,25,775], aspect='auto')
        axs[0].set_title('$Y^1$')
        axs[1].set_title('$Y^2$')
        #plt.show()
        plt.savefig(f'cos_scan_duration_small_{duration}.png')
