import argparse
import subprocess

import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--conf', type=str, help='Path to conf file')
    parser.add_argument('-p', '--pro', type=str, help='Path to pro file')
    parser.add_argument('-n', '--num-trial', type=int, help='Number of trials')
    parser.add_argument('-s', '--sim-version', type=int, help='Simulator version')
    args = parser.parse_args()

    if args.sim_version == 7:
        flysim_target = 'simulator/flysim7_21_3.out'
    elif args.sim_version == 8:
        flysim_target = 'simulator/flysim8_15.out'
    else:
        print('Default Flysim version: 8')
        flysim_target = 'simulator/flysim8_15.out'

    timer = 0.0
    time = np.linspace(0, 3, 600)
    ord_spikes = [[] for i in range(50)]
    ord_counts = [[0] for i in range(5)]
    last_time = 0.0
    counts = [0 for i in range(5)]

    for trial in range(args.num_trial):
        subprocess.call([flysim_target, '-conf', args.conf, '-pro', args.pro])
        #with open('ordinal.dat', 'r') as rate_file:
        #    for line in rate_file:
        #        line_parse = line.split(' ')
        #        t.append(float(line_parse[0]))
        #        ord0.append(float(line_parse[1]))
        #        ord1.append(float(line_parse[2]))
        #        ord2.append(float(line_parse[3]))
        #        ord3.append(float(line_parse[4]))
        #        ord4.append(float(line_parse[5]))
        with open('ordinal.dat', 'r') as spike_file:
            for event in spike_file:
                t, neuron = event.split(' ')
                ord_spikes[int(neuron)].append(float(t))

                while (float(t) - timer) >= 0:
                    timer += 0.0001
                    if (timer - last_time) >= 0.005:
                        last_time = timer
                        for num, count in enumerate(counts):
                            ord_counts[num].append(count)
                        counts = [0 for i in range(5)]
                
                counts[int(neuron)//10] += 1

        for ele in ord_counts:
            while len(ele) < 600:
                ele.append(0)
                    
    fig, axs = plt.subplots(2, 1)
    colors1 = [f'C{i//10}' for i in range(50)]
    axs[0].set_xlim(0, 3)
    axs[0].eventplot(ord_spikes, linelengths = 0.8, linewidths = 1.0, colors = colors1)
    axs[1].plot(time, ord_counts[0], time, ord_counts[1], time, ord_counts[2], time, ord_counts[3], time, ord_counts[4])
    plt.show()

