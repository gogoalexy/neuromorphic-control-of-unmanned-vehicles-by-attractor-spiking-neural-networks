import argparse
from itertools import zip_longest
import subprocess

import matplotlib.pyplot as plt
import numpy as np

time = np.linspace(0, 3, 60)

def getBumps(spike_ratios, threshold):
    """Find the intervals of spike ratios are greater than the threshold."""
    bump_durations = []
    winning_streak = False
    start, end = None, None
    if threshold < 0.5 or threshold > 1.0:
        print('Illegal threshold')
        return bump_durations
    for num, ratio in enumerate(spike_ratios):
        if winning_streak:
            if ratio > threshold:
                continue
            else:
                winning_streak = False
                end = num - 1
                bump_durations.append((start, end))
                start, end = None, None
        else:
            if ratio > threshold:
                start = num
                winning_streak = True

    if start != None and end == None:
        end = len(spike_ratios) - 1
        bump_durations.append((start, end))
    return bump_durations

def getTransition(bump_series0, bump_series1):
    """Find the transitions between two series of bump durations. Assume no overlap of bump duration."""
    transitions = []
    diff, direction = None, None
    for series0 in bump_series0:
        start0, end0 = series0
        for series1 in bump_series1:
            start1, end1 = series1
            if start0 > start1:
                direction = '-'
                diff = start0 - end1
                transition_point = start0
            else:
                direction = '+'
                diff = start1 - end0
                transition_point = start1
            if diff < 3:
                transitions.append((transition_point, direction))
    return transitions

def getNumberConsecutiveSuccessTransitions(transitions, ground_truth):
    counter = 0
    for real, expect in zip(transitions, ground_truth):
        if (real[0] - expect[0]) < 3 and (real[1] == expect[1]):
            counter += 1
    return counter

def calculateRobustness():
    pass


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
    time = np.linspace(0, 3, 60)
    
    
    time_window = 0.05
    transition_ground_truth = [(20, '+'), (30, '+'), (40, '+'), (50, '+')]

    x = [0, 0, 0, 0, 0]

    subprocess.call([flysim_target, '-conf', args.conf, '-pro', args.pro, '-rp', str(args.num_trial)])
    for trial in range(args.num_trial):
        task_spike_ratios = [[] for i in range(5)]
        spike_counts = [0 for i in range(5)]
        time_boundary = 0.05
        if trial == 0:
            filename = 'task.dat'
        else:
            filename = f'task.dat_{trial+1}'
        with open(filename, 'r') as spike_file:
            for event in spike_file:
                timestamp, neuron = event.split(' ')
                timestamp, neuron = float(timestamp), int(neuron)
                if time_boundary > timestamp:
                    spike_counts[neuron//10-12] += 1
                else:
                    time_boundary += time_window
                    total_spike_count = sum(spike_counts) if sum(spike_counts) != 0 else 1
                    spike_ratios = [count/total_spike_count for count in spike_counts]
                    for num, ratio in enumerate(spike_ratios):
                        task_spike_ratios[num].append(ratio)
                        spike_counts = [0 for i in range(5)]
        
            while time_boundary < 3:
                time_boundary += time_window
                for num in range(5):
                    task_spike_ratios[num].append(0)

            b0 = getBumps(task_spike_ratios[0], 0.5)
            b1 = getBumps(task_spike_ratios[1], 0.5)
            t0 = getTransition(b0, b1)
            b2 = getBumps(task_spike_ratios[2], 0.5)
            t1 = getTransition(b1, b2)
            b3 = getBumps(task_spike_ratios[3], 0.5)
            t2 = getTransition(b2, b3)
            b4 = getBumps(task_spike_ratios[4], 0.5)
            t3 = getTransition(b3, b4)
            t0.extend(t1)
            t0.extend(t2)
            t0.extend(t3)
            success = getNumberConsecutiveSuccessTransitions(t0, transition_ground_truth)
            for i in range(1, success+1):
                x[i] += 1

    print(f'Y1={x[1]/100}')
    print(f'Y2={x[2]/100}')
    print(f'Y3={x[3]/100}')
    print(f'Y4={x[4]/100}')
    

    