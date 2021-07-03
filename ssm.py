import sys
import numpy as np

from flysim_format import FlysimSNN

class SNNStateMachine:

    def __init__(self, length, fork=False, cos=False, transitions=0, experiment_time=1000, repetition=100):
        self.length = length
        self.transitions = transitions
        self.enableFork = fork
        self.enableCoS = cos
        self.sim = FlysimSNN(experiment_time, repetition)time)
        self.stimulus = {'total_time': experiment_time}
        self.time_window = 0.05
        self.spawnNodes()
        self.assembleNodes()

    def spawnNodes(self):
        self.sim.addNeuron('OrdinalInh')
        self.sim.addNeuron('TaskInh')
        self.sim.addNeuron('Next')
        for id in range(self.length):
            self.declareNodeNeurons(id)
            self.sim.addCoonection(f'Ordinal{id}', 'OrdinalInh', 'AMPA')
            self.sim.addCoonection(f'Task{id}', 'TaskInh', 'AMPA')
            self.sim.addCoonection('Next', f'Shifter{id}', 'AMPA')
            self.sim.addCoonection('OrdinalInh', f'Ordinal{id}', 'GABA')
            self.sim.addCoonection('TaskInh', f'Task{id}', 'GABA')

    def declareNodeNeurons(self, id):
        self.sim.addNeuron(f'Ordinal{id}')
        self.sim.addNeuron(f'Task{id}')
        self.sim.addNeuron(f'Shifter{id}')
        self.sim.addReceptor(f'Ordinal{id}', 'AMPA')
        self.sim.addReceptor(f'Ordinal{id}', 'GABA')
        self.sim.addReceptor(f'Task{id}', 'AMPA')
        self.sim.addReceptor(f'Task{id}', 'GABA')
        self.sim.addReceptor(f'Shifter{id}', 'AMPA')
        self.sim.addReceptor(f'Shifter{id}', 'GABA')
        self.sim.addCoonection(f'Ordinal{id}', f'Task{id}', 'AMPA')
        self.sim.addCoonection(f'Ordinal{id}', f'Shifter{id}', 'AMPA')

    def assembleNodes(self):
        prev_id = -1
        for id in range(self.length):
            if prev_id == -1:
                prev_id = id
                continue
            self.sim.addCoonection(f'Shifter{prev_id}', f'Ordinal{id}', 'AMPA')
            prev_id = id
        if self.enableFork:
            pass
        if self.enableCoS:
            pass

    def generateTransitionStimuli(self, stimulus_type, *args):
        for i in self.transitions:
            start = self.stimulus['start'] + i*self.stimulus['interval']
            end = start + self.stimulus['duration']
            self.sim.addStimulus(stimulus_type, (start, end), 'Next', *args)

    def setTransitionPeriod(self, start, duration, interval):
        if start > self.stimulus['total_time']:
            print('[Warning] Start time is after the experiment ends. Nothing happens.')
        self.stimulus['start'] = start
        self.stimulus['interval'] = interval
        self.stimulus['duration'] = duration

    def startSimulation(self):
        self.sim.start()

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

    def calculateRobustness(self):
        bumps = []
        trans = []
        for trial in range(self.num_trial):
            task_spike_ratios = [[] for i in range(self.length)]
            spike_counts = [0 for i in range(self.length)]
            time_boundary = self.time_window
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
                        time_boundary += self.time_window
                        total_spike_count = sum(spike_counts) if sum(spike_counts) != 0 else 1
                        spike_ratios = [count/total_spike_count for count in spike_counts]
                        for num, ratio in enumerate(spike_ratios):
                            task_spike_ratios[num].append(ratio)
                            spike_counts = [0 for i in range(5)]
        
                while time_boundary < self.stimulus['total_time']:
                    time_boundary += self.time_window
                    for num in range(self.length):
                        task_spike_ratios[num].append(0)
                
                for data in task_spike_ratios:
                    bumps.append(getBumps(data, 0.5))
                prev_bump = None
                for bump in bumps:
                    if prev_bump == None:
                        prev_bump = bump
                    else:
                        trans.append(getTransition(prev_bump, bump))
                        prev_bump = bump
                
                success = getNumberConsecutiveSuccessTransitions(t0, transition_ground_truth)
                for i in range(1, success+1):
                    x[i] += 1

    def plotStimulusRobustness(self):
        pass

                
if __name__ == '__main__':
    ssm = SNNStateMachine(5)
    ssm.setTransitionPeriod(500, 500)