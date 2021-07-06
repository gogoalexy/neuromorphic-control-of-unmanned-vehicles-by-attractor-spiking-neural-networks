import sys
import numpy as np
import matplotlib.pyplot as plt

from flysim_format import FlysimSNN

class SNNStateMachine:

    def __init__(self, length, fork=False, cos=False, transitions=0, experiment_time=1000, repetition=100):
        self.length = length
        self.transitions = transitions
        self.enableFork = fork
        self.enableCoS = cos
        self.log_filename_base = 'ssm'
        self.repete = repetition
        self.sim = FlysimSNN(experiment_time, repetition, self.log_filename_base)
        self.stimulus = {'total_time': experiment_time}
        self.time_window = 0.05
        self.population_size = 10
        self.spawnNodes()
        self.assembleNodes()
        self.setOutputs()

    def spawnNodes(self):
        self.sim.addNeuron('OrdinalInh', n=self.population_size, taum=5)
        self.sim.addNeuron('TaskInh', n=self.population_size)
        self.sim.addNeuron('Next', n=self.population_size, c=0.1, taum=3, restpot=-55)
        self.sim.addReceptor('OrdinalInh', 'AMPA', tau=20, meanexteff=10.5)
        self.sim.addReceptor('TaskInh', 'AMPA', tau=20, meanexteff=10.5)
        self.sim.addReceptor('Next', 'AMPA', tau=1, meanexteff=10)
        for id in range(self.length):
            self.declareNodeNeurons(id)
            self.sim.addCoonection(f'Ordinal{id}', 'OrdinalInh', 'AMPA', 0.5)
            self.sim.addCoonection(f'Task{id}', 'TaskInh', 'AMPA', 1.0)
            self.sim.addCoonection('Next', f'Shifter{id}', 'AMPA', 2.0)
            self.sim.addCoonection('OrdinalInh', f'Ordinal{id}', 'GABA', 10)
            self.sim.addCoonection('TaskInh', f'Task{id}', 'GABA', 50.0)
        self.sim.defineGroup('Ordinal', [f'Ordinal{id}' for id in range(self.length)])
        self.sim.defineGroup('Task', [f'Task{id}' for id in range(self.length)])
        self.sim.defineGroup('Shifter', [f'Shifter{id}' for id in range(self.length)])

    def declareNodeNeurons(self, id):
        self.sim.addNeuron(f'Ordinal{id}', n=self.population_size)
        self.sim.addNeuron(f'Task{id}', n=self.population_size)
        self.sim.addNeuron(f'Shifter{id}', n=self.population_size, c=0.1, taum=1.0)
        self.sim.addReceptor(f'Ordinal{id}', 'AMPA', meanexteff=10.5)
        self.sim.addReceptor(f'Ordinal{id}', 'GABA', tau=5, revpot=-90, meanextconn=0.0)
        self.sim.addReceptor(f'Task{id}', 'AMPA', tau=20, meanexteff=10.5)
        self.sim.addReceptor(f'Task{id}', 'GABA', tau=5, revpot=-90)
        self.sim.addReceptor(f'Shifter{id}', 'AMPA', tau=20, meanexteff=10.5)
        self.sim.addReceptor(f'Shifter{id}', 'GABA', tau=5, revpot=-90)
        self.sim.addCoonection(f'Ordinal{id}', f'Task{id}', 'AMPA', 4)
        self.sim.addCoonection(f'Ordinal{id}', f'Shifter{id}', 'AMPA', 0.4)
        self.sim.addCoonection(f'Ordinal{id}', f'Ordinal{id}', 'AMPA', 2.5)

    def assembleNodes(self):
        prev_id = -1
        for id in range(self.length):
            if prev_id == -1:
                prev_id = id
                continue
            self.sim.addCoonection(f'Shifter{prev_id}', f'Ordinal{id}', 'AMPA', 1.0)
            prev_id = id
        if self.enableFork:
            pass
        if self.enableCoS:
            pass

    def spawnAttractor(self, start, duration, stimulus_type, *args):
        self.sim.addStimulus(stimulus_type, (start, start + duration), 'Ordinal0', *args)

    def generateTransitionStimuli(self, stimulus_type, *args):
        for i in range(self.transitions):
            start = self.stimulus['start'] + i*self.stimulus['interval']
            end = start + self.stimulus['duration']
            self.sim.addStimulus(stimulus_type, (start, end), 'Next', *args)

    def setTransitionPeriod(self, start, duration, interval):
        if start > self.stimulus['total_time']:
            print('[Warning] Start time is after the experiment ends. Nothing happens.')
        self.stimulus['start'] = start
        self.stimulus['interval'] = interval
        self.stimulus['duration'] = duration

    def setOutputs(self):
        self.sim.defineOutput('Spike', f'{self.log_filename_base}_task.dat', 'AllPopulation')

    def startSimulation(self):
        self.sim.start()

    def getBumps(self, spike_ratios, threshold):
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

    def getTransition(self, bump_series0, bump_series1):
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

    def getGroundTruthTransition(self):
        gt = []
        for i in range(self.transitions):
            start = self.stimulus['start'] + i*self.stimulus['interval']
            gt.append((start/(self.time_window*1000), '+'))
        return gt

    def getNumberConsecutiveSuccessTransitions(self, transitions, ground_truth):
        counter = 0
        for real, expect in zip(transitions, ground_truth):
            if (real[0] - expect[0]) < 3 and (real[1] == expect[1]):
                counter += 1
        return counter

    def calculateRobustness(self):
        bumps = []
        trans = []
        x = [0 for i in range(self.length)]
        base_id = self.sim.getNeuron('Ordinal0')['id'] * self.population_size
        for trial in range(self.repete):
            task_spike_ratios = [[] for i in range(self.length)]
            spike_counts = [0 for i in range(self.length)]
            time_boundary = self.time_window
            if trial == 0:
                filename = f'{self.log_filename_base}_task.dat'
            else:
                filename = f'{self.log_filename_base}_task.dat_{trial+1}'
            with open(filename, 'r') as spike_file:
                for event in spike_file:
                    timestamp, neuron = event.split(' ')
                    timestamp, neuron = float(timestamp), int(neuron)
                    if time_boundary > timestamp:
                        spike_counts[(neuron-base_id)//(3*self.population_size)] += 1
                    else:
                        time_boundary += self.time_window
                        total_spike_count = sum(spike_counts) if sum(spike_counts) != 0 else 1
                        spike_ratios = [count/total_spike_count for count in spike_counts]
                        for num, ratio in enumerate(spike_ratios):
                            task_spike_ratios[num].append(ratio)
                            spike_counts = [0 for i in range(self.length)]
        
                while time_boundary < self.stimulus['total_time']:
                    time_boundary += self.time_window
                    for num in range(self.length):
                        task_spike_ratios[num].append(0)
                
                for data in task_spike_ratios:
                    bumps.append(self.getBumps(data, 0.5))
                prev_bump = None
                for bump in bumps:
                    if prev_bump == None:
                        prev_bump = bump
                    else:
                        trans.extend(self.getTransition(prev_bump, bump))
                        prev_bump = bump
                print(bumps)
                success = self.getNumberConsecutiveSuccessTransitions(trans, self.getGroundTruthTransition())
                for i in range(1, success+1):
                    x[i] += 1

    def plotStimulusRobustness(self):
        pass
        
    def plotRaster(self, save=False, show=True, name_modifier=''):
        task_spikes = [[] for i in range(180)]
        with open(f'{self.log_filename_base}_task.dat', 'r') as spike_file:
            for event in spike_file:
                t, neuron = event.split(' ')
                task_spikes[int(neuron)].append(float(t))
                    
        fig, ax = plt.subplots()
        colors1 = [f'C{i//10}' for i in range(180)]
        ax.set_xlim(-0.1, 3.1)
        ax.eventplot(task_spikes, linelengths = 0.8, linewidths = 1.0, colors = colors1)
        if save:
            plt.savefig(f'{self.log_filename_base}_{name_modifier}.png')
        if show:
            plt.show()
                
if __name__ == '__main__':
    ssm = SNNStateMachine(5, transitions=4, experiment_time=3000, repetition=3)
    ssm.setTransitionPeriod(500, 50, 500)
    ssm.spawnAttractor(100, 50, 'spike', 'AMPA', 400)
    ssm.generateTransitionStimuli('spike', 'AMPA', 260)
    ssm.startSimulation()
    #ssm.calculateRobustness()
    ssm.plotRaster()
