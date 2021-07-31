import random
import sys
import numpy as np
import matplotlib.pyplot as plt

from flysim_format import FlysimSNN

class SNNStateMachine:

    def __init__(self, trunk_length, fork_pos_len_w=None, task_weights=None, transitions=0, experiment_time=1000, repetition=100):
        self.length = trunk_length
        self.transitions = transitions
        self.fork_pos_len_w = fork_pos_len_w
        self.log_filename_base = 'ssc'
        self.repete = repetition
        self.sim = FlysimSNN(experiment_time, repetition, self.log_filename_base)
        self.stimulus = {'total_time': experiment_time}
        self.time_window = 0.05
        self.population_size = 10
        self.task_weights = task_weights
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
        if self.fork_pos_len_w:
            for index, pair in enumerate(self.fork_pos_len_w, start=1):
                self.concatenateBranch(pair, index)
        if self.task_weights:
            self.sim.addNeuron('TaskTarget', n=self.population_size, c=0.5, taum=10, restpot=-55)
            self.sim.addReceptor('TaskTarget', 'AMPA', tau=20, meanexteff=10.5)
            self.sim.addReceptor('TaskTarget', 'GABA', tau=5, revpot=-90, meanexteff=0)
            self.sim.addNeuron('CurrentStatus', n=self.population_size, c=0.5, taum=10, restpot=-55)
            self.sim.addReceptor('CurrentStatus', 'AMPA', tau=20, meanexteff=10.5)
            self.sim.addReceptor('CurrentStatus', 'GABA', tau=5, revpot=-90, meanexteff=0)
            self.sim.addCoonection('TaskTarget', 'TaskTarget', 'AMPA', 0.05)
            self.sim.addCoonection('CurrentStatus', 'CurrentStatus', 'AMPA', 0.05)
            self.sim.addCoonection('TaskTarget', 'CurrentStatus', 'GABA', 5.0)
            self.sim.addCoonection('CurrentStatus', 'TaskTarget', 'GABA', 5.0)
            self.sim.addCoonection('CurrentStatus', 'Next', 'AMPA', 1.0)
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

    def concatenateBranch(self, pos_len_w, id):
        if pos_len_w[0] < self.length:
            if pos_len_w[1] > 9:
                raise Exception('Individual branch length should not exceed 9 nodes.')
            self.sim.addNeuron(f'OrdinalInh{index}', n=self.population_size, taum=5)
            self.sim.addNeuron(f'TaskInh{index}', n=self.population_size)
            self.sim.addNeuron(f'Switch{index}', n=self.population_size, c=0.1, taum=3, restpot=-55)
            self.sim.addReceptor('OrdinalInh{index}', 'AMPA', tau=20, meanexteff=10.5)
            self.sim.addReceptor('TaskInh{index}', 'AMPA', tau=20, meanexteff=10.5)
            self.sim.addReceptor('Switch{index}', 'AMPA', tau=1, meanexteff=10)
            for subid in range(pair[1]):
                if len(pos_len_w) == 3:
                    self.declareNodeNeurons(pos_len_w[0]+subid/10, pos_len_w[2])
                else:
                    self.declareNodeNeurons(pos_len_w[0]+subid/10)
                    self.sim.addCoonection(f'Ordinal{id}', 'OrdinalInh', 'AMPA', 0.5)
                    self.sim.addCoonection(f'Task{id}', 'TaskInh', 'AMPA', 1.0)
                    self.sim.addCoonection('Next', f'Shifter{id}', 'AMPA', 2.0)
                    self.sim.addCoonection('OrdinalInh', f'Ordinal{id}', 'GABA', 10)
                    self.sim.addCoonection('TaskInh', f'Task{id}', 'GABA', 50.0)

    def declareNodeNeurons(self, id, fork_task_weights=None):
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
        if self.task_weights:
            self.sim.addCoonection(f'Task{id}', 'TaskTarget', 'AMPA', self.task_weights[id])
        if fork_task_weights:
            self.sim.addCoonection(f'Task{id}', 'TaskTarget', 'AMPA', fork_task_weights[id%10])
            

    def assembleNodes(self):
        prev_id = -1
        for id in range(self.length):
            if prev_id == -1:
                prev_id = id
                continue
            self.sim.addCoonection(f'Shifter{prev_id}', f'Ordinal{id}', 'AMPA', 1.0)
            prev_id = id
        if self.fork_pos_len_w:
            pass

    def spawnAttractor(self, start, duration, stimulus_type, *args):
        self.sim.addStimulus(stimulus_type, (start, start + duration), 'Ordinal0', *args)

    def generateTransitionStimuli(self, stimulus_type, *args):
        for i in range(self.transitions):
            start = self.stimulus['start'] + i*self.stimulus['interval']
            if isinstance(self.stimulus['duration'], list):
                end = start + self.stimulus['duration'][i]
            else:
                end = start + self.stimulus['duration']
            if self.task_weights:
                self.sim.addStimulus(stimulus_type, (start, end), 'CurrentStatus', *args)
            else:
                self.sim.addStimulus(stimulus_type, (start, end), 'Next', *args)

    def setTransitionPeriod(self, start, duration, interval):
        if start > self.stimulus['total_time']:
            print('[Warning] Start time is after the experiment ends. Nothing happens.')
        self.stimulus['start'] = start
        self.stimulus['interval'] = interval
        self.stimulus['duration'] = duration

    def setOutputs(self):
        self.sim.defineOutput('Spike', f'{self.log_filename_base}_all.dat', 'AllPopulation')
        self.sim.defineOutput('Spike', f'{self.log_filename_base}_task.dat', 'Task')

    def startSimulation(self, thread=1):
        self.sim.start(thread)

    def getBumps(self, spike_ratios, threshold=0.5):
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
        x = [0 for i in range(self.length)]
        base_id = self.sim.getNeuron('Ordinal0')['id'] * self.population_size
        for trial in range(self.repete):
            bumps = []
            trans = []
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
                success = self.getNumberConsecutiveSuccessTransitions(trans, self.getGroundTruthTransition())
                for i in range(1, success+1):
                    x[i] += 1
        return x

    def plotStimulusRobustness(self):
        pass
        
    def plotRaster(self, save=False, show=True, name_modifier=''):
        num_neuron = 30 + 30*self.length
        hl = [20, 30]
        neuron_labels = ['Inh', 'Next']
        if self.task_weights:
            num_neuron += 20
            colors1 = [f'C{i//10+7}' for i in range(30)]
            colorcos =[f'C{i//10+3}' for i in range(20)]
            colors1.extend(colorcos)
            hl.append(50)
            neuron_labels.append('CoS')
            ytick = [x for x in range(65, num_neuron, 30)]
            ytick.insert(0, 40)
        else:
            colors1 = [f'C{i//10+7}' for i in range(30)]
            ytick = [x for x in range(45, num_neuron, 30)]
        vl = [x for x in np.arange(self.stimulus['start']/1000, self.stimulus['start']/1000+self.stimulus['interval']/1000*self.transitions, self.stimulus['interval']/1000)]
        neuron_labels.extend([f'Node{i}' for i in range(1, self.length+1)])
        hl.extend([i for i in range(hl[-1], num_neuron, 30)])
        task_spikes = [[] for i in range(num_neuron)]
        with open(f'{self.log_filename_base}_all.dat', 'r') as spike_file:
            for event in spike_file:
                t, neuron = event.split(' ')
                task_spikes[int(neuron)].append(float(t))
                    
        fig, ax = plt.subplots()
        colors2 = [f'C{i//10}' for i in range(30)]
        colors1.extend(colors2*self.length)
        ax.set_xlim(0.0, self.stimulus['total_time']/1000)
        ax.set_ylim(0, num_neuron)     
        plt.hlines(hl, -0.1, self.stimulus['total_time']+0.1)
        plt.vlines(vl, 0, num_neuron, colors='r', linestyles='dashed')
        ytick.insert(0, 25)
        ytick.insert(0, 10)
        ax.set_yticklabels(neuron_labels)
        ax.set_xlabel('Time (s)')
        ax.set_yticks(ytick)
        ax.eventplot(task_spikes, linelengths = 0.8, linewidths = 1.0, colors = colors1)
        if save:
            plt.savefig(f'{self.log_filename_base}_{name_modifier}.png')
        if show:
            plt.show()
                
if __name__ == '__main__':
    ssm = SNNStateMachine(5, transitions=4, task_weights=[1, 1, 2, 1, 3], experiment_time=3000, repetition=1)
    ssm.setTransitionPeriod(500, 50, 500)
    ssm.spawnAttractor(100, 50, 'spike', 'AMPA', 400)
    ssm.generateTransitionStimuli('spike', 'AMPA', 250)
    ssm.startSimulation()
    #ssm.calculateRobustness()
    ssm.plotRaster()
