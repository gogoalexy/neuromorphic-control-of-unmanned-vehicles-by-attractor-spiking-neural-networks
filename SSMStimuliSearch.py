import sys
import numpy as np

class SNNStateMachine:

    def __init__(self, length, fork=False, cos=False, experiment_time=3000):
        self.length = length
        self.enableFork = fork
        self.enableCoS = cos
        self.protocol = StimulationProtocol(experiment_time)
        self.stimulus = {'total_time': experiment_time}
        self.num_trial = 500
        self.time_window = 0.05

    def spawnNodes(self):
        self.nodes = [SSMUnit(i) for i in range(self.length)]

    def assembleNodes(self):
        prevNode = None
        for node in self.nodes:
            if prevNode is None:
                prevNode = node
                continue
            prevNode.linkTo(node)
            prevNode = node
        if self.enableFork:
            pass
        if self.enableCoS:
            pass

    def setTransitionStimulus(self, stimulus_type, duration, arg0, arg1):
        self.stimulus['type'] = stimulus_type
        self.stimulus['duration'] = duration
        self.stimulus['arg0'] = arg0
        self.stimulus['arg1'] = arg1

    def setTransitionPeriod(self, start, interval):
        if start > self.stimulus['total_time']:
            print('[Warning] Start time is after the experiment ends. Nothing happens.')
        self.stimulus['start'] = start
        self.stimulus['interval'] = interval
        self.stimulus['end'] = start + interval * self.length

    def generatePro(self):
        self.protocol.periodicStimuli(self.stimulus['start'], self.stimulus['end'], self.stimulus['interval'], self.stimulus['duration'], self.stimulus['type'], 'Next', self.stimulus['arg0'], self.stimulus['arg1'])
        with open('network_ssm.pro', 'w') as pro:
            pro.write(self.protocol.getPro())

    def generateConf(self):
        with open('network_ssm.conf', 'w') as conf:
            for node in self.nodes:
                conf.write(node.getUnitConf())
            conf.write(node.getCrossUnitConf())

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



class SSMUnit:
    
    ordinalInh = NeuralPopulation('OrdinalInh')
    taskInh = NeuralPopulation('TaskInh')
    next = NeuralPopulation('Next')
    crossUnitWrote = False

    def __init__(self, n):
        self.ordinal = NeuralPopulation(f'Ordinal{n}')
        self.shifter = NeuralPopulation(f'Shifter{n}')
        self.task = NeuralPopulation(f'Task{n}')

    def formStereotypicalUnitLinks(self):
        self.ordinal.innervate(self.task)
        self.ordinal.innervate(self.shifter)
        self.ordinal.innervate(ordinalInh)
        self.task.innervate(taskInh)
        ordinalInh.innervate(self.ordinal)
        taskInh.innervate(self.task)
        next.innervate(self.ordinal)

    def linkTo(self, target):
        self.shifter.innervate(target.ordinal, 'AMPA')

    def getUnitConf(self):
        conf = self.ordinal.getConf() + '\n' + \
               self.task.getConf() + '\n' + \
               self.shifter.getConf()
        return conf

    def getCrossUnitConf(self):
        if SSMUnit.crossUnitWrote:
            print('[Warning] Cross unit neurons should be only written once.')
        else:
            SSMUnit.crossUnitWrote = True
        conf = SSMUnit.ordinalInh.getConf() + \
               SSMUnit.taskInh.getConf() + \
               SSMUnit.next.getConf()
        return conf
                
if __name__ == '__main__':
    ssm = SNNStateMachine(5)
    ssm.spawnNodes()
    ssm.assembleNodes()
    ssm.generateConf()
    ssm.setTransitionStimulus('spike', 50, 'AMPA', 250)
    ssm.setTransitionPeriod(500, 500)
    ssm.generatePro()