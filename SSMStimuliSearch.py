import sys
import numpy as np

class NeuralPopulation:
    '''Only support LIF model ,and STP and LTP are not considered here.'''
    def __init__(self, name, n=1, c=0.5, leakyC=10, taum=20, threshold=-50, restpot=-70, resetpot=-55, refracperiod=20, spikedly=0, selfconnect=False):
        self.Name = name
        self.N = n
        self.Capacitance = c
        self.LeakyConductance = leakyC
        self.Taum = taum
        self.Threshold = threshold
        self.RestingPotential = restpot
        self.ResetPotential = resetpot
        self.RefractoryPeriod = refracperiod
        self.SpikeDelay = spikedly
        self.SelfConnection = selfconnect
        self.receptorConf = ''
        self.connectionConf = ''

    class Receptor:

        types = ('AMPA', 'GABA', 'NMDA', 'Ach', 'GluCl', 'Gap', 'Sine')
       
        def __init__(self, receptpr_type, tau, revpot, freqext, meanexteff, meanextconn):
            if self.isReceptorType(receptpr_type):
                self._legal = True
            else:
                print('[Error] Unknown receptor, skip.')
                self._legal = False
            self.type = receptpr_type
            self.Tau = tau
            self.ReversePotential = revpot
            self.FreqExt = freqext
            self.MeanExtEff = meanexteff
            self.MeanExtCon = meanextconn
                
        def isReceptorType(self, type_name):
            return type_name in Receptor.types

        def getConf(self):
            if self._legal:
                out = f'Receptor: {self.type}\n' + \
                      f'Tau={self.Tau}\n' + \
                      f'RevPot={self.ReversePotential}\n' + \
                      f'FreqExt={self.FreqExt}\n' + \
                      f'MeanExtEff={self.MeanExtEff}\n' + \
                      f'MeanExtCon={self.MeanExtCon}\n' + \
                      'EndReceptor\n'
            else:
                out = ''
            return out
                
    class TargetPopulation:
        def __init__(self, name, receptor, mean_effect=0.0, weight=1.0, connectivity=1.0):
            self.Target = name
            self.Receptor = receptor
            self.MeanEff = mean_effect
            self.Weight = weight
            self.Connectivity = connectivity
                
        def getConf(self):
            out = f'TargetPopulation: {self.Target}\n' + \
                  f'TargetReceptor={self.Receptor}\n' + \
                  f'MeanEff={self.MeanEff}\n' + \
                  f'weight={self.Weight}\n' + \
                  'EndTargetPopulation\n'
            return out

    def haveReceptor(self, type):
        receptor = self.Receptor(type)
        self.receptorConf += (receptor.getConf() + '\n')

    def innervate(self, target, receptor):
        connection = self.TargetPopulation(target.Name, receptor)
        self.connectionConf += (connection.getConf() + '\n')

    def getConf(self):
        out = f'NeuralPopulation: {self.Name}\n' + \
              f'N={self.N}\n' + \
              f'C={self.Capacitance}\n' + \
              f'Taum={self.Taum}\n' + \
              f'RestPot={self.RestingPotential}\n' + \
              f'ResetPot={self.ResetPotential}\n' + \
              f'Threshold={self.Threshold}\n' + \
              self.receptorConf + '\n' + \
              self.connectionConf + '\n' + \
              'EndNeuralPopulation\n'
        return out
             
class StimulationProtocol:

    def __init__(self, experiment_time):
        self.groups = {}
        self.events = []
        self.outfiles = []
        self.endTime = experiment_time

    def addGroup(self, group_name, member_list):
        self.groups[group_name] = member_list

    def outputFiringRates(self, file_name, target, window_size, step_size):
        self.outfiles.append({'type': 'FiringRate', 'name': file_name, 'population': target, 'window': window_size, 'step': step_size})

    def outputSpikes(self, file_name, target):
        self.outfiles.append({'type': 'Spike', 'name': file_name, 'population': target})

    def outputMembranePotential(self, file_name, target):
        self.outfiles.append({'type': 'MemPot','name': file_name, 'population': target})

    def injectCurrent(self, time_point, to, mean, std):
        self.events.append({'time': time_point, 'type': 'ChangeMembraneNoise', 'to': to, 'mean': mean, 'std': std})

    def injectSpikes(self, time_point, to, receptor, hz):
        self.events.append({'time': time_point, 'type': 'ChangeExtFreq', 'to': to, 'receptor': receptor, 'hz': hz})

    def periodicStimuli(self, start, end, interval, duration, stimuli_type, to, *args):
        if end > self.endTime:
            print('[Warning] The end time of stimuli exceeds the experiment time, truncate the excess stimuli.')
            end = self.endTime
        for time_point in np.arange(start, end, interval):
            if stimuli_type == 'current':
                self.injectCurrent(time_point, to, args[0], args[1])
                self.injectCurrent(time_point+duration, to, 0.0, 0.0)
            elif stimuli_type == 'spike':
                self.injectSpikes(time_point, to, args[0], args[1])
                self.injectSpikes(time_point+duration, to, args[0], 0.0)

    def getPro(self):
        out = ''
        if self.groups:
            out += 'DefineMacro\n\n'
            for group_name, group_member in self.groups.items():
                out += f'GroupName:{group_name}\n' + \
                       f"GroupMembers:{','.join(group_member)}\n" + \
                       'EndGroupMembers\n\n'
            out += 'EndDefineMacro\n\n'
        for event in self.events:
            out += f"EventTime {event['time']}\n" + \
                   f"Type={event['type']}\n" + \
                   "Label=#1#\n" + \
                   f"Population:{event['to']}\n"
            if event['type'] == 'ChangeExtFreq':
                out += f"Receptor:{event['receptor']}\n" + \
                       f"FreqExt={event['hz']}\n"
            elif event['type'] == 'ChangeMembraneNoise':
                out += f"GaussMean:{event['mean']}\n" + \
                       f"GaussSTD:{event['std']}\n"
            out += 'EndEvent\n\n'
        out += f"EventTime {self.endTime}\n" + \
               'Type=EndTrial\n' + \
               'Label=End_of_the_trial\n' + \
               'EndEvent\n\n'
        if self.outfiles:
            out += 'OutControl\n'
            for outfile in self.outfiles:
                out += f"FileName:{outfile['name']}\n" + \
                       f"Type={outfile['type']}\n"
                if outfile['type'] == 'FiringRate':
                    out += f"FiringRateWinodw={outfile['window']}\n" + \
                           f"PrintStep={outfile['step']}\n"
                out += f"population:{outfile['population']}\n" + \
                       'EndOutputFile\n\n'
            out += 'EndOutControl\n'

        return out


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