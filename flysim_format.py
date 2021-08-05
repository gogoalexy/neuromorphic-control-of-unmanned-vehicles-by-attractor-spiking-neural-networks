import subprocess

import networkx as nx
import numpy as np

class FlysimSNN:

    def __init__(self, trial_time, iterations, dat_base_name='network'):
        self.flysim_target = 'simulators/sim08_15/flysim.out'
        self.network = nx.DiGraph()
        self.subgraph = {}
        self.iter = iterations
        self.protocol = StimulationProtocol(trial_time)
        self.id_counter = 0
        self.conf_name = f'{dat_base_name}.conf'
        self.pro_name = f'{dat_base_name}.pro'

    def addNeuron(self, name, n=1, c=0.5, leakyC=2.5, taum=20, threshold=-50, restpot=-70, resetpot=-55, refracperiod=20, spikedly=0, selfconnect=False, layer=None):
        conf = NeuralPopulation(name, n, c, leakyC, taum, threshold, restpot, resetpot, refracperiod, spikedly, selfconnect)
        self.network.add_node(name, id=self.id_counter, spike_count=0, config=conf)
        self.id_counter += 1
        if layer:
            self.network.nodes[name]['layer'] = layer

    def addReceptor(self, neuron_name, receptpr_type, tau=10, revpot=0, freqext=0.0, meanexteff=0.0, meanextconn=1.0):
        if self.isNeuronExist(neuron_name):
            neuron = self.getNeuron(neuron_name)
            neuron['config'].haveReceptor(receptpr_type, tau, revpot, freqext, meanexteff, meanextconn)

    def addCoonection(self, source_name, target_name, receptor, mean_effect=0.0, weight=1.0, connectivity=1.0):
        if self.isNeuronExist(source_name) and self.isNeuronExist(target_name):
            self.network.add_edge(source_name, target_name)
            neuron = self.getNeuron(source_name)
            neuron['config'].innervate(target_name, receptor, mean_effect, weight, connectivity)

    def addStimulus(self, stimulus_type, time_point, to, *args):
        if self.isNeuronExist(to) or self.protocol.isGroupExist(to):
            if stimulus_type == 'current':
                self.protocol.injectCurrent(time_point[0], to, *args)
                if time_point[1] != None:
                    self.protocol.injectCurrent(time_point[1], to, 0, 0)
            elif stimulus_type == 'spike':
                self.protocol.injectSpikes(time_point[0], to, *args)
                if time_point[1] != None:
                    self.protocol.injectSpikes(time_point[1], to, args[0], 0)
        else:
            return

    def defineGroup(self, group_name, group_members):
        for member in group_members:
            if not self.isNeuronExist(member):
                return
        self.protocol.addGroup(group_name, group_members)

    def defineOutput(self, format, file_name, target, **kwargs):
        if self.isNeuronExist(target) or self.protocol.isGroupExist(target) or target=='AllPopulation':
            if format == 'FiringRate':
                self.protocol.outputFiringRates(file_name, target, **kwargs)
            elif format == 'Spike':
                self.protocol.outputSpikes(file_name, target)
            elif format == 'MemPot':
                self.protocol.outputMembranePotential(file_name, target)
        else:
            return

    def getNeuron(self, neuron_name):
        try:
            return self.network.nodes[neuron_name]
        except:
            return None

    def getAllConf(self):
        conf = ''
        for neuron_name, neural_population in self.network.nodes(data='config'):
            conf += neural_population.getConf()
        return conf

    def generatePro(self):
        with open(self.pro_name, 'w') as pro:
            pro.write(self.protocol.getPro())

    def generateConf(self):
        with open(self.conf_name, 'w') as conf:
            conf.write(self.getAllConf())

    def isNeuronExist(self, neuron_name):
        if self.getNeuron(neuron_name) == None:
            return False
        else:
            return True

    def start(self, thread=1):
        self.generateConf()
        self.generatePro()
        subprocess.call([self.flysim_target, '-conf', self.conf_name, '-pro', self.pro_name, '-rp', str(self.iter), '-t', str(thread)])
    

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
            return type_name in self.types

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
        def __init__(self, name, receptor, mean_effect, weight, connectivity):
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

    def haveReceptor(self, receptor_type, tau, revpot, freqext, meanexteff, meanextconn):
        receptor = self.Receptor(receptor_type, tau, revpot, freqext, meanexteff, meanextconn)
        self.receptorConf += receptor.getConf()

    def innervate(self, target, receptor, mean_effect, weight, connectivity):
        connection = self.TargetPopulation(target, receptor, mean_effect, weight, connectivity)
        self.connectionConf += connection.getConf()

    def getConf(self):
        out = f'NeuralPopulation: {self.Name}\n' + \
              f'N={self.N}\n' + \
              f'C={self.Capacitance}\n' + \
              f'Taum={self.Taum}\n' + \
              f'RestPot={self.RestingPotential}\n' + \
              f'ResetPot={self.ResetPotential}\n' + \
              f'Threshold={self.Threshold}\n\n' + \
              self.receptorConf + '\n' + \
              self.connectionConf + '\n' + \
              'EndNeuralPopulation\n\n'
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

    def isGroupExist(self, group_name):
        try:
            g = self.groups[group_name]
            return True
        except:
            return False

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

if __name__ == '__main__':
    sim = FlysimSNN(1000, 100, 'ssm')
    sim.addNeuron('Task')
    sim.addReceptor('Task', 'AMPA')
    sim.addStimulus('current', (500, 550), 'Task', 10, 100)
    sim.defineOutput('Spike', 'spike.dat', 'Task')
    sim.start()

