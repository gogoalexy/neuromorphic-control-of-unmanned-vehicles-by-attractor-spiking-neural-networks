import argparse
import matplotlib.pyplot as plt
import numpy as np

from flysim_format import FlysimSNN
from ssc import SNNSequenceControl


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('target', choices=['next', 'task', 'decision', 'cos'])
    args = parser.parse_args()
    
    if args.next:
        results = []
        stimulus_duration_list = [d for d in range(0, 1050, 50)]
        stimulus_strength_list = [s for s in range(0, 1050, 50)]
        for d in stimulus_duration_list:
            row = []
            for s in stimulus_strength_list:
                ssm = SNNSequenceControl(5, transitions=4, experiment_time=5000, repetition=10)
                ssm.setTransitionPeriod(1000, d, 1000)
                ssm.spawnAttractor(100, 50, 'spike', 'AMPA', 400)
                ssm.generateTransitionStimuli('spike', 'AMPA', s)
                ssm.startSimulation()
                x = ssm.calculateRobustness()
                row.append(x[4])
                print(x)
                #ssm.plotRaster(show=False, save=True, name_modifier=str(d)+'_'+str(s))
            results.insert(0, row)
    
        res = np.array(results)
        res = res/50
        fig, ax = plt.subplots()
        im = ax.imshow(res)
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel("Robustness", rotation=-90, va="bottom")
        ax.set_xticks(np.arange(len(stimulus_strength_list)))
        ax.set_yticks(np.arange(len(stimulus_duration_list)))
        ax.set_xticklabels(stimulus_strength_list)
        ax.set_yticklabels(list(reversed(stimulus_duration_list)))
        ax.set_xlabel('Stimulus Frequency')
        ax.set_ylabel('Stimulus Duration')
        plt.show()
        
    if args.task:
        stimulus_duration_list = [d for d in range(0, 550, 50)]
        stimulus_strength_list = [s for s in range(0, 550, 50)]
        task_weight_list = [0.01, 0.1, 1.0, 3.0, 5.0, 10.0]
        for w in task_weight_list:
            results = []
            for d in stimulus_duration_list:
                row = []
                for s in stimulus_strength_list:
                    ssm = SNNSequenceControl(2, transitions=1, task_weights=[w, w], experiment_time=1000, repetition=10)
                    ssm.setTransitionPeriod(500, d, 1000)
                    ssm.spawnAttractor(100, 50, 'spike', 'AMPA', 400)
                    ssm.generateTransitionStimuli('spike', 'AMPA', s)
                    ssm.startSimulation()
                    x = ssm.calculateRobustness()
                    row.append(x[1])
                    print(x)
                results.insert(0, row)
                
            res = np.array(results)
            res = res/10
            fig, ax = plt.subplots()
            im = ax.imshow(res)
            plt.title(f'Task weight {w}')
            cbar = ax.figure.colorbar(im, ax=ax)
            cbar.ax.set_ylabel("Robustness", rotation=-90, va="bottom")
            ax.set_xticks(np.arange(len(stimulus_strength_list)))
            ax.set_yticks(np.arange(len(stimulus_duration_list)))
            ax.set_xticklabels(stimulus_strength_list)
            ax.set_yticklabels(list(reversed(stimulus_duration_list)))
            ax.set_xlabel('Stimulus Frequency')
            ax.set_ylabel('Stimulus Duration')
            plt.show()
            
    if args.decision:
        mean_rates = []
        weights = np.linspace(0.0, 2.0, 400)
        for w in weights:
            snn = FlysimSNN(1000, 1, 'decision')
            snn.addNeuron('TaskTarget', n=10, c=0.5, taum=10, restpot=-55)
            snn.addReceptor('TaskTarget', 'AMPA', tau=20, meanexteff=10.5)
            snn.addReceptor('TaskTarget', 'GABA', tau=5, revpot=-90, meanexteff=0)
            snn.addNeuron('CurrentStatus', n=10, c=0.5, taum=10, restpot=-55)
            snn.addReceptor('CurrentStatus', 'AMPA', tau=20, meanexteff=10.5)
            snn.addReceptor('CurrentStatus', 'GABA', tau=5, revpot=-90, meanexteff=0)
            snn.addCoonection(f'TaskTarget', 'TaskTarget', 'AMPA', 0.05)
            snn.addCoonection(f'CurrentStatus', 'CurrentStatus', 'AMPA', 0.05)
            snn.addCoonection(f'TaskTarget', 'CurrentStatus', 'GABA', 5.0)
            snn.addCoonection(f'CurrentStatus', 'TaskTarget', 'GABA', 5.0)
            snn.addCoonection(f'CurrentStatus', 'Next', 'AMPA', 1.0)
            snn.addNeuron(f'Ordinal', n=10)
            snn.addNeuron('Task', n=10)
            snn.addReceptor('Ordinal', 'AMPA', meanexteff=10.5)
            snn.addReceptor('Ordinal', 'GABA', tau=5, revpot=-90, meanextconn=0.0)
            snn.addReceptor('Task', 'AMPA', tau=20, meanexteff=10.5)
            snn.addReceptor('Task', 'GABA', tau=5, revpot=-90)
            snn.addCoonection('Ordinal', 'Task', 'AMPA', 4)
            snn.addCoonection('Ordinal', 'Ordinal', 'AMPA', 1.0)
            snn.addCoonection(f'Task', 'TaskTarget', 'AMPA', w)

            snn.addStimulus('spike', (400, 50), 'Ordinal', 'AMPA', 400)
            snn.addStimulus('spike', (500, None), 'CurrentStatus', 'AMPA', 500)
            snn.defineOutput('FiringRate', 'decision.dat', 'TaskTarget', window_size=50, step_size=50)
            snn.start()
            rates = []
            with open('decision.dat', 'r') as dat:
                for rate in dat:
                    t, r = rate.split(' ')
                    if float(t) > 0.6:
                        rates.append(float(r))
            mean = np.mean(rates)
            mean_rates.append(mean)
        fig, ax = plt.subplots()
        plt.title('TaskTarget Neuron Firing Rates against Connection Weights from Task Neuron')
        ax.plot(weights, mean_rates, color='k')
        ax.set_xlabel('Connection weight')
        ax.set_ylabel('Mean firing rate (Hz)')
        plt.show()
    
    if args.cos:  
        for weight in np.logspace(0.1, 10, 10):
            results = []
            for strength in range(100, 1000, 100):
                row = []
                for duration in range(50, 500, 50):
                    ssc = SNNSequenceControl(2, transitions=1, task_weights=[weight, weight], experiment_time=1000, repetition=1)
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
