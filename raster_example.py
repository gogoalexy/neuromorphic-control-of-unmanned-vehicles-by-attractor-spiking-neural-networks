import argparse
from ssc import SNNSequenceControl

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('configuration', choices=['linear', 'branched', 'cos'])
    args = parser.parse_args()

    if args.configuration == 'linear':
        ssc = SNNSequenceControl(5, transitions=4, experiment_time=3000, repetition=1)
        ssc.setTransitionPeriod(1000, 50, 500)
        ssc.generateTransitionStimuli('spike', 'AMPA', 250)
    elif args.configuration == 'branched':
        ssc = SNNSequenceControl(5, transitions=7, fork_pos_len_w=[(2, 3)], experiment_time=5000, repetition=1)
        ssc.setTransitionPeriod(1000, 50, 500)
        ssc.setSwitchEvents([1700], [50])
        ssc.generateExceptionSwitches('spike', 'AMPA', 300)
        ssc.generateTransitionStimuli('spike', 'AMPA', 250)
    elif args.configuration == 'cos':
        ssc = SNNSequenceControl(5, transitions=4, task_weights=[0.5, 0.8, 1.0, 1.3, 1.5], experiment_time=3000, repetition=1)
        ssc.setTransitionPeriod(1000, [60, 70, 90, 120], 500)
        ssc.generateTransitionStimuli('spike', 'AMPA', 250)

    print(ssc.sequence.nodes.data())
    ssc.plotNetwork(mode='ssc')
    ssc.spawnAttractor(100, 50, 'spike', 'AMPA', 400)
    #ssc.startSimulation()
    #ssc.plotRaster()
