import argparse

import matplotlib.pyplot as plt

from ssc import SNNSequenceControl


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--conf', type=str, help='Path to conf file')
    parser.add_argument('-p', '--pro', type=str, help='Path to pro file')
    parser.add_argument('-n', '--num-trial', type=int, help='Number of trials')
    parser.add_argument('-s', '--sim-version', type=int, help='Simulator version')
    args = parser.parse_args()

    num_trial = 100

    ssm = SNNSequenceControl(20, transitions=19, experiment_time=10000, repetition=100)
    ssm.setTransitionPeriod(500, 50, 500)
    ssm.spawnAttractor(100, 50, 'spike', 'AMPA', 400)
    ssm.generateTransitionStimuli('spike', 'AMPA', 250)
    ssm.startSimulation()
    x = ssm.calculateRobustness()

    y = [xi*100/num_trial for xi in x]
    print(y)

    fig, ax = plt.subplots()
    plt.plot([i for i in range(1, 20)], y[1:], 'co-')
    #for i, j in enumerate(y[1:]):
    #    ax.text(i-0.2, j + 0.01, f'{round(j, 2)}%', color='black', fontweight='bold')
    plt.ylabel('Transitions successful (%)')
    plt.show()
    

    
