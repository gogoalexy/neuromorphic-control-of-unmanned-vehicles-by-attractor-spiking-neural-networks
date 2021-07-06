from ssm import SNNStateMachine


if __name__ == '__main__':
    stimulus_duration_list = [d for d in range(0, 300, 50)]
    stimulus_strength_list = [s for s in range(0, 600, 50)]
    for d in stimulus_duration_list:
        for s in stimulus_strength_list:
            ssm = SNNStateMachine(5, transitions=4, experiment_time=3000, repetition=2)
            ssm.setTransitionPeriod(500, d, 500)
            ssm.spawnAttractor(100, 50, 'spike', 'AMPA', 400)
            ssm.generateTransitionStimuli('spike', 'AMPA', s)
            ssm.startSimulation()
            #ssm.calculateRobustness()
            ssm.plotRaster(show=False, save=True, name_modifier=str(d)+'_'+str(s))
