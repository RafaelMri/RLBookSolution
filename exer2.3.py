'''
    Description: Solutions to exercise 2.3 of [Sutton and Barto, 2016]
    Author: Tianlin Liu
    Date created: 6/20/2017
    Date last modified: 6/20/2017
    Python Version: 2.7
'''
import numpy as np
import matplotlib.pyplot as plt

def make_bandit_randwalk(k):
# generate a random bandit which start out equal and then take independent random walks
    #bandit = np.random.standard_normal(k)
    bandit = np.zeros(k)
    return bandit


def bandit_algorithm_sample_average(epsilon,steps,bandit):
    k = np.size(bandit)
    Q = np.zeros(k)
    N = np.zeros(k)
    AverageReward = np.zeros(steps)
    OptimalActionPercent = np.zeros(steps)
    
    bandit = bandit + [np.random.uniform(-0.5, 0.5) for i in range(k)]
    
    Optimal_Action = np.argmax(bandit)
    
    step = 0
    R_total = 0
    Optimal_Action_total = 0
    
    for step in range(steps):
        coin_flipped = np.random.random()
        if coin_flipped >= epsilon:
                A = np.argmax(Q)
        else: 
            A = np.random.randint(0, k)
    
        R = bandit[A] + np.random.standard_normal()
        R_total = R_total + R
        Optimal_Action_total = Optimal_Action_total + (A == Optimal_Action)
        
        N[A] = N[A] + 1
        Q[A] = Q[A] + (1.0/N[A])*(R - Q[A])
        
        if step == 0:
            AverageReward[step] = 0
            OptimalActionPercent[step] = 0
        else:
            AverageReward[step] = float(R_total)/float(step)
            OptimalActionPercent[step] = float(Optimal_Action_total) /float(step)
            
    return AverageReward, OptimalActionPercent
    
def bandit_algorithm_weighted_average(epsilon,steps,bandit):
    k = np.size(bandit)
    Q = np.zeros(k)
    N = np.zeros(k)
    AverageReward = np.zeros(steps)
    OptimalActionPercent = np.zeros(steps)
    
    bandit = bandit + [np.random.uniform(-1, 1) for i in range(k)]
    
    Optimal_Action = np.argmax(bandit)
    
    step = 0
    R_total = 0
    Optimal_Action_total = 0
    
    for step in range(steps):
        coin_flipped = np.random.random()
        if coin_flipped >= epsilon:
                A = np.argmax(Q)
        else: 
            A = np.random.randint(0, k)

        R = bandit[A] + np.random.standard_normal()
        R_total = R_total + R
        Optimal_Action_total = Optimal_Action_total + (A == Optimal_Action)
        
        N[A] = N[A] + 1
        Q[A] = Q[A] + (0.1)*(R - Q[A])
        
        if step == 0:
            AverageReward[step] = 0
            OptimalActionPercent[step] = 0
        else:
            AverageReward[step] = float(R_total)/float(step)
            OptimalActionPercent[step] = float(Optimal_Action_total) /float(step)
            
    return AverageReward, OptimalActionPercent
    
            
k = 10
steps = 2000
epsilon = 0.1
bandit_problem_nr = 100
AverageReward_sample_av_bandit_prob = np.zeros((bandit_problem_nr,steps))
OptimalActionPercent_sample_av_bandit_prob = np.zeros((bandit_problem_nr,steps))

AverageReward_weighted_av_bandit_prob = np.zeros((bandit_problem_nr,steps))
OptimalActionPercent_weighted_av_bandit_prob = np.zeros((bandit_problem_nr,steps))


for index in range(bandit_problem_nr):
    bandit = make_bandit_randwalk(k)
    
    AverageReward_sample_av, OptimalActionPercent_sample_av = bandit_algorithm_sample_average(epsilon,steps,bandit)    
    AverageReward_sample_av_bandit_prob[index,:] = AverageReward_sample_av
    OptimalActionPercent_sample_av_bandit_prob[index,:] = OptimalActionPercent_sample_av

    AverageReward_weighted_av, OptimalActionPercent_weighted_av = bandit_algorithm_weighted_average(epsilon,steps,bandit)    
    AverageReward_weighted_av_bandit_prob[index,:] = AverageReward_weighted_av
    OptimalActionPercent_weighted_av_bandit_prob[index,:] = OptimalActionPercent_weighted_av    
    
AverageReward_sample_av_bandit_overall = np.array(map(sum,zip(*AverageReward_sample_av_bandit_prob)))/bandit_problem_nr
OptimalActionPercent_sample_av_overall = np.array(map(sum,zip(*OptimalActionPercent_sample_av_bandit_prob)))/bandit_problem_nr
 
   
AverageReward_weighted_av_bandit_overall = np.array(map(sum,zip(*AverageReward_weighted_av_bandit_prob)))/bandit_problem_nr
OptimalActionPercent_weighted_av_overall = np.array(map(sum,zip(*OptimalActionPercent_weighted_av_bandit_prob)))/bandit_problem_nr
     
    
    
plt.figure(1)


plt.subplot(211)
line1,= plt.plot(range(steps), AverageReward_sample_av_bandit_overall, 'r',label="sample average")
line2,= plt.plot(range(steps), AverageReward_weighted_av_bandit_overall, 'b',label="weighted average")
plt.xlabel('Steps')
plt.ylabel('Average reward')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)
plt.title('epsilon = 0.1')


plt.subplot(212)
line1,= plt.plot(range(steps), OptimalActionPercent_sample_av_overall, 'r',label="sample average")
line2,= plt.plot(range(steps), OptimalActionPercent_weighted_av_overall, 'b',label="weighted average")
plt.xlabel('Steps')
plt.ylabel('% Optimal action')
#plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
#           ncol=2, mode="expand", borderaxespad=0.)
plt.title('epsilon = 0.1')

plt.show()


    
    