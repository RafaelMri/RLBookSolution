# -*- coding: utf-8 -*-
'''
    Description: This script is the solution to Exercise 5.8 [Sutton and Barto, 2016]
    Author: Tianlin Liu
    Date created: 6/28/2017
    Date last modified: 6/28/2017
    Python Version: 2.7
'''

import numpy as np 
import matplotlib.pyplot as plt
from pylab import *

# Define the racetrack
RaceTrack = np.zeros((32,17))
RaceTrack[0,3:17] =1
RaceTrack[1,2:17] =1
RaceTrack[2,2:17] =1
RaceTrack[3,1:17] =1
RaceTrack[4,0:17] =1
RaceTrack[5,0:17] =1
RaceTrack[6,0:10] =1
RaceTrack[7,0:9] =1
RaceTrack[8,0:9] =1
RaceTrack[9,0:9] =1
RaceTrack[10,0:9] =1
RaceTrack[11,0:9] =1
RaceTrack[12,0:9] =1
RaceTrack[13,0:9] =1
RaceTrack[14,1:9] =1
RaceTrack[15,1:9] =1
RaceTrack[16,1:9] =1
RaceTrack[17,1:9] =1
RaceTrack[18,1:9] =1
RaceTrack[19,1:9] =1
RaceTrack[20,1:9] =1
RaceTrack[21,1:9] =1
RaceTrack[22,2:9] =1
RaceTrack[23,2:9] =1
RaceTrack[24,2:9] =1
RaceTrack[25,2:9] =1
RaceTrack[26,2:9] =1
RaceTrack[27,2:9] =1
RaceTrack[28,2:9] =1
RaceTrack[29,3:9] =1
RaceTrack[30,3:9] =1
RaceTrack[31,3:9] =1


# plt.imshow(RaceTrack)
# plt.show()


# maximum velocity 
MAX_VELOCITY = 5


# The states in this racetrack task: [loc_y, loc_x, velo_y, velo_x]
[yIndex,xIndex] = np.where(RaceTrack)


# The coordinates in the racetrack
TrackCoordinates = []
for i in range(0, len(yIndex)):
    TrackCoordinates.append([yIndex[i],xIndex[i]])


# |states| = |coordinates in the trackroad| * |horizontal action| * |vertical action| = 10728  
states = []
for i in range(0, len(yIndex)):
     for velo_y in range(0, MAX_VELOCITY+1):
        for velo_x in range(0, MAX_VELOCITY+1):
            states.append([yIndex[i], xIndex[i], velo_y, velo_x])    

        

# Initialize Q(s,a) for each state (a 4-dim tuple) and each action (1,-1, or 0).
#Q = np.random.random((32,17,6,6,3,3))
Q = -50 * np.ones((32,17,6,6,3,3))



# Initialize C

C = np.zeros((32,17,6,6,3,3))


# Initialize the policy 
Policy = - inf * np.ones((32,17,6,6),dtype=object)

for i, j, speedY, speedX in states:
    Policy[i,j, speedY, speedX] = [0,0]


def randargmax(b,**kw):
  """ a random tie-breaking argmax"""
  return np.argmax(np.random.random(b.shape) * (b==b.max()), **kw)
  
  
def GreedyPolicy(StAt, Q, Policy):
    # given Q and s_t, let policy(s_t) = argmax_a Q(s_t,a)
    # print(int(St[0]), int(St[1]),int(St[2]),int(St[3]))
    ActionChoices = Q[int(StAt[0]), int(StAt[1]),int(StAt[2]),int(StAt[3]),:,:]    
    # ActionIndexChosed = list(np.unravel_index(randargmax(ActionChoices), ActionChoices.shape))
    ActionIndexChosed = list(np.unravel_index(np.argmax(ActionChoices), ActionChoices.shape))
    ActionValues = [ActionIndexChosed[0]-1, ActionIndexChosed[1]-1]
    NewPolicy = np.array(list(Policy))
    NewPolicy[int(StAt[0]), int(StAt[1]),int(StAt[2]),int(StAt[3])] = ActionValues
    
    return NewPolicy


#BehaviorPolicy = -inf * np.ones((32,17,3))

# for i,j in states:
#  BehaviorPolicy[i,j,:] = 0.333 # the uniform dist. for each action -1, 0 and +1
  

# probability that velocity increments are both 0
NOISE_PROB = 0.1


# Initialize the Speed

def GenerateEpisode(SoftPolicy, NOISE_PROB):
    ThisState = [31, int(np.random.choice([3,4,5,6,7,8])), 0, 0]
    t = 0
    SpeedVertical = 0
    SpeedHorizontal = 0
    EpisodeState = [ThisState]
    EpisodeAction = []

    while True:
        ActionLegal = False
        
        
        while (ActionLegal is False):
            # randomly select the acceleration rate to the horizontal and vertical
            coin =  random()
            
            if coin <=  0.1:
                ActionHorizontal = 0
                ActionVertical = 0
            else: 
                P = SoftPolicy[int(ThisState[0]), int(ThisState[1]), int(ThisState[2]), int(ThisState[3]),:,:] # Initializing P matrix
                P = P/P.sum() # Normalizing probability matrix
                ij = np.random.choice(range(P.size), p=P.ravel()) 
                ActionIndexVertical, ActionIndexHorizontal = np.unravel_index(ij, P.shape) 
                ActionVertical = ActionIndexVertical - 1
                ActionHorizontal = ActionIndexHorizontal - 1
                
                            
            # calculate the attempted speed
            AttemptSpeedVertical = ActionVertical + SpeedVertical
            AttemptSpeedHorizontal = ActionHorizontal + SpeedHorizontal
        
        
            # if the speed is negative or both zero, the action is not legal.
            if ((AttemptSpeedHorizontal >=0) and (AttemptSpeedVertical >=0)) and (AttemptSpeedHorizontal != 0 or AttemptSpeedVertical != 0 ):
                ActionLegal = True
                SpeedHorizontal = AttemptSpeedHorizontal
                SpeedVertical = AttemptSpeedVertical

            
                    
        # record the action (= acceleration)
        EpisodeAction.append([ActionVertical,ActionHorizontal])
        
        # increase the ocunter
        t += 1
        
        # update the state
        NextState = [ThisState[0] - SpeedVertical, ThisState[1] + SpeedHorizontal, SpeedVertical, SpeedHorizontal]
       
        
        if NextState in states:
           # the next state is still in the racetrack 
           ThisState = NextState
           EpisodeState.append(ThisState)
        elif (NextState[0] <= 5 and NextState[1] >= 16):       
           # the next state surpasses the finish line of the racetrack
           ThisState = [NextState[0],17, SpeedVertical,SpeedHorizontal]
           EpisodeState.append(ThisState)
           EpisodeReward = (-1)*np.ones(t+1) # R0 .... Rt needs t + 1 spots.
           EpisodeReward[t] = 0 # the last reward is 0
           EpisodeReward[0] = None # R0 is not defined.
           break
        else: # the next state hits the track boundary
            SpeedVertical = 0
            SpeedHorizontal = 0
            # go back to the starting line
            ThisState = [31, int(np.random.choice([3,4,5,6,7,8])), SpeedVertical, SpeedHorizontal]
            EpisodeState.append(ThisState)                          
        
    return t, EpisodeState, EpisodeAction, EpisodeReward


    
# Randomly choose a starting state on the starting line

# maximum number of episodes in MC control algorithm
MAX_EPISODE = 10000

# discount factor 
GAMMA = 1


EpisodeNr = 1

SoftPolicy = np.random.random((32,17,6,6,3,3))


while EpisodeNr < MAX_EPISODE:
    print EpisodeNr
    # Generate an episode using any soft policy mu
    t, EpisodeState, EpisodeAction, EpisodeReward = GenerateEpisode(SoftPolicy, NOISE_PROB)
    G = 0
    W = 1
    TimeIndex = np.arange(t)
    TimeIndexInverse = TimeIndex[::-1]
    backwardstep = 0
    for tInverse in TimeIndexInverse:
    #   tInverse = TimeIndexInverse[0]
        backwardstep += 1
        G = GAMMA*G + EpisodeReward[tInverse+1]
        ActionValues = EpisodeAction[tInverse]
        StAt = EpisodeState[tInverse]+ [ActionValues[0]+1,ActionValues[1]+1]
        C[StAt[0], StAt[1], StAt[2], StAt[3], StAt[4], StAt[5]] += W
        Q[StAt[0], StAt[1], StAt[2], StAt[3], StAt[4], StAt[5]] += (float(W)/(C[StAt[0], StAt[1], StAt[2], StAt[3], StAt[4], StAt[5]]))*(G - Q[StAt[0], StAt[1], StAt[2], StAt[3], StAt[4], StAt[5]])
        NewPolicy = GreedyPolicy([StAt[0], StAt[1],StAt[2],StAt[3]], Q, Policy)
        ActionByNewPolicy = NewPolicy[StAt[0],StAt[1], StAt[2], StAt[3]]
    #   NewPolicy[int(StAt[0]), int(StAt[1]),int(StAt[2]),int(StAt[3])]
        policyChanges = np.sum(NewPolicy != Policy)
        print('# changed policies: %d. backwardsteps: %d' %(policyChanges, backwardstep))
        Policy = np.array(list(NewPolicy))
        SoftPolicy[StAt[0], StAt[1], StAt[2], StAt[3],ActionByNewPolicy[0]+1,ActionByNewPolicy[1]+1] = 1
        SoftPolicy[StAt[0], StAt[1], StAt[2], StAt[3],:,:] = SoftPolicy[StAt[0], StAt[1], StAt[2], StAt[3],:,:]/((SoftPolicy[StAt[0], StAt[1], StAt[2], StAt[3],:,:]).sum()) # normalize
        
        
        
        if (EpisodeAction[tInverse] != ActionByNewPolicy):
            EpisodeNr +=  1
            break
        else:      
            # W = W*9
            W = W/SoftPolicy[StAt[0], StAt[1], StAt[2], StAt[3],ActionByNewPolicy[0]+1,ActionByNewPolicy[1]+1]

            # mu(A_t|S_t) = 1/9, so 1/mu(A_t|S_t)  = 9   
     
   
# Generate trajectories with learnt policies

def GenOptimalPolicyTrejactories(Policy):
    ThisState = [31, int(np.random.choice([3,4,5,6,7,8])), 0, 0]
    t = 0
    SpeedVertical = 0
    SpeedHorizontal = 0
    TrejactoriesState = [ThisState]
    TrejactoriesAction = []

    while True:     
        ActionVertical, ActionHorizontal = Policy[int(ThisState[0]),int(ThisState[1]),int(ThisState[2]),int(ThisState[3])]
        SpeedVertical += ActionVertical
        SpeedHorizontal += ActionHorizontal
                   
        # record the action (= acceleration)
        TrejactoriesAction.append([ActionVertical,ActionHorizontal])
        
        # increase the ocunter
        t += 1
        
        # update the state
        NextState = [ThisState[0] - SpeedVertical, ThisState[1] + SpeedHorizontal, SpeedVertical, SpeedHorizontal]
       
        
        if NextState in states:
           # the next state is still in the racetrack 
           ThisState = NextState
           TrejactoriesState.append(ThisState)
        elif (NextState[0] <= 5 and NextState[1] >= 16):       
           # the next state surpasses the finish line of the racetrack
           ThisState = [NextState[0],17, SpeedVertical,SpeedHorizontal]
           TrejactoriesState.append(ThisState)
           TrejactoriesReward = (-1)*np.ones(t+1) # R0 .... Rt needs t + 1 spots.
           TrejactoriesReward[t] = 0 # the last reward is 0
           TrejactoriesReward[0] = None # R0 is not defined.
           break
        else: # the next state hits the track boundary
            SpeedVertical = 0
            SpeedHorizontal = 0
            # go back to the starting line
            ThisState = [31, int(np.random.choice([3,4,5,6,7,8])), SpeedVertical, SpeedHorizontal]
            TrejactoriesState.append(ThisState)                          
        
    return t, TrejactoriesState, TrejactoriesAction, EpisodeReward   
   
            
    
    
        
        
        
        
        
        
        
        
        
        
        