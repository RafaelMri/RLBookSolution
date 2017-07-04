# -*- coding: utf-8 -*-
'''
    Description: This script aims to reconstruct exercise 6.7 of [Sutton and Barto, 2016]
    Author: Tianlin Liu
    Date created: 7/3/2017
    Date last modified: 7/3/2017
    Python Version: 2.7
'''

import numpy as np 
import matplotlib.pyplot as plt
    
        
# number of y coordiniates
DIM_Y = 7

# number of x coordinates
DIM_X = 10

# number of actions
Actions = np.array([[-1,0],[1,0],[0,1],[0,-1],[-1,1],[1,1],[1,-1],[-1,-1]]) # up, down, right, left, up-right, down-right, down-left, up-left.
DIM_ACTIONS = len(Actions)

# Initialize the Wind Matrix
WindMatrix = np.zeros((DIM_Y,DIM_X))
WindMatrix[:,[3,4,5,8]] = np.ones((DIM_Y,4))
WindMatrix[:,[6,7]] = 2*np.ones((DIM_Y,2))

def StochasticWind(State):
    
    StochasticWindOnState = int(WindMatrix[State[0],State[1]] + np.random.choice([1,-1,0]))
    
    return StochasticWindOnState



# The starting coordinate in the grid world
STARTING_STATE = np.array([3,0])

# The terminal coordinate in the grid world
TERMINAL_STATE = np.array([3,7])

# Discount Rate
GAMMA = 1


# States are just [y,x] coordinates in the grid world
States = []

for i in range(DIM_Y):
    for j in range(DIM_X):
        States.append([i,j])
        
DIM_STATE = len(States)

# Compute the next state, given the current state and the current action
def GenNextStateAndReward(State, Move):
    NextStateNoWind = [0,0]
    NextStateAfterWind = [0,0]
    
    NextStateTrial = np.array(State) + np.array(Move)
    NextStateNoWind[0] = min(6,max(0, NextStateTrial[0]))        
    NextStateNoWind[1] = min(9,max(0, NextStateTrial[1]))

    StochasticWindOnState =  StochasticWind(NextStateNoWind)
    
    NextStateAfterWindTrial = [int(NextStateNoWind[0] - StochasticWindOnState),NextStateNoWind[1]]
    
    NextStateAfterWind[0] = min(6,max(0, NextStateAfterWindTrial[0]))        
    NextStateAfterWind[1] = min(9,max(0, NextStateAfterWindTrial[1]))
    
    if (NextStateAfterWind == TERMINAL_STATE).all():
        Reward = 0
    else:
        Reward = -1
    return NextStateAfterWind, Reward      
        

def randargmax(b,**kw):
  """ a random tie-breaking argmax"""
  return np.argmax(np.random.random(b.shape) * (b==b.max()), **kw)


# Initialize the Q values
Q = np.zeros((DIM_Y,DIM_X,DIM_ACTIONS))

# Initialize a Policy Matrix. 
EpsilonGreedyPolicy = np.zeros((DIM_Y,DIM_X,DIM_ACTIONS))

# The soft threshold epsilon
EPSILON = 0.1

# The learning rate alpha
ALPHA = 0.5


# Define the epsilon greedy policy, given a state and a Q        
def GenActionFromSoftPolicy(State, Q, EPSILON):
    #GreedyActionIndex = np.(Q[State[0],State[1],:])
    # GreedyAction = Actions[GreedyActionIndex]
    GreedyActionIndex = randargmax(Q[State[0],State[1],:])
    
    ActionDistVector = ((EPSILON/float(DIM_ACTIONS))*np.ones(DIM_ACTIONS))
    ActionDistVector[GreedyActionIndex] += 1 - EPSILON
    ActionIndex = np.random.choice(np.arange(0, DIM_ACTIONS), p=ActionDistVector)
    
    return ActionIndex
    


# The number of episodes in the traning 
EPISODE_NUMBER = 1000


  


# Initialize the episode number
EpiNr = 0
State = STARTING_STATE
TimeStep = 0
EpiNrList = []
TimeStepList = []


for EpiNr in range(EPISODE_NUMBER):
    State = STARTING_STATE
    ActionIndex = GenActionFromSoftPolicy(State, Q, EPSILON)  
    ChosenAction = Actions[ActionIndex]    
    while True:        
        NextState, Reward = GenNextStateAndReward(State,ChosenAction)
        if Reward == 0:
            break
        NextActionIndex = GenActionFromSoftPolicy(NextState, Q, EPSILON)  
        
        Q[State[0],State[1],ActionIndex] += ALPHA*(Reward + GAMMA* Q[NextState[0], NextState[1], NextActionIndex] -  Q[State[0],State[1],ActionIndex])
        
        State = NextState
        ActionIndex = NextActionIndex
        ChosenAction = Actions[ActionIndex]    
        TimeStep += 1
        
        EpiNrList.append(EpiNr)
        TimeStepList.append(TimeStep)
        
        # print State, ChosenAction, EpiNr, TimeStep


plt.plot(TimeStepList, EpiNrList, 'k')
plt.xlabel('Time Steps')
plt.ylabel('Episodes')
plt.title('Results of Sarsa applied to the windy gridworld')
plt.show()

         
                
# Generate the moves according to the optimal policy learnt                       
                                     
State = STARTING_STATE
ActionIndex = GenActionFromSoftPolicy(State, Q, 0)  
ChosenAction = Actions[ActionIndex]    

MovesString = np.array(['up', 'down', 'right', 'left', 'up-right', 'down-right', 'down-left', 'up-left'])      

OptStatesAndMoves = [STARTING_STATE,MovesString[ActionIndex]]

            
while True:        
    NextState, Reward = GenNextStateAndReward(State,ChosenAction)
    if Reward == 0:
        OptStatesAndMoves.append([NextState, MovesString[ActionIndex]])
        break
    NextActionIndex = GenActionFromSoftPolicy(NextState, Q, 0)  
    OptStatesAndMoves.append([NextState, MovesString[NextActionIndex]])
    
    
    Q[State[0],State[1],ActionIndex] += ALPHA*(Reward + GAMMA* Q[NextState[0], NextState[1], NextActionIndex] -  Q[State[0],State[1],ActionIndex])
    
    State = NextState
    ActionIndex = NextActionIndex
    ChosenAction = Actions[ActionIndex]    
    TimeStep += 1
    
print OptStatesAndMoves    
            
        
        
        