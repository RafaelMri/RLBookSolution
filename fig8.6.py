# -*- coding: utf-8 -*-
'''
    Description: This script aims to reconstruct Fig. 6.4 of [Sutton and Barto, 2016]
    Author: Tianlin Liu
    Date created: 7/5/2017
    Date last modified: 7/6/2017
    Python Version: 2.7
'''

import numpy as np 
import matplotlib.pyplot as plt
import random

        
# number of y coordiniates
DIM_Y = 6

# number of x coordinates
DIM_X = 9

# number of actions
# Actions = [[-1,0],[1,0],[0,1],[0,-1]] # up, down, right, and left.
Actions = [[-1,0],[0,1],[0,-1]] # up, down, right, and left.

DIM_ACTIONS = len(Actions)

# Specify the coordinates of the Blocks 
BlocksNoShortCut = [[3,1],[3,2],[3,3],[3,4],[3,5],[3,6],[3,7],[3,8]]
BlocksShortcut = [[3,1],[3,2],[3,3],[3,4],[3,5],[3,6],[3,7]]


# The starting coordinate in the grid world
STARTING_STATE = [5,3]

# The terminal coordinate in the grid world
TERMINAL_STATE = [0,8]

# Discount Rate
GAMMA = 1

# list all coordinates
AllCoordinates = []

for i in range(DIM_Y):
    for j in range(DIM_X):
        AllCoordinates.append([i,j])

# States are just coordinates which are not blocks

StatesNoShortCut = [x for x in AllCoordinates if x not in BlocksNoShortCut]
StatesShortCut = [x for x in AllCoordinates if x not in BlocksShortcut]

                
DIM_NO_SHORTCUT = len(StatesNoShortCut)
DIM_SHORTCUT = len(StatesShortCut)


# The model function: given the current state and the current action, output the next state, and next reward
def ModelDyna(AllStates, State, ActionIndex):
    NextStateBlocked = [0,0]
    NextStateGW = [0,0]
    Move = Actions[ActionIndex]
    
    NextStateTrial = np.array(State) + np.array(Move)
    if list(NextStateTrial) in AllStates:
        NextStateBlocked = NextStateTrial
    else:
        NextStateBlocked = State
        
        
    NextStateGW[0] = min(5,max(0, NextStateBlocked[0]))        
    NextStateGW[1] = min(8,max(0, NextStateBlocked[1]))
    
    if (NextStateGW == TERMINAL_STATE):
        Reward = 1
    else:
        Reward = -0.01
    return NextStateGW, Reward      
    
    
KAPPA = 0.0055
def ModelDynaPlus(AllStates, State, ActionIndex, TimeSpanNotVisited, KAPPA):
    NextStateBlocked = [0,0]
    NextStateGW = [0,0]
        
    
    Move = Actions[ActionIndex]
    
    
    NextStateTrial = np.array(State) + np.array(Move)
    if list(NextStateTrial) in AllStates:
        NextStateBlocked = NextStateTrial
    else:
        NextStateBlocked = State
        
        
        
    NextStateGW[0] = min(5,max(0, NextStateBlocked[0]))        
    NextStateGW[1] = min(8,max(0, NextStateBlocked[1]))
    
    if (NextStateGW == TERMINAL_STATE):
        Reward = 1
    else:
        #Reward = min( -0.01 + int(ActionIndex != 1)*(KAPPA*np.sqrt(TimeSpanNotVisited)), 0.001)
        Reward =  -0.01 + int(ActionIndex != 1)*(KAPPA*np.sqrt(TimeSpanNotVisited))
        
    return NextStateGW, Reward      
        
    
        

def randargmax(b,**kw):
  """ a random tie-breaking argmax"""
  return np.argmax(np.random.random(b.shape) * (b==b.max()), **kw)


# Initialize the Q values
Q = np.zeros((DIM_Y,DIM_X,DIM_ACTIONS))

# Initialize a Policy Matrix. 
EpsilonGreedyPolicy = np.zeros((DIM_Y,DIM_X,DIM_ACTIONS))

# The soft threshold epsilon
EPSILON = 0.15

# The learning rate alpha
ALPHA = 0.1


# Define the epsilon greedy policy, given a state and a Q        
def GenActionFromSoftPolicy(State, Q, EPSILON):
    GreedyActionIndex = randargmax(Q[State[0],State[1],:])
    
    ActionDistVector = ((EPSILON/float(DIM_ACTIONS))*np.ones(DIM_ACTIONS))
    ActionDistVector[GreedyActionIndex] += 1 - EPSILON
    # ActionDistVector[1] = 0
    # ActionDistVector = np.array(ActionDistVector)/sum(np.array(ActionDistVector))
    
    ActionIndex = np.random.choice(np.arange(0, DIM_ACTIONS), p=ActionDistVector)
    
    return ActionIndex
    


# The number of episodes in the traning 
EPISODE_NUMBER = 1000
STEPS_NUMBER = 6000

# Time steps for model planning
MODEL_PLANNING_NUMBER = 5




'''
Learn and plan in the shortcut maze using Dyna-Q
'''

# Initialize the episode number
EpiNr = 0
State = STARTING_STATE
TimeStep = 0
CumReward = 0 
CumRewardListDynaQ = []
CumRewardListDynaQPlus = []
breaking = False

for EpiNr in range(EPISODE_NUMBER):
    PrevObsStatesAndAction = []
    State = STARTING_STATE
    while True:
        ActionIndex = GenActionFromSoftPolicy(State, Q, EPSILON)          
        ChosenAction = Actions[ActionIndex]
        PrevObsStatesAndAction.append([State,ActionIndex])
        TimeStep += 1
    
        
        if (TimeStep <= 3000):
            AllStates = StatesNoShortCut
        else:
            AllStates = StatesShortCut
            
        NextState, Reward = ModelDyna(AllStates,State,ActionIndex)
        
        CumReward += Reward
        
        CumRewardListDynaQ.append(CumReward)        
        
                                      
        if TimeStep > STEPS_NUMBER:        
            breaking = True
            break
        
        if NextState == [0,8]:
            break
            

        
                
        Q[State[0],State[1],ActionIndex] += ALPHA*(Reward + GAMMA* max(Q[NextState[0], NextState[1], :]) -  Q[State[0],State[1],ActionIndex])
         
        for PlanningSteps in range(MODEL_PLANNING_NUMBER):
            random_index = random.randrange(0,len(PrevObsStatesAndAction))
            RandPrevState, RandPrevActionIndex = PrevObsStatesAndAction[random_index]
            SimulateNextState, SimulateReward = ModelDyna(AllStates,RandPrevState,RandPrevActionIndex)
            Q[State[0],State[1],ActionIndex] += ALPHA*(Reward + GAMMA* max(Q[NextState[0], NextState[1], :]) -  Q[State[0],State[1],ActionIndex])
            
            
        
        State = NextState
        
    if breaking:
        break

        
        # print State, ChosenAction, EpiNr, TimeStep

        
'''
Learn and plan in the shortcut maze using Dyna-Q+
'''

# Initialize the Q values
Q = np.zeros((DIM_Y,DIM_X,DIM_ACTIONS))

# Initialize a Policy Matrix. 
EpsilonGreedyPolicy = np.zeros((DIM_Y,DIM_X,DIM_ACTIONS))


# Initialize the episode number
EpiNr = 0
State = STARTING_STATE
TimeStep = 0
CumReward = 0 
CumRewardListDynaQPlus = []
breaking = False
LatestVisitTimeStep = np.zeros(DIM_ACTIONS)        


for EpiNr in range(EPISODE_NUMBER):
    PrevObsStatesAndAction = []
    State = STARTING_STATE
    
    
    
    while True:
        ActionIndex = GenActionFromSoftPolicy(State, Q, EPSILON)
        TimeStep += 1
                 
            
        ChosenAction = Actions[ActionIndex]
        PrevObsStatesAndAction.append([State,ActionIndex])

        
        if (TimeStep <= 3000):
            AllStates = StatesNoShortCut
        else:
            AllStates = StatesShortCut
        
        TimeSpanNotVisited = TimeStep - LatestVisitTimeStep[ActionIndex]      
        
        
        #print LatestVisitTimeStep,  TimeStep, ActionIndex
        
        NextState, Reward = ModelDynaPlus(AllStates, State, ActionIndex, TimeSpanNotVisited , KAPPA)

        print TimeSpanNotVisited, Reward

                
        CumReward += Reward
        CumRewardListDynaQPlus.append(CumReward)                
        
        LatestVisitTimeStep[ActionIndex] =  TimeStep          

                
        if TimeStep > STEPS_NUMBER:        
            breaking = True
            break
            
        if NextState == [0,8]:
            break
        
                
        Q[State[0],State[1],ActionIndex] += ALPHA*(Reward + GAMMA* max(Q[NextState[0], NextState[1], :]) -  Q[State[0],State[1],ActionIndex])
         
        for PlanningSteps in range(MODEL_PLANNING_NUMBER):
            random_index = random.randrange(0,len(PrevObsStatesAndAction))
            RandPrevState, RandPrevActionIndex = PrevObsStatesAndAction[random_index]
            
            
            SimulateNextState, SimulateReward = ModelDynaPlus(AllStates,RandPrevState,RandPrevActionIndex, TimeSpanNotVisited, KAPPA)
            Q[State[0],State[1],ActionIndex] += ALPHA*(Reward + GAMMA* max(Q[NextState[0], NextState[1], :]) -  Q[State[0],State[1],ActionIndex])
            
            
        
        State = NextState
        
    if breaking:
        break
        
        # print State, ChosenAction, EpiNr, TimeStep



                        
                                                
                                                                        
                                                                                                                        

# plt.plot(range(TimeStep), CumRewardListDynaQ, 'k')
plt.plot(range(TimeStep), CumRewardListDynaQ, 'k', range(TimeStep), CumRewardListDynaQPlus, 'r')

plt.xlabel('Time Steps')
plt.ylabel('Episodes')
plt.title('Tabular Dyna-Q and Tabular Dyna-Q+ applied on Shortcut maze')
plt.show()

         
                
                
PrevObsStatesAndAction = []
State = STARTING_STATE
CumReward = 0
AllStates = StatesShortCut
TimeStep = 0   
    
while True:
    ActionIndex = GenActionFromSoftPolicy(State, Q, 0)
    TimeStep += 1
                 
            
    ChosenAction = Actions[ActionIndex]
    PrevObsStatesAndAction.append([State,ActionIndex])
                
    NextState, Reward = ModelDynaPlus(AllStates, State, ActionIndex, TimeSpanNotVisited , KAPPA)
     
    print NextState     
    State = NextState
                    
    if State == [0,8]:
        break
                 
        
    
print PrevObsStatesAndAction
        


            
        
        
        