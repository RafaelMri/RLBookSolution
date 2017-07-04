# -*- coding: utf-8 -*-
'''
    Description: This script reconstructs Fig. 4.2 of [Sutton and Barto, 2016]
    Author: Tianlin Liu
    Date created: 6/21/2017
    Date last modified: 6/22/2017
    Python Version: 2.7
    
    Note: This scripts is partially credited to https://github.com/ShangtongZhang/reinforcement-learning-an-introduction/blob/master/chapter04/CarRental.py
'''

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import math


def PoissonPmf(lam,k):
    pmf = float((lam**k)*(np.exp(-lam)))/(math.factorial(k))
    return pmf


# the reward for each car rented
RENT_CREDIT = 10
LAMBDA_RENT_LOC1 = 3
LAMBDA_RENT_LOC2 = 4
LAMBDA_RETURN_LOC1 = 3
LAMBDA_RETURN_LOC2 = 2
MAX_NR_CARS = 20
CAR_MOVING_COSTS = 2
MAX_MOVE_OF_CARS = 5

states = []
for i in range(0, MAX_NR_CARS + 1):
    for j in range(0, MAX_NR_CARS + 1):
        states.append([i, j])    



def ExpectedReturn(StateTuple, Action, Values, GAMMA):
    # Input: s, a, and V(s') for all s', 
    # Output: V(s) = \sum_{s',r} p(s',r|s,a) [r + \GAMMA V(s')].
    returns = 0 
    TotalCarMovingCost = 0
    
    if Action >= 1:  
        TotalCarMovingCost = (abs(Action)-1)*CAR_MOVING_COSTS # the reward of car transportation, if at least one car move from loc 1 to 2
    else:
        TotalCarMovingCost = abs(Action)*CAR_MOVING_COSTS # the reward of car transportation, if no car move from loc 1 to 2
        
    returns -= TotalCarMovingCost # the reward of car transportation
    
    CarsAfterMoved = np.zeros(2)
    CarsAfterRent = np.zeros(2)
    CarsAfterReturn = np.zeros(2)
    CarsAfterMoved[0] = int(min(StateTuple[0] - Action, MAX_NR_CARS))
    CarsAfterMoved[1] = int(min(StateTuple[1] + Action, MAX_NR_CARS))  
    
    ParkingFee = 4*(max(CarsAfterMoved[0] - 10, 0) + max(CarsAfterMoved[1] - 10, 0))
    returns -= ParkingFee
                                           
    for NrRentLoc1 in range(11): # the situation that 0 - 10 cars are rented from loc 1
        for NrRentLoc2 in range(11): # the situation that 0 - 10 cars are rented from loc 2
            RentableCarsLoc1 = int(min(NrRentLoc1,CarsAfterMoved[0]))
            RentableCarsLoc2 = int(min(NrRentLoc2,CarsAfterMoved[1]))
            
            RentReward = (RentableCarsLoc1 + RentableCarsLoc2)*RENT_CREDIT  
            CarsAfterRent[0] = CarsAfterMoved[0] - RentableCarsLoc1
            CarsAfterRent[1] = CarsAfterMoved[1] - RentableCarsLoc2
            
            approx_return = True
            
            if approx_return:
                NrReturnLoc1 = LAMBDA_RETURN_LOC1
                NrReturnLoc2 = LAMBDA_RETURN_LOC2
                CarsAfterReturn[0] = min(MAX_NR_CARS, CarsAfterRent[0] + NrReturnLoc1)
                CarsAfterReturn[1] = min(MAX_NR_CARS, CarsAfterRent[1] + NrReturnLoc2)
                Probability = PoissonPmf(LAMBDA_RENT_LOC1,NrRentLoc1)*PoissonPmf(LAMBDA_RENT_LOC2,NrRentLoc2)
                returns += Probability*(RentReward + GAMMA*Values[int(CarsAfterReturn[0]),int(CarsAfterReturn[1])])              
                
            else: 
                for NrReturnLoc1 in range(11):  # the situation that 0 - 10 cars are returned from loc 1
                    for NrReturnLoc2 in range(11):  # the situation that 0 - 10 cars are rented from loc 2
                        CarsAfterReturn[0] = min(MAX_NR_CARS, CarsAfterRent[0] + NrReturnLoc1)
                        CarsAfterReturn[1] = min(MAX_NR_CARS, CarsAfterRent[1] + NrReturnLoc2)
                        Probability = PoissonPmf(LAMBDA_RENT_LOC1,NrRentLoc1)*PoissonPmf(LAMBDA_RENT_LOC2,NrRentLoc2)*PoissonPmf(LAMBDA_RETURN_LOC1,NrReturnLoc1)*PoissonPmf(LAMBDA_RETURN_LOC2,NrReturnLoc2)
                        returns += Probability*(RentReward + GAMMA*Values[int(CarsAfterReturn[0]),int(CarsAfterReturn[1])])
            
                                    
    return returns





def PolicyIteration(Values,Actions,Policy,THETA,GAMMA):
    #Initialization
    
    ValuesUpdates = np.zeros(Values.shape)
    while True:   
                
        #Policy Evaluation Step
        DELTA = THETA + 1
        while (DELTA > THETA):
            for i,j in states:
                ValuesUpdates[i,j] = ExpectedReturn([i,j], Policy[i,j], Values, GAMMA)
            DELTA = np.sum(abs(ValuesUpdates - Values)) 
            # print(DELTA)   
            Values[:] = ValuesUpdates
        
        #Policy Improvement
        PolicyUpdates = np.zeros(Policy.shape)

        for i,j in states:
            ValuesCandidates = []            
            for ThisAction in Actions:  
               if (ThisAction >= 0 and i >= ThisAction) or (ThisAction < 0 and j >= abs(ThisAction)):             
                    ValuesCandidates.append(ExpectedReturn([i,j], ThisAction, Values, GAMMA))
               else:
                   ValuesCandidates.append(-float('inf'))
            BestAction = np.argmax(ValuesCandidates)
            PolicyUpdates[i,j] = Actions[BestAction]
        
                    
        policyChanges = np.sum(PolicyUpdates != Policy)
        print('Policy for', policyChanges, 'states changed')
        Policy = PolicyUpdates
        
        if (policyChanges == 0):
            break
                   
    return Values, Policy
    

Actions = np.arange(-MAX_MOVE_OF_CARS, MAX_MOVE_OF_CARS + 1)
Values = np.zeros((MAX_NR_CARS + 1, MAX_NR_CARS + 1))
Policy = np.zeros((MAX_NR_CARS + 1, MAX_NR_CARS + 1))
THETA = 1e-4
GAMMA = 0.9

FinalValues, FinalPolicy = PolicyIteration(Values,Actions,Policy,THETA,GAMMA)

# Plot the figure

        


fig = plt.figure()
ax = fig.gca(projection='3d')

# Make data.
X = np.arange(0, 21, 1)
Y = np.arange(0, 21, 1)
X, Y = np.meshgrid(X, Y)

        
        
surf = ax.plot_surface(X, Y, FinalPolicy, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)


# Customize the z axis.
ax.set_zlim(-5, 5)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
ax.set_xlabel('# of cars in first location')
ax.set_ylabel('# of cars in second location')
ax.set_zlabel('# of cars to move during night')
# ax.invert_xaxis()

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()



