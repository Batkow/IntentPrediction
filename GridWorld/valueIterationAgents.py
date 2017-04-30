# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util, copy, random
import numpy as np
from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0

        # Write value iteration code here
        "*** YOUR CODE HERE ***"

        # Loop iterations times
        for k in range(0,iterations):
          # Get the states
          states = self.mdp.getStates()
          # Take the current value function
          v = copy.deepcopy(self.values)

          # Loop through the states
          for s in states:

            # Get the action from the value function
            action = self.computeActionFromValues(s)
            
            # If the action is not empty
            if action is not None:
              # Get the Q-value
              v[s] = self.computeQValueFromValues(s,action)

          # Update the values
          self.values = copy.deepcopy(v)


    def isAllowed(self,state):
        if state >= 0 and state <= 9 : return True
        return False

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        if state == "TERMINAL_STATE" or type(state[0]) == int or type(state[1]) == int:
          return self.values[state]

        else:

          x,y = state
          xInter = []
          yInter = []

          if type(x) != int:
            if self.isAllowed(int(np.round(x-1))):
              xInter.append(int(np.round(x-1)))

            if self.isAllowed(int(np.round(x+1))):
              xInter.append(int(np.round(x+1)))


          if type(y) != int:
            if self.isAllowed(int(np.round(y-1))):
              yInter.append(int(np.round(y-1)))

            if self.isAllowed(int(np.round(y+1))):
              yInter.append(int(np.round(y+1)))
          

          P = 0


          if len(xInter) < 2 and len(yInter) < 2:
            P = self.values[(xInter[0],yInter[0])]
            #print "Qvals", self.values[(xInter[0],yInter[0])]

          elif len(xInter) < 2:
            P = (yInter[1]-y) * self.values[(xInter[0],yInter[0])] + (y - yInter[0]) * self.values[(xInter[0],yInter[1])]
            #print "Qvals", self.values[(xInter[0],yInter[0])],self.values[(xInter[0],yInter[1])]

          elif len(yInter) < 2:
            P = (xInter[1]-x) * self.values[(xInter[0],yInter[0])] + (x - xInter[0]) * self.values[(xInter[1],yInter[0])]
            #print "Qvals", self.values[(xInter[0],yInter[0])],self.values[(xInter[1],yInter[0])]

          else :
            P = (xInter[1]-x)*(yInter[1]-y) * self.values[(xInter[0],yInter[0])] + \
                (x - xInter[0])*(yInter[1]-y) * self.values[(xInter[1],yInter[0])] + \
                (xInter[1]-x)*(y - yInter[0]) * self.values[(xInter[0],yInter[1])] + \
                (x - xInter[0])*(y - yInter[0]) * self.values[(xInter[1],yInter[1])]
            #print "Qvals", self.values[(xInter[0],yInter[0])],self.values[(xInter[1],yInter[0])],self.values[(xInter[0],yInter[1])],self.values[(xInter[1],yInter[1])]  

          #print len(self.values)
          return P






    def interpolateClosestStates(self,state):
      xLeft = int(np.floor(state[0]))
      xRight = int(np.ceil(state[0]))

      yUp = int(np.ceil(state[1]))
      yDown = int(np.floor(state[1]))

      states = []
      states.append((xLeft,yUp))
      states.append((xRight,yUp))
      states.append((xLeft,yDown))
      states.append((xRight,yDown))

      return states




    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        
        # Get the transition prob for each state
        transStatesAndProbs = self.mdp.getTransitionStatesAndProbs(state,action)
        
        # Initialize Q-value
        Q = 0
        
        # Loop through the states and increment to Q
        for [newState,transition] in transStatesAndProbs:
          dQ = transition*(self.mdp.getReward(state,action,newState) +
              self.discount*self.getValue(newState))
          Q += dQ

        # Return the Q-value
        return Q

        "*** YOUR CODE HERE ***"
        #util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"

        # Get possible actions
        actions = self.mdp.getPossibleActions(state)

        # If no possible actions
        if not actions:
          return None
        # If possible actiosn then continue
        else:
          # Holder for q value and its action
          qValueAndAction = []
          
          # Loop through actions and append the values
          for a in actions:
            qValueAndAction.append([self.getQValue(state,a), a])

          # print "-------"
          # print "state:", state
          # print "Qvals:",qValueAndAction
          # print "-------"
          # Return best action
          maxVal = max(qValueAndAction)[0]
          idx = [i for i, j in enumerate(qValueAndAction) if j[0] == maxVal]
          return actions[random.choice(idx)]

      

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)




if __name__ == '__main__':


    ###########################
    # GET THE GRIDWORLD
    ###########################

    import gridworld
    opts = gridworld.parseOptions()

    mdpFunction = getattr(gridworld, "get"+opts.grid)
    mdp = mdpFunction()
    mdp.setLivingReward(opts.livingReward)
    mdp.setNoise(opts.noise)
    env = gridworld.GridworldEnvironment(mdp)


