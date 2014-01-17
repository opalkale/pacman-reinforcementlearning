# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for 
# educational purposes provided that (1) you do not distribute or publish 
# solutions, (2) you retain this notice, and (3) you provide clear 
# attribution to UC Berkeley, including a link to 
# http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero 
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and 
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

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
        self.values = util.Counter() 
        
        for i in range(iterations): # running the alg on the indicated number of iterations
            y = self.values.copy() #V sub k-1
            
            for state in mdp.getStates():
                actions = util.Counter()
                
                if  mdp.isTerminal(state) == False:
                    for possibleActions in mdp.getPossibleActions(state):

                        for transitionState, prob in mdp.getTransitionStatesAndProbs(state, possibleActions):
                                value_iteration = prob * (mdp.getReward(state, possibleActions, transitionState) + (discount* y[transitionState]))
                                actions[possibleActions] += value_iteration
                    self.values[state] = actions[actions.argMax()] 
    
    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """

        QValue = 0 
        if self.mdp.isTerminal(state) == False:
            for transitionState, prob in self.mdp.getTransitionStatesAndProbs(state, action):
                
                QValue += prob * (self.mdp.getReward(state, action, transitionState) + (self.discount*self.values[transitionState]) )
        
        return QValue

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        if self.mdp.isTerminal(state) == False:
            QValue = -100000
            optimalAction = None
            possibleActions = self.mdp.getPossibleActions(state)
            
            for action in possibleActions:
                temp = self.getQValue(state, action)
                if temp > QValue:
                    optimalAction = action
                    QValue = temp
            return optimalAction
        else: 
            return None

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
