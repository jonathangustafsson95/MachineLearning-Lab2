import mdp, util, math

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

        for _ in range(iterations):    
            values_copy = self.values.copy()

            for state in self.mdp.getStates():
                vals = []
                possible_actions = self.mdp.getPossibleActions(state)
                
                if len(possible_actions) == 0:
                    values_copy[state] = 0
                else:
                    max_q = self.getQValue(state, possible_actions[0])
                    
                    for i in range(1, len(possible_actions)):
                        next_qval = self.getQValue(state, possible_actions[i])
                        if next_qval > max_q:
                            max_q = next_qval

                    values_copy[state] = max_q
            
            self.values = values_copy

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
        "*** YOUR CODE HERE ***"
        q_val = 0
        for trans, prob in self.mdp.getTransitionStatesAndProbs(state, action):
            q_val += prob * (self.mdp.getReward(state, action, trans) + self.discount * self.values[trans])

        return q_val
        
    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"

        if self.mdp.isTerminal(state):
            return None

        vals = util.Counter()
        for act in self.mdp.getPossibleActions(state):
            vals[act] = self.getQValue(state, act)

        if vals == []:
            return None
        else:
            return vals.argMax()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
