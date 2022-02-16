# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best


        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]


        dis = 0
        x, y = newPos
        for fd in newFood.asList():
            fx, fy = fd
            if newFood[fx][fy]:
                dis = dis + abs(fx - x) + abs(fy - y) 

        cdis = 0
        for cp in currentGameState.getCapsules():
            cx, cy = cp
            cdis = cdis + abs(cx - x) + abs(cy - y) 


        for i in range(len(newGhostStates)):
            x, y = newGhostStates[i].getPosition()
            # print ("Ghost: ", newGhostStates[i].getPosition())
            GhostPossible = [(x, y), (x - 1, y), (x + 1, y), (x, y + 1), (x, y - 1)]
            if newScaredTimes[i] == 0 and newPos in GhostPossible:
                return -3000
        
        res = 0.1 * random.random() + 0.08 * -dis + 0.02 * cdis + 0.9 * (successorGameState.getScore() - currentGameState.getScore())

        "*** YOUR CODE HERE ***"
        return res

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def _value(self, gameState, move):
        if gameState.isLose() or gameState.isWin() or move == self.depth * gameState.getNumAgents():
            return (self.evaluationFunction(gameState), None)

        if move % gameState.getNumAgents() == 0:
            return self._max(gameState, move)
        else:
            return self._min(gameState, move)

    def _max(self, gameState, move):
        agentId = 0
        pmActs = gameState.getLegalActions(agentId)
        cur = -9999999
        best = None
        for act in pmActs:
            succGameState = gameState.generateSuccessor(agentId, act)
            if(cur < self._value(succGameState, move + 1)[0]):
                cur = self._value(succGameState, move + 1)[0]
                best = act
        return (cur, best)
            
    def _min(self, gameState, move):
        agentId = move % gameState.getNumAgents()
        ghActs = gameState.getLegalActions(agentId)
        cur = 9999999
        best = None
        for act in ghActs:
            succGameState = gameState.generateSuccessor(agentId, act)
            if(cur > self._value(succGameState, move + 1)[0]):
                cur = self._value(succGameState, move + 1)[0]
                best = act
        return (cur, best)

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        # for i in range(1, gameState.getNumAgents()):
            # ghActs = gameState.getLegalActions(i)
            # for act in ghActs:
        return self._value(gameState, 0)[1]
        


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def _value(self, gameState, move, alpha, beta):
        if gameState.isLose() or gameState.isWin() or move == self.depth * gameState.getNumAgents():
            return (self.evaluationFunction(gameState), None)

        if move % gameState.getNumAgents() == 0:
            return self._max(gameState, move, alpha, beta)
        else:
            return self._min(gameState, move, alpha, beta)

    def _max(self, gameState, move, alpha, beta):
        agentId = 0
        pmActs = gameState.getLegalActions(agentId)
        cur = -99999
        best = None
        for act in pmActs:
            succGameState = gameState.generateSuccessor(agentId, act)
            val = self._value(succGameState, move + 1, alpha, beta)[0]
            if cur < val:
                cur = val
                best = act
            if(cur > beta):
                return (cur, best)
            alpha = max(alpha, cur)
        return (cur, best)
            
    def _min(self, gameState, move, alpha, beta):
        agentId = move % gameState.getNumAgents()
        ghActs = gameState.getLegalActions(agentId)
        cur = 99999
        best = None
        for act in ghActs:
            succGameState = gameState.generateSuccessor(agentId, act)
            val = self._value(succGameState, move + 1, alpha, beta)[0]
            if cur > val:
                cur = val
                best = act
                if(cur < alpha):
                    return (cur, best)
                beta = min(beta, cur)
        return (cur, best)

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        # for i in range(1, gameState.getNumAgents()):
            # ghActs = gameState.getLegalActions(i)
            # for act in ghActs:
        alpha = -99999
        beta  =  99999
        return self._value(gameState, 0, alpha, beta)[1]


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def _value(self, gameState, move):
        if gameState.isLose() or gameState.isWin() or move == self.depth * gameState.getNumAgents():
            return (self.evaluationFunction(gameState), None)

        if move % gameState.getNumAgents() == 0:
            return self._max(gameState, move)
        else:
            return self._exp(gameState, move)

    def _max(self, gameState, move):
        agentId = 0
        pmActs = gameState.getLegalActions(agentId)
        cur = -9999999
        best = None
        for act in pmActs:
            succGameState = gameState.generateSuccessor(agentId, act)
            if(cur < self._value(succGameState, move + 1)[0]):
                cur = self._value(succGameState, move + 1)[0]
                best = act
        return (cur, best)
            
    def _exp(self, gameState, move):
        agentId = move % gameState.getNumAgents()
        ghActs = gameState.getLegalActions(agentId)
        numActs = len(ghActs)
        cur = 0
        for act in ghActs:
            succGameState = gameState.generateSuccessor(agentId, act)
            cur = cur + (self._value(succGameState, move + 1)[0] / numActs)
        return (cur, None)

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        return self._value(gameState, 0)[1]

def betterEvaluationFunction(currentGameState):

    def _dist(pos1, pos2):
        x1, y1 = pos1
        x2, y2 = pos2
        
        return abs(x1 - x2) + abs(y1 - y2)

    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    # Useful information you can extract from a GameState (pacman.py)
    # successorGameState = currentGameState.generatePacmanSuccessor(action)
    pos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]
    numFood = 0

    if currentGameState.isWin():
        return 99999;
    if currentGameState.isLose():
        return -99999;
    import math


    # food
    ftmp = 99999
    for fd in food.asList():
        fx, fy = fd
        if food[fx][fy]:
            numFood = numFood + 1
            ftmp = min(ftmp, _dist(pos, fd)) 
    fscore = -ftmp 

    # capsule
    ctmp = 99999
    for cp in currentGameState.getCapsules():
        ctmp = min(ctmp, _dist(pos, cp))
    cscore = -ctmp

    # ghost
    gscore = [ 0 for _ in ghostStates]
    for i in range(len(ghostStates)):
        # x, y = ghostStates[i].getPosition()
        # ghostPossible = [(x, y), (x - 1, y), (x + 1, y), (x, y + 1), (x, y - 1)]
        # if scaredTimes[i] == 0 and newPos in ghostPossible:
            # return -3000
        if scaredTimes[i] != 0:
            gscore[i] = -(_dist(pos, ghostStates[i].getPosition()) - scaredTimes[i]) / len(ghostStates)
        else:
            gscore[i] = _dist(pos, ghostStates[i].getPosition()) / len(ghostStates)
            
    
    res = 0.01 * random.random() + 0.1 * (1 - 1 / (1 + math.exp(numFood))) * (0.5 * cscore + 0.5 * sum(gscore)) + 0.3 * (1 / (1 + math.exp(numFood))) * fscore + 0.6 * currentGameState.getScore()
    return res


# Abbreviation
better = betterEvaluationFunction
