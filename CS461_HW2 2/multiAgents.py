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
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
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

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
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

        "*** YOUR CODE HERE ***"
        if action == Directions.STOP:
            return float("-inf")

        for ghost in newGhostStates:
            if ghost.getPosition() == newPos:
                return float("-inf")

        if newPos in currentGameState.getFood().asList():
            return float("inf")

        if newPos in currentGameState.getCapsules():
            return float("inf")

        if successorGameState.hasWall(newPos[0], newPos[1]):
            return float("-inf")

        # distance to closest food
        foodList = newFood.asList()
        distance = float("inf")
        for food in foodList:
            dist = manhattanDistance(newPos, food)
            if dist < distance:
                distance = dist

        score = successorGameState.getScore() - distance
        return score

    
def scoreEvaluationFunction(currentGameState: GameState):
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

    def getAction(self, gameState: GameState):
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
        def minimax(state, depth, agent):

            # Base case
            if state.isWin() or state.isLose() or depth == self.depth or state.getLegalActions(agent) == 0:
                return self.evaluationFunction(state), None

            val = float("-inf")
            max_action = None
            # For pacman agent
            if agent == 0:
                for action in state.getLegalActions(agent):
                    v1, a1 = minimax(state.generateSuccessor(agent, action), depth,
                                       (agent + 1) % state.getNumAgents())
                    if v1 > val:
                        val = v1
                        max_action = action
            if val != float("-inf"):
                return val, max_action

            val = float("inf")
            min_action = None
            # For ghost agent
            if agent != 0:
                for action in state.getLegalActions(agent):
                    # increase depth on the last ghost agent
                    new_depth = depth + 1 if ((agent + 1) % state.getNumAgents()) == 0 else depth
                    v1, a1 = minimax(state.generateSuccessor(agent, action), new_depth,
                                              (agent + 1) % state.getNumAgents())

                    if v1 < val:
                        val = v1
                        min_action = action
            if val != float("inf"):
                return val, min_action

        # main function call
        return minimax(gameState, 0, 0)[1]
     

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def alpha_beta_pruning(state, depth, agent, A, B):

            # Base case
            if state.isWin() or state.isLose() or depth == self.depth or state.getLegalActions(agent) == 0:
                return self.evaluationFunction(state), None

            val = float("-inf")
            max_action = None
            # For pacman agent
            if agent == 0:
                for action in state.getLegalActions(agent):
                    v1, a1 = alpha_beta_pruning(state.generateSuccessor(agent, action), depth,
                                                  (agent + 1) % state.getNumAgents(), A, B)
                    if v1 > val:
                        val = v1
                        max_action = action

                    if val > B:
                        return val, max_action
                    A = max(A, val)
            if val != float("-inf"):
                return val, max_action

            val = float("inf")
            min_action = None
            # For ghost agent
            if agent != 0:
                for action in state.getLegalActions(agent):
                    # increase depth on the last ghost agent
                    new_depth = depth + 1 if ((agent + 1) % state.getNumAgents()) == 0 else depth
                    v1, a1 = alpha_beta_pruning(state.generateSuccessor(agent, action), new_depth,
                                              (agent + 1) % state.getNumAgents(), A, B)
                    if v1 < val:
                        val = v1
                        min_action = action

                    if val < A:
                        return val, min_action
                    B = min(B, val)

            if val != float("inf"):
                return val, min_action

        # main function call
        return alpha_beta_pruning(gameState, 0, 0, float("-inf"), float("inf"))[1]
    

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        def expectimax(state, depth, agent):

            # Base case
            if state.isWin() or state.isLose() or depth == self.depth or state.getLegalActions(agent) == 0:
                return self.evaluationFunction(state), None

            val = float("-inf")
            max_action = None
            # For pacman agent
            if agent == 0:
                for action in state.getLegalActions(agent):
                    v1, a1 = expectimax(state.generateSuccessor(agent, action), depth,
                                          (agent + 1) % state.getNumAgents())
                    if v1 > val:
                        val = v1
                        max_action = action
            if val != float("-inf"):
                return val, max_action

            val = 0
            min_action = None
            count = 0
            # For ghost agent
            if agent != 0:
                for action in state.getLegalActions(agent):
                    # increase depth on the last ghost agent
                    new_depth = depth + 1 if ((agent + 1) % state.getNumAgents()) == 0 else depth
                    v1, a1 = expectimax(state.generateSuccessor(agent, action), new_depth,
                                              (agent + 1) % state.getNumAgents())

                    val += v1
                    count += 1
                    min_action = action
            if val != float("inf"):
                return val / count, min_action

        # main function call
        return expectimax(gameState, 0, 0)[1]
    
def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: This evaluation function takes the current game state and returns a score based on the following:
    1. The current score of the game state
    2. The distance to the closest food
    3. The distance to the closest capsule
    4. The distance to the closest active ghost
    5. The number of food left
    6. The number of capsules left

    The heuristic is a weighted sum of the above factors and they are decided by trial and error.
    The heuristic for the distances is Manhattan distance.
    To avoid division by zero, a small value is added to the denominator of  activeGhostdist and capsuleDist. 
    If the ghosts aren't active, i.e. scaredTimer is not 0, then the distance is set to -10.

    """
    "*** YOUR CODE HERE ***"
    currentScore = currentGameState.getScore()
    currentPos = currentGameState.getPacmanPosition()

    
    foodList = currentGameState.getFood().asList()
    foodNum = len(foodList)
    foodDist = float("inf")
    for food in foodList:
        foodDist = min(manhattanDistance(currentPos, food), foodDist)
    if foodNum == 0:
        foodDist = 0  

    ghostStates = currentGameState.getGhostStates()
    activeGhostdist = float("inf")
    for ghost in ghostStates:
        if ghost.scaredTimer == 0:
            activeGhostdist = min(manhattanDistance(currentPos, ghost.getPosition()), activeGhostdist)     
        else:
            activeGhostdist = -10
           
    
    capsuleList = currentGameState.getCapsules()
    capsuleNum = len(capsuleList)
    capsuleDist = float("inf")
    for capsule in capsuleList:
        capsuleDist = min(manhattanDistance(currentPos, capsule), capsuleDist)
    if capsuleNum == 0:
        capsuleDist = 1
        
    return currentScore-20/(activeGhostdist+1) -foodDist/2 - 10*foodNum - 10*capsuleNum - 10/(capsuleDist+1)
          


    

# Abbreviation
better = betterEvaluationFunction
