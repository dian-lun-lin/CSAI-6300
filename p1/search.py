# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    """
    "*** YOUR CODE HERE ***"
    from game import Directions
    from util import Stack, Counter

    
    # init
    dfs = Stack()
    rs = Stack()
    act = []
    visit = Counter()
    start = problem.getStartState()
    visit[start] = 1
    for suc in problem.getSuccessors(start):
        dfs.push((start, suc))

    # dfs
    while not dfs.isEmpty():
        s = dfs.pop()
        visit[s[1][0]] = 1
        act.append(s[1][1])
        rs.push(s[0])

        if problem.isGoalState(s[1][0]):
            return act

        # empty
        # reverse untill get to org status
        go = False
        for suc in problem.getSuccessors(s[1][0]):
            if visit[suc[0]] == 0:
                dfs.push((s[1][0], suc))
                go = True
        if not go:
            act.pop()
            while rs.pop() != dfs.list[-1][0]:
                act.pop()


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    from util import Queue, Counter

    
    # init
    bfs = Queue()
    visit = Counter()
    start = problem.getStartState()
    visit[start] = 1

    # dst, path
    for suc in problem.getSuccessors(start):
        visit[suc[0]] = 1
        bfs.push((suc[0], [suc[1]]))

    # dfs
    while not bfs.isEmpty():
        s = bfs.pop()
        cur_node = s[0]
        path = s[1]

        if problem.isGoalState(cur_node):
            return path

        for suc_tuple in problem.getSuccessors(cur_node):
            suc = suc_tuple[0]
            act = suc_tuple[1]
            if visit[suc] == 0:
                visit[suc] = 1
                bfs.push((suc, path + [act]))

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    from util import PriorityQueue, Counter
    
    # init
    pq = PriorityQueue()
    dist = Counter()
    visit = Counter()
    start = problem.getStartState()
    dist[start] = -1
    visit[start] = 1
    act_dict = {}

    # dst
    for suc in problem.getSuccessors(start):
        visit[suc[0]] = 1
        act_dict[suc[0]] = [suc[1]]

        dist[suc[0]] = problem.getCostOfActions(act_dict[suc[0]])
        pq.push(suc[0], problem.getCostOfActions(act_dict[suc[0]]))

    # dfs
    while not pq.isEmpty():
        s = pq.pop()
        cur_node = s
        acs = act_dict[s]
        cur_cost = problem.getCostOfActions(acs)

        if problem.isGoalState(cur_node):
            return acs

        for suc_tuple in problem.getSuccessors(cur_node):
            suc = suc_tuple[0]
            act = suc_tuple[1]
            cost = suc_tuple[2]

            total_cost = problem.getCostOfActions(acs + [act])


            if visit[suc] == 0:
                visit[suc] = 1
                dist[suc] = total_cost
                act_dict[suc] = acs + [act]
                pq.push(suc, total_cost)
            elif dist[suc] > total_cost:
                dist[suc] = total_cost
                act_dict[suc] = acs + [act]
                pq.update(suc, total_cost)

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    from util import PriorityQueue, Counter
    from searchAgents import manhattanHeuristic
    
    # init
    pq = PriorityQueue()
    dist = Counter()
    visit = Counter()
    start = problem.getStartState()
    dist[start] = -1
    visit[start] = 1
    act_dict = {}

    # dst
    for suc in problem.getSuccessors(start):
        visit[suc[0]] = 1
        act_dict[suc[0]] = [suc[1]]

        dist[suc[0]] = problem.getCostOfActions(act_dict[suc[0]]) + heuristic(suc[0], problem)
        pq.push(suc[0], problem.getCostOfActions(act_dict[suc[0]]) + heuristic(suc[0], problem))
    # dfs
    while not pq.isEmpty():
        s = pq.pop()
        cur_node = s
        acs = act_dict[s]
        cur_cost = problem.getCostOfActions(acs)

        if problem.isGoalState(cur_node):
            return acs

        for suc_tuple in problem.getSuccessors(cur_node):
            suc = suc_tuple[0]
            act = suc_tuple[1]
            total_cost = problem.getCostOfActions(acs + [act])

            if visit[suc] == 0:
                visit[suc] = 1
                dist[suc] = total_cost
                act_dict[suc] = acs + [act]
                pq.push(suc, total_cost + heuristic(suc, problem))
            elif dist[suc] > total_cost:
                dist[suc] = total_cost
                act_dict[suc] = acs + [act]
                pq.update(suc, total_cost + heuristic(suc, problem))


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
