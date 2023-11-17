"""
A minimal implementation of Monte Carlo tree search (MCTS) in Python 3
Luke Harold Miles, July 2019, Public Domain Dedication
See also https://en.wikipedia.org/wiki/Monte_Carlo_tree_search
https://gist.github.com/qpwo/c538c6f73727e254fdc7fab81024f6e1
"""
from abc import ABC, abstractmethod
from collections import defaultdict
import math
from random import choice


class MCTS:
    "Monte Carlo tree searcher. First rollout the tree then choose a move."

    def __init__(self, exploration_weight=1):
        self.Q = defaultdict(int)  # total reward of each node
        self.N = defaultdict(int)  # total visit count for each node
        self.children = dict()  # children of each node
        self.exploration_weight = exploration_weight


    def choose(self, node):
        "Choose the best successor of node. (Choose a move in the game)"
        # print("THIS IS THE NODE THAT WANTS TO CHOOSE SOMETHING")
        # node.Draw()
        if node.isOver():
            raise RuntimeError(f"choose called on terminal node {node}")

        if node not in self.children:
            return node.find_random_child()

        def score(n):
            if self.N[n] == 0:
                return float("-inf")  # avoid unseen moves
            return self.Q[n] / self.N[n]  # average reward
        
        # for each in self.children[node]: 
        #     print("**START E BACHE HAYE NODE E ROOT")
            # grandchildren = self.children[each]
            # print(len(grandchildren))
            # for child in grandchildren:
            #     print(self.Q[child])
            #     print(self.N[child])
            #     print(score(child))
            #     child.Draw()

            
            # print(each.p1_actions)
            # print(each.p2_actions)

            # each.Draw()
            # print(self.Q[each])
            # print(self.N[each])
            # print(score(each))
            # input("*******")
        

        best_board = max(self.children[node], key=score)
        # return best_board
        return best_board.p2_actions


    def do_rollout(self, node, player):
        "Make the tree one layer better. (Train for one iteration.)"

        path = self._select(node, player)
        # print("PATH", path)
        leaf = path[-1]
        # print("LEAF", leaf)
        # print("TYEP E LEAF", leaf.nodeType)
        self._expand(leaf, player)
        # print("**********AFTER FIND/EXPAND e children")
        # print("STARTED SIMULATION")
        reward = self._simulate(leaf, player)
        # print("*****AFTER SIMULATE")
        # print("REWARD", reward)
        self._backpropagate(path, reward)
        # print("**AFTER BACKPROP")

    def _select(self, node, player):
        "Find an unexplored descendent of `node`"
        path = []
        # print("PLAYER IN SELECT")
        # print(player)
        while True:
            path.append(node)
            
            if node not in self.children or not self.children[node]:
                # node is either unexplored or terminal
                input("UMAD TOO IN IF")
                return path
            unexplored = self.children[node] - self.children.keys()

            if unexplored:
                n = unexplored.pop()
                path.append(n)
                return path
            node = self._uct_select(node, player)  # descend a layer deeper

    def _expand(self, node, player):
        "Update the `children` dict with the children of `node`"
        if node in self.children:
            return  # already expanded
        self.children[node] = node.find_children(player)



    def _simulate(self, node, player):
        "Returns the reward for a random simulation (to completion) of `node`"
        # print("begin simulate")
        invert_reward = True
        i =0 
        while True:
            if node.isOver() == True:

                if node.winner == 0:
                    AssertionError
                    print("BAD STATUS")
                elif node.winner ==2:
                    return 1
                elif node.winner == 1:
                    return 0
                elif node.winner == 0.5:
                    return 0.5
                else:
                    AssertionError
                    return None

            node = node.find_random_child(player)
            # print("\nINFO FOR the random child with i=", i)
            # print(node.nodeType)
            # node.Draw()
            # for u in node.unitList:
            #     print(u.name, ": ", u.hitpoints, "\t", u.x,"\t", u.y, "\t", u.x*4 + u.y)

            # print("****************\n")

            i+=1
            # invert_reward = not invert_reward


    def _backpropagate(self, path, reward):
        "Send the reward back up to the ancestors of the leaf"
        for node in reversed(path):
            self.N[node] += 1
            self.Q[node] += reward
            # reward = 1 - reward  # 1 for me is 0 for my enemy, and vice versa

    def _uct_select(self, node, player):
        "Select a child of node, balancing exploration & exploitation"

        # All children of node should already be expanded:
        assert all(n in self.children for n in self.children[node])
        log_N_vertex = math.log(self.N[node])
        def uct(n):
            "Upper confidence bound for trees"
            return self.Q[n] / self.N[n] + self.exploration_weight * math.sqrt(
                log_N_vertex / self.N[n]
            )
        if player =="max":
            return max(self.children[node], key=uct)
        elif player == "min":
            return min(self.children[node], key=uct)



class Node(ABC):
    """
    A representation of a single board state.
    MCTS works by constructing a tree of these Nodes.
    Could be e.g. a chess or checkers board state.
    """

    @abstractmethod
    def find_children(self):
        "All possible successors of this board state"
        return set()

    @abstractmethod
    def find_random_child(self):
        "Random successor of this board state (for more efficient simulation)"
        return None

    @abstractmethod
    def is_terminal(self):
        "Returns True if the node has no children"
        return True

    @abstractmethod
    def reward(self):
        "Assumes `self` is terminal node. 1=win, 0=loss, .5=tie, etc"
        return 0

    @abstractmethod
    def __hash__(self):
        "Nodes must be hashable"
        return 123456789

    @abstractmethod
    def __eq__(node1, node2):
        "Nodes must be comparable"
        return True
