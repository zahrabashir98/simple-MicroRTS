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
import numpy as np
import random

# np.random.seed(1)
# random.seed(1)
class MCTS:

    "Monte Carlo tree searcher. First rollout the tree then choose a move."

    def __init__(self, exploration_weight=1):
        self.Q = defaultdict(int)  # total reward of each node
        self.N = defaultdict(int)  # total visit count for each node
        self.children = dict()  # children of each node
        self.exploration_weight = exploration_weight
        self.childrenObjects = dict()

    def choose(self, node):
        "Choose the best successor of node. (Choose a move in the game)"
        # print("THIS IS THE NODE THAT WANTS TO CHOOSE SOMETHING")
        # node.Draw()
        NodeHash = node.hash()
        if NodeHash[7] == True:
            raise RuntimeError(f"choose called on terminal node {NodeHash}")

        if NodeHash not in self.children:
            # print(NodeHash)
            # input("HEEERREEE")
            return node.find_random_child()

        def score(n):
            if self.N[n] == 0:
                return float("-inf")  # avoid unseen moves
            return self.Q[n] / self.N[n]  # average reward

        # for child in self.children[NodeHash]:
        #     print(child)
        #     print(self.Q[child]/self.N[child])
        #     # child.Draw()
        #     print(child[2])
        #     print(child[3])
        #     input()

        # best_board = max(self.children[node.hash()], key=score)
        best_board = max(self.children[NodeHash], key=score)
        return best_board
        # return best_board.p2_actions


    def do_rollout(self, node):
        "Make the tree one layer better. (Train for one iteration.)"

        path = self._select(node)
        print(node)
        # print("AFTER SELECT")
        print(path)
        leaf = path[-1]
        self._expand(leaf)
        # print("AFTER EXPAND")
        reward = self._simulate(leaf)
        print("REWARD", reward)
        self._backpropagate(path, reward)
        # print("AFTER BACKPROP\n")
        # input("******************\n")

    def _select(self, node):
        "Find an unexplored descendent of `node`"
        path = []
        # print("PLAYER IN SELECT")
        # print(player)
        while True:
            path.append(node)
            if node.hash() not in self.children or not self.children[node.hash()]:
                return path

            unexplored = self.childrenObjects[node] - self.childrenObjects.keys()
            # for each in self.children.keys():
            #     if each in self.children[node.hash()]:
            #         self.children[node.hash()].remove(each)
            # unexplored = self.children[node.hash()]
            if unexplored:
                n = unexplored.pop()
                path.append(n)
                return path
            node = self._uct_select(node)  # descend a layer deeper

    def _expand(self, node):
        "Update the `children` dict with the children of `node`"

        if node.hash() in self.children:
            print(node.hash())
            print("ALREADY EXPANDED")
            # input()
            return  # already expanded
        try:
            childernObjs, childrenHash = node.find_children()
        except:
            print(node.hash())
            print("INJA")
            input()
        self.children[node.hash()] = childrenHash
        self.childrenObjects[node] = childernObjs
        print((childrenHash))
        print("CHILDREN FOUND")



    def _simulate(self, node):
        "Returns the reward for a random simulation (to completion) of `node`"
        print("begin simulate")
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
            node.Draw()
            for u in node.unitList:
                print(u.name, ": ", u.hitpoints, "\t", u.x,"\t", u.y, "\t", u.x*4 + u.y)

            node = node.find_random_child()

            # print("\nINFO FOR the random child with i=", i)
            # print(node.nodeType)
            # node.Draw()

            # print("****************\n")

            i+=1
            # invert_reward = not invert_reward

        
    def _backpropagate(self, path, reward):
        "Send the reward back up to the ancestors of the leaf"
        for node in reversed(path):
            self.N[node.hash()] += 1
            self.Q[node.hash()] += reward
            # reward = 1 - reward  # 1 for me is 0 for my enemy, and vice versa

    def _uct_select(self, node):
        "Select a child of node, balancing exploration & exploitation"
        nodeHash = node.hash()
        # All children of node should already be expanded:
        assert all(n in self.children for n in self.children[nodeHash])
        log_N_vertex = math.log(self.N[nodeHash])
        def uct(n):
            "Upper confidence bound for trees"
            return self.Q[n] / self.N[n] + self.exploration_weight * math.sqrt(
                log_N_vertex / self.N[n]
            )
        def uctObj(n):
            "Upper confidence bound for trees"
            n = n.hash()
            return self.Q[n] / self.N[n] + self.exploration_weight * math.sqrt(
                log_N_vertex / self.N[n]
            )
        def uctMin(n):
            "Upper confidence bound for trees"
            return self.Q[n] / self.N[n] - self.exploration_weight * math.sqrt(
                log_N_vertex / self.N[n]
            )
        def uctObjMin(n):
            "Upper confidence bound for trees"
            n = n.hash()
            return self.Q[n] / self.N[n] - self.exploration_weight * math.sqrt(
                log_N_vertex / self.N[n]
            )
        
        if node.nodeType =="max":
            bestHash = max(self.children[nodeHash], key=uct)
            bestobj = max(self.childrenObjects[node], key=uctObj)
            if bestobj.hash() != bestHash:
                bestobj.Draw()
                print(bestHash)
                print("SHOULDN't have happened")
                AssertionError
            return bestobj

    
        elif node.nodeType == "min":
            bestHash = min(self.children[nodeHash], key=uctMin)
            bestobj = min(self.childrenObjects[node], key=uctObjMin)
            if bestobj.hash() != bestHash:
                bestobj.Draw()
                print(bestHash)
                print("SHOULDN't have happened")
                AssertionError
            return bestobj



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
