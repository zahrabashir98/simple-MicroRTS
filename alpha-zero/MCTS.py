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
from NN import *

# np.random.seed(1)
# random.seed(1)
class MCTS:

    "Monte Carlo tree searcher. First rollout the tree then choose a move."

    def __init__(self, exploration_weight=1.5):
        self.Q = defaultdict(int)  # total reward of each node
        self.N = defaultdict(int)  # total visit count for each node
        self.children = dict()  # children of each node
        self.exploration_weight = exploration_weight
        self.childrenObjects = dict()
        self.Pi = dict()

    def returnDist(self, node):
        # NodeHash = node.hash()
        if node.is_over == True:
            raise RuntimeError(f"choose called on terminal node {node}")
        visit_array = np.zeros(21)
        
        if node.nodeType == "max": #player2's turn to play (in my setting, player 2 starts) child[3] = p2_actions
            for child in self.childrenObjects[node]:
                # input(child.p2_actions[0][0])
                if child.p2_actions[0][0] == "attack":
                    for unit in node.unitList:
                        if unit.ownerID ==1: #!=player 2
                            pos = unit.x*4 + unit.y
                            visit_array[pos] = self.N[child]

                elif child.p2_actions[0][0] == "left":
                    visit_array[16] = self.N[child]
                elif child.p2_actions[0][0] == "right":
                    visit_array[17] = self.N[child]
                elif child.p2_actions[0][0] == "up":
                    visit_array[18] = self.N[child]
                elif child.p2_actions[0][0] == "down":
                    visit_array[19] = self.N[child]
                else:
                    visit_array[20] = 1
                    assert 10!=10
        
        elif node.nodeType == "min": #player1's turn to play child[2] = p1_actions
            for child in self.childrenObjects[node]:
                # input(child.p1_actions[0][0])
                if child.p1_actions[0][0] == "attack": #[1] is the target
                    for unit in node.unitList:
                        if unit.ownerID ==2:
                            pos = unit.x*4 + unit.y
                            visit_array[pos] = self.N[child]

                elif child.p1_actions[0][0] == "left":
                    visit_array[16] = self.N[child]
                elif child.p1_actions[0][0] == "right":
                    visit_array[17] = self.N[child]
                elif child.p1_actions[0][0] == "up":
                    visit_array[18] = self.N[child]
                elif child.p1_actions[0][0] == "down":
                    visit_array[19] = self.N[child]  
                else:
                    visit_array[20] = 1
                    assert 11!=11
        try:
            visit_array /= sum(visit_array)
        except:
            print(visit_array)
            print(node.hash())
            print(self.childrenObjects[node])
            input("THIS HAPPENED")
            assert 12!=12
        return visit_array
        
    def choose(self, node):
        "Choose the best successor of node. (Choose a move in the game)"
        # print("THIS IS THE NODE THAT WANTS TO CHOOSE SOMETHING")
        # node.Draw()
        print("RAFT TOO CHOOSE\n*****************\n\n")

        if node.is_over == True:
            raise RuntimeError(f"choose called on terminal node {node}")

        if node not in self.childrenObjects:
            input("HEEERREEE")
            assert 3!=3
            return node.find_random_child()

        def score(n):
            if self.N[n] == 0:
                return float("-inf")  # avoid unseen moves
            return self.Q[n] / self.N[n]  # average reward

        count = 0
        for each in self.childrenObjects[node]:
            print(self.N[each])
            print(self.Q[each])
            if score(each) == float("-inf"):
                count += 1
        if count == len(self.childrenObjects[node]):
            print("WOWWWWW")
            # assert 20!=20
        best_board = max(self.childrenObjects[node], key=score)
        return best_board
        # return best_board.p2_actions


    def do_rollout(self, node, model):
        "Make the tree one layer better. (Train for one iteration.)"
        print("AVVALE ROLLOUT")
        terminal_set = set()
        root = node
        path = self._select(node, terminal_set)
        print("AFTER SELECT")
        print(path)
        leaf = path[-1]
        reward = self._expand(leaf, model, root)
        if leaf.is_over == True:
            terminal_set.add(leaf)
            print("LEAF ADDED TO TERMINAL NODES")
        print("reward", reward)
        try:
            if np.isnan(reward):
                assert 100!=100
        except:
            input()
        if reward: # if already expanded, then do nothing?
            self._backpropagate(path, reward)
        
        print("******************\n")
    


    def _select(self, node, terminal_set): # a probable error happens when you are not expanding the tree that much and i cuases you to want to select a node that it is not still in the childreb, (but it isnt a terminal or )
        #solution: add a if statment, check if the node is in self.childrenobjects, otherwise, (kolan moshkele) fek kon A baz shode shode b o c -> b 5 ta child dare c d e f g - yekish expand nashode baghiash shode, baad oon yeki choose mishe chon baghie manfi budan ya harchi
        "Find an unexplored descendent of `node`"
        path = []
        # print("PLAYER IN SELECT")
        # print(player)
        while True:
            path.append(node)
            if node not in self.childrenObjects or not self.childrenObjects[node]:
                return path
        
            unexplored = self.childrenObjects[node] - self.childrenObjects.keys()#- terminal_set
            if unexplored:
                n = unexplored.pop()
                path.append(n)
                return path
            
            node = self._uct_select(node)  # descend a layer deeper
            # print("UCT - NOW AFTER ONE LAYER DEEPENING\nNEW NODE:\n")
            # print(node.hash())
            
    
    def mask_probability_vector(self, allIndexes, P):
        mask = np.zeros_like(P)
        mask[allIndexes] = 1
        masked_probabilities = P * mask
        normalization_factor = np.sum(masked_probabilities)
        normalized_probabilities = masked_probabilities / normalization_factor
        return normalized_probabilities


    def create_dataset(self, board):
        data = np.zeros((2, 4, 4))
        fourth_layer = np.zeros((1, 4, 4))
        if board.nodeType == "max":
            for eachU in board.unitList:
                if eachU.ownerID == 2:
                    data[0][eachU.x][eachU.y] = 1
                    fourth_layer[0][eachU.x][eachU.y] = eachU.hitpoints
                else:
                    data[1][eachU.x][eachU.y] = 1
                    fourth_layer[0][eachU.x][eachU.y] = eachU.hitpoints

            third_layer = np.zeros((1,4,4)) #Player turn to play - o for p2 (player1)
            data = np.concatenate((data,third_layer, fourth_layer))


        elif board.nodeType == "min":
            for eachU in board.unitList:
                if eachU.ownerID == 2:
                    data[0][eachU.x][eachU.y] = 1
                    fourth_layer[0][eachU.x][eachU.y] = eachU.hitpoints
                else:
                    data[1][eachU.x][eachU.y] = 1
                    fourth_layer[0][eachU.x][eachU.y] = eachU.hitpoints

            third_layer = np.ones((1,4,4))
            data = np.concatenate((data,third_layer, fourth_layer))

        return data


    def _expand(self, node, model, root):
        "Update the `children` dict with the children of `node`"

        if node in self.childrenObjects:
            print("ALREADY EXPANDED\n")
            assert 14!=14
            return  # already expanded
    
        data = self.create_dataset(node)
        data = np.array(data).reshape(1, 4, 4, 4)
        output = model.model.predict(data)
        # print(output)
        p = output[0][0]
        v = output[1][0][0]
        # print(p , v)

        try:
            childernObjs, _, allIndexes = node.find_children(p)
        except:
            print(node.hash())
            print("INJA")
            assert 2!=2
            input()
        
        # print(allIndexes)
        #CALL NN and expand the node using P 

        if node.is_over == True:
            # if node.winner ==2:
            if root.nodeType == "max":
                print(self.Pi[node])
                print("#################\n\n")
                if node.winner == 1:
                    value = -1
                if node.winner == 2:
                    value = 1
                if node.winner == 0.5:
                    value = 0
            elif root.nodeType == "min":
                print(self.Pi[node])
                print("#################\n\n")
                if node.winner == 1:
                    value = 1
                if node.winner == 2:
                    value = -1
                if node.winner == 0.5:
                    value = 0
            print(node.hash())
            # assert 17!=17
            # input
            return value #AS V
        

        if allIndexes == []:
            print(p)
            print(node.hash())
            assert 5!=5
        
        newP = self.mask_probability_vector(allIndexes, p)
        newP = np.array(newP)
        self.childrenObjects[node] = childernObjs

        i = 0
        for each in childernObjs:
            # print(each.hash())
            # each.Draw()
            self.Pi[each] = newP[i]
            i += 1
            # print("****\n")
        # input()
        print("CHILDREN FOUND and Ps assigned") #TODO: assing 1 to the root node
        return v
        
    def _backpropagate(self, path, reward):
        "Send the reward back up to the ancestors of the leaf"
        for node in reversed(path):
            self.N[node] += 1
            self.Q[node] += reward
            # reward *= -1  # 1 for me is 0 for my enemy, and vice versa

    def _uct_select(self, node):
        "Select a child of node, balancing exploration & exploitation"
        # nodeHash = node.hash()
        # All children of node should already be expanded:
        assert all(n in self.childrenObjects for n in self.childrenObjects[node])
        log_N_vertex = math.log(self.N[node]) 
        def uct(n): # if N = 0-> ignore the first term
            "Upper confidence bound for trees"
            return (self.Q[n] / self.N[n]) + self.exploration_weight * self.Pi[n] * math.sqrt(
                log_N_vertex)/ (self.N[n]+1)

        def uctMin(n):
            "Upper confidence bound for trees"
            return (self.Q[n] / self.N[n]) - self.exploration_weight * self.Pi[n] * math.sqrt(
                log_N_vertex)/ (self.N[n]+1)

        if node.nodeType =="max":
            print("UCT MAX")
            bestobj = max(self.childrenObjects[node], key=uct)
            return bestobj

    
        elif node.nodeType == "min":
            print("UCT MIN")
            bestobj = min(self.childrenObjects[node], key=uctMin)
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
