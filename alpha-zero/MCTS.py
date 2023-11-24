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

    def __init__(self, exploration_weight=1):
        self.Q = defaultdict(int)  # total reward of each node
        self.N = defaultdict(int)  # total visit count for each node
        self.children = dict()  # children of each node
        self.exploration_weight = exploration_weight
        self.childrenObjects = dict()
        self.Pi = dict()

    def returnDist(self, node):
        NodeHash = node.hash()
        if NodeHash[7] == True:
            raise RuntimeError(f"choose called on terminal node {NodeHash}")
        visit_array = np.zeros(21)
        
        if node.nodeType == "max": #player2's turn to play (in my setting, player 2 starts) child[3] = p2_actions
            for child in self.children[NodeHash]:
                
                if child[3][0] == "attack":
                    for unit in node.unitList:
                        if unit.ownerID ==1:
                            pos = unit.x*4 + unit.y
                            visit_array[pos] = self.N[child]

                elif child[3][0] == "left":
                    visit_array[16] = self.N[child]
                elif child[3][0] == "right":
                    visit_array[17] = self.N[child]
                elif child[3][0] == "up":
                    visit_array[18] = self.N[child]
                elif child[3][0] == "down":
                    visit_array[19] = self.N[child]
        
        elif node.nodeType == "min": #player1's turn to play child[2] = p1_actions
            for child in self.children[NodeHash]:
                if child[2][0] == "attack": #[1] is the target
                    for unit in node.unitList:
                        if unit.ownerID ==2:
                            pos = unit.x*4 + unit.y
                            visit_array[pos] = self.N[child]

                elif child[2][0] == "left":
                    visit_array[16] = self.N[child]
                elif child[2][0] == "right":
                    visit_array[17] = self.N[child]
                elif child[2][0] == "up":
                    visit_array[18] = self.N[child]
                elif child[2][0] == "down":
                    visit_array[19] = self.N[child]  

                    
        print(visit_array)
        visit_array /= sum(visit_array)
        print(visit_array)
        return visit_array
        
    def choose(self, node):
        "Choose the best successor of node. (Choose a move in the game)"
        # print("THIS IS THE NODE THAT WANTS TO CHOOSE SOMETHING")
        # node.Draw()
        print("RAFT TOO CHOOSE")
        NodeHash = node.hash()
        if NodeHash[7] == True:
            raise RuntimeError(f"choose called on terminal node {NodeHash}")

        if NodeHash not in self.children:
            input("HEEERREEE")
            assert 3!=3
            return node.find_random_child()

        def score(n):
            if self.N[n] == 0:
                return float("-inf")  # avoid unseen moves
            return self.Q[n] / self.N[n]  # average reward


        # best_board = max(self.children[NodeHash], key=score)      
        best_board = max(self.childrenObjects[node], key=score)
        return best_board
        # return best_board.p2_actions


    def do_rollout(self, node, model):
        "Make the tree one layer better. (Train for one iteration.)"
        root = node
        path = self._select(node)
        print("AFTER SELECT")
        print(path)
        leaf = path[-1]
        reward = self._expand(leaf, model, root)
        print("reward", reward)
        if reward: # if already expanded, then do nothing?
            self._backpropagate(path, reward)
        
        print("******************\n")
    


    def _select(self, node): # a probable error happens when you are not expanding the tree that much and i cuases you to want to select a node that it is not still in the childreb, (but it isnt a terminal or )
        #solution: add a if statment, check if the node is in self.childrenobjects, otherwise, (kolan moshkele) fek kon A baz shode shode b o c -> b 5 ta child dare c d e f g - yekish expand nashode baghiash shode, baad oon yeki choose mishe chon baghie manfi budan ya harchi
        "Find an unexplored descendent of `node`"
        path = []
        # print("PLAYER IN SELECT")
        # print(player)
        while True:
            path.append(node)
            if node.hash() not in self.children or not self.children[node.hash()]:
                return path

            unexplored = self.childrenObjects[node] - self.childrenObjects.keys()
            flag = True
            if unexplored:
                n = unexplored.pop()
                # if n.hash() in self.children.keys():
                #     print("HAST TOO IN")
                #     flag = False
                # if flag:
                path.append(n)
                return path
            print("WANTS TO USE UCT")
            node = self._uct_select(node)  # descend a layer deeper
            print("UCT - NOW AFTER ONE LAYER DEEPENING\nNEW NODE:\n")
            print(node.hash())
            
    
    def mask_probability_vector(self, allIndexes, P):
        mask = np.zeros_like(P)
        mask[allIndexes] = 1
        masked_probabilities = P * mask
        normalization_factor = np.sum(masked_probabilities)
        normalized_probabilities = masked_probabilities / normalization_factor
        return normalized_probabilities


    def create_dataset(self, board):
        data = np.zeros((4,4, 2))
        for eachU in board.unitList:
            if eachU.ownerID == 2:
                data[eachU.x][eachU.y] = [1, 0]
            else:
                data[eachU.x][eachU.y] = [0, 1]
        return data


    def _expand(self, node, model, root):
        "Update the `children` dict with the children of `node`"

        # if node.hash() in self.children:
        #     print(node.hash())
        #     print("ALREADY EXPANDED/game is over")
        #     # input()
        #     # assert node.is_over!=True
        #     return  # already expanded
        
        data = self.create_dataset(node)
        # print(data)
        data = np.array(data).reshape(1, 4, 4, 2)
        output = model.model.predict(data)

        p = output[0][0]
        v = output[1][0][0]
        print(p , v)

        

        try:
            childernObjs, childrenHash, allIndexes = node.find_children(p)
        except:
            print(node.hash())
            print("INJA")
            assert 2!=2
            input()

        #CALL NN and expand the node using P 
        if node.is_over == True:
            # if node.winner ==2:
            if root.nodeType == "max":
                if node.winner == 1:
                    value = -1
                if node.winner == 2:
                    value = 1
                if node.winner == 0.5:
                    value = 0.5
            elif root.nodeType == "min":
                if node.winner == 1:
                    value = 1
                if node.winner == 2:
                    value = -1
                if node.winner == 0.5:
                    value = 0.5
            return value #AS V
        
        newP = self.mask_probability_vector(allIndexes, p)
        print(allIndexes)
        if allIndexes == []:
            print(childernObjs, childrenHash)
        print(newP)
        newP = np.array(newP)
        if np.isnan(newP).all()==True:
            assert 1!=1
        self.children[node.hash()] = childrenHash 
        self.childrenObjects[node] = childernObjs
        i = 0
        for childHash in childrenHash:
            self.Pi[childHash] = newP[i]
            i += 1

        print((childrenHash))
        print("CHILDREN FOUND and Ps assigned") #TODO: assing 1 to the root node
        return v
        
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
        def uct(n): # if N = 0-> ignore the first term
            "Upper confidence bound for trees"
            return (self.Q[n] / self.N[n]) + self.exploration_weight * self.Pi[n] * math.sqrt(
                log_N_vertex)/ (self.N[n]+1)


        def uctObj(n):
            "Upper confidence bound for trees"
            n = n.hash()
            print(n)
            return (self.Q[n] / self.N[n]) + self.exploration_weight * self.Pi[n] * math.sqrt(
                log_N_vertex)/ (self.N[n]+1)


        def uctMin(n):
            "Upper confidence bound for trees"
            return (self.Q[n] / self.N[n]) - self.exploration_weight * self.Pi[n] * math.sqrt(
                log_N_vertex)/ (self.N[n]+1)


        def uctObjMin(n):
            "Upper confidence bound for trees"
            n = n.hash()
            return (self.Q[n] / self.N[n]) - self.exploration_weight * self.Pi[n] * math.sqrt(
                log_N_vertex)/ (self.N[n]+1)


        if node.nodeType =="max":
            print("UCT MAX")
            bestHash = max(self.children[nodeHash], key=uct)
            bestobj = max(self.childrenObjects[node], key=uctObj)
            if bestobj.hash() != bestHash:
                bestobj.Draw()
                print(bestHash)
                print(bestobj.hash())
                print("SHOULDN't have happened")
                assert 1!=1
            return bestobj

    
        elif node.nodeType == "min":
            print("UCT MIN")
            bestHash = min(self.children[nodeHash], key=uctMin)
            bestobj = min(self.childrenObjects[node], key=uctObjMin)
            if bestobj.hash() != bestHash:
                bestobj.Draw()
                print(bestHash)
                print("SHOULDN't have happened")
                assert 1!=1
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
